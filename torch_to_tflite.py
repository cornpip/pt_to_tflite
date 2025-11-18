import argparse
import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights, ResNet50_Weights
from torch.hub import download_url_to_file
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

MOBILENET_V2_URL = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"

# --------------------------------------------------------------------------- #
# 1) 모델 정의 영역
# --------------------------------------------------------------------------- #
class CustomHeadModel(nn.Module):
    """
    예시용 커스텀 모델.
    """
    
    def __init__(self, backbone_name: str = "efficientnet_b4", num_classes: int = 3, pretrained: bool = False):
        super().__init__()

        if "efficientnet" in backbone_name:
            base = getattr(models, backbone_name)(
                weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            )
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
        elif "resnet" in backbone_name:
            base = getattr(models, backbone_name)(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            in_features = base.fc.in_features
            base.fc = nn.Identity()
        else:
            raise ValueError(f"지원하지 않는 backbone: {backbone_name}")

        self.backbone = base
        self.bn = nn.BatchNorm1d(in_features)
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if feat.dim() == 4:
            feat = torch.flatten(feat, 1)
        feat = self.bn(feat)
        return self.head(feat)


def build_model(model_type: str, num_classes: int, backbone: str, use_pretrained_backbone: bool) -> nn.Module:
    if model_type == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_type == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_type == "efficientnet_b4":
        model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_type == "custom_head":
        return CustomHeadModel(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=use_pretrained_backbone,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


# --------------------------------------------------------------------------- #
# 2) 유틸 함수
# --------------------------------------------------------------------------- #
def load_state_dict_flexible(checkpoint_path: str, device: torch.device) -> dict:
    """키 이름이 조금 달라도 불러올 수 있도록 여유롭게 state_dict 를 읽는다."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    return checkpoint


def ensure_dirs() -> None:
    """결과물이 저장될 기본 폴더 생성."""
    for p in ["onnx", "saved_model", "tflite"]:
        os.makedirs(p, exist_ok=True)


def download_mobilenet_v2(pt_path: str) -> str:
    """공개 MobileNet v2 ImageNet 가중치(.pth) 다운로드."""
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    if os.path.exists(pt_path):
        print(f"[download] reuse existing {pt_path}")
        return pt_path
    print(f"[download] downloading mobilenet_v2 to {pt_path} ...")
    download_url_to_file(MOBILENET_V2_URL, pt_path, progress=True)
    return pt_path


# --------------------------------------------------------------------------- #
# 3) 변환 파이프라인
# --------------------------------------------------------------------------- #
def export_onnx(model: nn.Module, dummy_input: torch.Tensor, onnx_path: str) -> None:
    """PyTorch → ONNX"""
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )

def onnx_to_tf_nhwc(onnx_model_path: str, saved_model_dir: str) -> None:
    """
    ONNX → TensorFlow SavedModel
    TensorFlow 는 기본적으로 NHWC 를 기대하므로 입력 차원을 맞춰주는 서빙 함수를 다시 정의한다.
    """
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(saved_model_dir)

    model = tf.saved_model.load(saved_model_dir)
    concrete_func = model.signatures["serving_default"]
    input_tensor = concrete_func.inputs[0]
    input_shape = input_tensor.shape.as_list()  # [1, C, H, W]
    nhwc_shape = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]

    @tf.function(input_signature=[tf.TensorSpec(shape=nhwc_shape, dtype=tf.float32)])
    def new_serving_fn(inputs):
        nchw_input = tf.transpose(inputs, [0, 3, 1, 2])
        outputs = concrete_func(nchw_input)
        return outputs

    tf.saved_model.save(model, saved_model_dir, signatures={"serving_default": new_serving_fn})


def tf_to_tflite(saved_model_dir: str, tflite_path: str, optimize: bool = True) -> None:
    """TensorFlow SavedModel → TFLite"""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


# --------------------------------------------------------------------------- #
# 4) CLI & 메인 루틴
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TFLite")
    parser.add_argument("--input_height", type=int, default=224, help="더미 입력 이미지 높이 (기본: 224)")
    parser.add_argument("--input_width", type=int, default=224, help="더미 입력 이미지 너비 (기본: 224)")
    parser.add_argument("--num_classes", type=int, default=1000, help="최종 클래스 수 (기본: 1000)")
    parser.add_argument("--pt_path", type=str, default="./pt/model.pth", help="학습된 .pt 또는 .pth 파일 경로")
    parser.add_argument("--result_name", type=str, default="resnet50_model", help="출력 파일 이름")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b4", "mobilenet_v2", "custom_head"],
        help="백본 혹은 커스텀 모델 타입 (custom_head = 커스텀 헤드 예시)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b4",
        help="BaseTypeModel 에서 사용할 백본 이름 (efficientnet_b4, resnet50 등)",
    )
    parser.add_argument(
        "--use_pretrained_backbone",
        action="store_true",
        help="백본을 ImageNet 가중치로 초기화한 뒤 .pt 가중치를 로드",
    )
    parser.add_argument(
        "--example_mobilenet_v2",
        action="store_true",
        help="모바일넷 v2 ImageNet 가중치 자동 다운로드 후 224x224 입력으로 변환 예시 실행",
    )
    return parser.parse_args()


def configure_mobilenet_example(args: argparse.Namespace) -> None:
    args.model = "mobilenet_v2"
    args.input_height = 224
    args.input_width = 224
    args.num_classes = 1000
    args.result_name = "mobilenet_v2"
    args.pt_path = "./pt/mobilenet_v2-b0353104.pth"
    download_mobilenet_v2(args.pt_path)


def convert(args: argparse.Namespace, device: torch.device) -> None:
    ensure_dirs()

    # 1) 모델 준비 + 가중치 로드
    model = build_model(
        model_type=args.model,
        num_classes=args.num_classes,
        backbone=args.backbone,
        use_pretrained_backbone=args.use_pretrained_backbone,
    )
    state_dict = load_state_dict_flexible(args.pt_path, device=torch.device("cpu"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("[state_dict] missing:", missing)
    print("[state_dict] unexpected:", unexpected)
    model.eval().to(device)

    # 2) PyTorch → ONNX
    dummy_input = torch.randn(1, 3, args.input_height, args.input_width, device=device)
    onnx_model_path = f"./onnx/{args.result_name}.onnx"
    export_onnx(model, dummy_input, onnx_model_path)
    print(f"[onnx] saved to {onnx_model_path}")

    # 3) ONNX → TensorFlow SavedModel
    saved_model_dir = f"./saved_model/{args.result_name}"
    print("[tf] converting onnx → saved_model ...")
    onnx_to_tf_nhwc(onnx_model_path, saved_model_dir)
    print(f"[tf] saved_model ready at {saved_model_dir}")

    # 4) TensorFlow → TFLite
    tflite_model_path = f"./tflite/{args.result_name}.tflite"
    print("[tflite] converting saved_model → tflite ...")
    tf_to_tflite(saved_model_dir, tflite_model_path, optimize=True)
    print(f"[tflite] saved to {tflite_model_path}")


def main() -> None:
    args = parse_args()
    if args.example_mobilenet_v2:
        configure_mobilenet_example(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using {device}")
    convert(args, device)


if __name__ == "__main__":
    main()
