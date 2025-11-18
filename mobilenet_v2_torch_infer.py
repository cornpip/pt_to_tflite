"""
MobileNet v2 PyTorch 추론 예제 (다운로드한 ImageNet 가중치 사용).
이미지 전처리와 라벨 매핑까지 한 번에 수행한다.
"""

import argparse
import os
from typing import List, Tuple
from urllib.request import urlretrieve

import torch
from PIL import Image
from torchvision import models, transforms


MOBILENET_V2_URL = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
DEFAULT_PT_PATH = "./pt/mobilenet_v2-b0353104.pth"
SAMPLE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
SAMPLE_PATH = "./sample_data/dog.jpg"
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
IMAGENET_LABELS_PATH = "./sample_data/imagenet_classes.txt"


def ensure_file(path: str, url: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"[download] {url} -> {path}")
        urlretrieve(url, path)
    else:
        print(f"[reuse] {path}")
    return path


def load_model(pt_path: str, device: torch.device) -> torch.nn.Module:
    state_dict = torch.load(pt_path, map_location="cpu")
    model = models.mobilenet_v2()
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


def preprocess(img_path: str) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    return tfm(img).unsqueeze(0)  # add batch dim


def topk(logits: torch.Tensor, k: int = 5) -> List[Tuple[int, float]]:
    probs = torch.softmax(logits, dim=1)
    scores, idx = probs.topk(k)
    return [(int(i), float(s)) for i, s in zip(idx[0], scores[0])]


def load_labels(path: str) -> list:
    with open(path, "r") as f:
        return [line.strip() for line in f]


def parse_args():
    p = argparse.ArgumentParser(description="Run MobileNet v2 (PyTorch) on an image")
    p.add_argument("--pt_path", default=DEFAULT_PT_PATH, help=".pth 가중치 경로")
    p.add_argument("--image", help="입력 이미지 경로(없으면 샘플 이미지 다운로드)")
    p.add_argument("--topk", type=int, default=5, help="Top-K 출력")
    p.add_argument("--use_cpu", action="store_true", help="GPU가 있어도 CPU 사용")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print(f"[device] {device}")

    pt_path = ensure_file(args.pt_path, MOBILENET_V2_URL)
    img_path = args.image or ensure_file(SAMPLE_PATH, SAMPLE_URL)
    labels_path = ensure_file(IMAGENET_LABELS_PATH, IMAGENET_LABELS_URL)

    model = load_model(pt_path, device)
    inp = preprocess(img_path).to(device)
    with torch.no_grad():
        logits = model(inp)
    labels = load_labels(labels_path)

    for idx, score in topk(logits, k=args.topk):
        name = labels[idx] if idx < len(labels) else "unknown"
        print(f"{idx}: {score:.4f} ({name})")


if __name__ == "__main__":
    main()
