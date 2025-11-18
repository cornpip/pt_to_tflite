"""
간단한 MobileNet v2 TFLite 추론 예제.
입력: 224x224 RGB 이미지 (NHWC), ImageNet 정규화 적용.
"""

import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from urllib.request import urlretrieve
import urllib.request


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SAMPLE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
SAMPLE_PATH = "./sample_data/dog.jpg"
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
IMAGENET_LABELS_PATH = "./sample_data/imagenet_classes.txt"


def preprocess(image_path: str, size: int = 224) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr[np.newaxis, ...]  # NHWC with batch 1


def run_tflite(model_path: str, input_data: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ensure dtype matches model expectation
    tensor = input_data.astype(input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output[0]


def topk(preds: np.ndarray, k: int = 5):
    idx = preds.argsort()[-k:][::-1]
    return list(zip(idx, preds[idx]))


def ensure_sample_image(path: str = SAMPLE_PATH, url: str = SAMPLE_URL) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"[sample] downloading sample image to {path}")
        urlretrieve(url, path)
    else:
        print(f"[sample] using existing sample image {path}")
    return path


def ensure_imagenet_labels(path: str = IMAGENET_LABELS_PATH, url: str = IMAGENET_LABELS_URL):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"[labels] downloading imagenet labels to {path}")
        urlretrieve(url, path)
    else:
        print(f"[labels] using existing labels {path}")
    with open(path, "r") as f:
        labels = [line.strip() for line in f]
    return labels


def parse_args():
    parser = argparse.ArgumentParser(description="Run MobileNet v2 TFLite on one image")
    parser.add_argument("--model", default="./tflite/mobilenet_v2.tflite", help="TFLite 모델 경로")
    parser.add_argument("--image", help="입력 이미지 경로 (지정하지 않으면 샘플 이미지 다운로드/사용)")
    parser.add_argument("--topk", type=int, default=5, help="Top-K 출력 (기본 5)")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image or ensure_sample_image()
    labels = ensure_imagenet_labels()
    data = preprocess(image_path, size=224)
    logits = run_tflite(args.model, data)
    probs = tf.nn.softmax(logits).numpy()
    for cls_idx, score in topk(probs, k=args.topk):
        name = labels[cls_idx] if cls_idx < len(labels) else "unknown"
        print(f"{cls_idx}: {score:.4f} ({name})")


if __name__ == "__main__":
    main()
