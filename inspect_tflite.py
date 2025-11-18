import argparse
from pathlib import Path

import tensorflow as tf


def inspect_tflite(model_path: Path) -> None:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("===== INPUT DETAILS =====")
    for i, inp in enumerate(input_details):
        print(f"[Input {i}]")
        print("  name :", inp["name"])
        print("  shape:", inp["shape"])
        print("  dtype:", inp["dtype"])
        print("  quantization:", inp["quantization"])
        print("  shape_signature:", inp.get("shape_signature", "N/A"))
        print()

    print("===== OUTPUT DETAILS =====")
    for i, out in enumerate(output_details):
        print(f"[Output {i}]")
        print("  name :", out["name"])
        print("  shape:", out["shape"])
        print("  dtype:", out["dtype"])
        print("  quantization:", out["quantization"])
        print("  shape_signature:", out.get("shape_signature", "N/A"))
        print()


def main():
    parser = argparse.ArgumentParser(description="Inspect a TFLite model")
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the .tflite model file",
    )
    args = parser.parse_args()

    inspect_tflite(args.model.expanduser())


if __name__ == "__main__":
    main()
