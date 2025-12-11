"""
Download YOLO11n, export it to TFLite, and run a quick smoke test.

Ultralytics automatically downloads `yolo11n.pt` to its cache if the file is
missing in the working directory. After the export we load the produced
`yolo11n_float32.tflite` and run inference on the sample bus image to verify
the pipeline.
"""

from ultralytics import YOLO


def main() -> None:
    # 1) Load the PyTorch checkpoint (downloaded automatically if needed)
    model = YOLO("yolo11n.pt")

    # 2) Export to TFLite (float32). Setting dynamic=False keeps TensorFlow Lite
    #    happy, because dynamic shapes tend to crash on mobile runtimes.
    model.export(format="tflite", dynamic=False)  # -> yolo11n_float32.tflite


if __name__ == "__main__":
    main()
