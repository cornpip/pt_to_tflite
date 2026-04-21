# convert_to_tflite

A collection of scripts for converting PyTorch models to TFLite and doing quick checks on the converted outputs.

## Docker

```bash
docker pull cornpip77/tf_213_converter:latest
docker run -it --gpus all -v ${pwd}:/workspace cornpip77/tf_213_converter /bin/bash
```

If you want to use the Ultralytics image for YOLO:

```bash
docker pull ultralytics/ultralytics:latest
docker run -it --gpus all -v ${pwd}:/workspace ultralytics/ultralytics:latest /bin/bash
```

## MobileNet v2 Example

Run the full example from weight download to TFLite export:

```bash
python torch_to_tflite.py --example_mobilenet_v2
```

Check PyTorch inference output:

```bash
python mobilenet_v2_torch_infer.py \
  --image sample_data/dog.jpg \
  --pt_path ./pt/mobilenet_v2-b0353104.pth \
  --topk 5
```

Check TFLite inference output:

```bash
python tflite_mobilenet_v2_infer.py \
  --image sample_data/dog.jpg \
  --model ./tflite/mobilenet_v2.tflite \
  --topk 5
```

Inspect the TFLite model:

```bash
python inspect_tflite.py ./tflite/mobilenet_v2.tflite
```

## Generic PyTorch Conversion

Basic example:

```bash
python torch_to_tflite.py \
  --model resnet50 \
  --pt_path ./pt/your_model.pth \
  --num_classes 5 \
  --input_height 224 \
  --input_width 224 \
  --result_name my_model
```

Custom head example:

```bash
python torch_to_tflite.py \
  --model custom_head \
  --backbone efficientnet_b4 \
  --pt_path ./pt/custom_head.pth \
  --num_classes 3 \
  --input_height 380 \
  --input_width 380 \
  --result_name custom_head_model
```

## YOLO11 to TFLite

Use a YOLO11 `.pt` file:

```bash
python yolo11_to_tflite/yolo11_to_tflite.py yolo11_to_tflite/yolo11n.pt
```

Behavior:

- The script always requires an input `.pt` file path.
- Output files are generated according to Ultralytics export defaults.
- Example: `yolo11n.pt` -> `yolo11n_saved_model/`, `yolo11n_float32.tflite`
