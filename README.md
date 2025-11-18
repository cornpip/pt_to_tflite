## Docker Script
```
docker pull cornpip77/tf_213_converter:latest

docker run -it --gpus all -v ${pwd}:/workspace cornpip77/tf_213_converter /bin/bash
```

## MobileNet Test Sciprt
```
// Run MobileNet v2 example (downloads ImageNet weights)
python torch_to_tflite.py --example_mobilenet_v2

// pytorch inference result check
python mobilenet_v2_torch_infer.py --image sample_data/dog.jpg --pt_path ./pt/mobilenet_v2-b0353104.pth --topk 5

// tflite inference result check
python tflite_mobilenet_v2_infer.py --image sample_data/dog.jpg --model ./tflite/mobilenet_v2.tflite --topk 5

// tflite inspect (any .tflite)
python inspect_tflite.py ./tflite/mobilenet_v2.tflite
```

## Test Script
```
// Convert your torch file
python torch_to_tflite.py \
  --model resnet50 \
  --pt_path ./pt/your_model.pth \
  --num_classes 5 \
  --input_height 224 \
  --input_width 224 \
  --result_name my_model

// Custom head example (efficientnet_b4 backbone)
python torch_to_tflite.py \
  --model custom_head \
  --backbone efficientnet_b4 \
  --pt_path ./pt/custom_head.pth \
  --num_classes 3 \
  --input_height 380 \
  --input_width 380 \
  --result_name custom_head_model
```
