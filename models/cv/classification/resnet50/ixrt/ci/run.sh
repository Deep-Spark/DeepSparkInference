pip install -r requirements.txt
mkdir checkpoints
python3 export_onnx.py --origin_model resnet50.pth --output_model checkpoints/resnet50.onnx