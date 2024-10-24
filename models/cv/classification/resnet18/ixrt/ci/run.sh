pip install -r requirements.txt
mkdir checkpoints
python3 export_onnx.py --origin_model resnet18.pth --output_model checkpoints/resnet18.onnx