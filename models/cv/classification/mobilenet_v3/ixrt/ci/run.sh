pip install -r requirements.txt
mkdir checkpoints
python3 export_onnx.py --origin_model mobilenet_v3.pth --output_model checkpoints/mobilenet_v3.onnx