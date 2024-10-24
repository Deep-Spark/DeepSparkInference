pip install -r requirements.txt
mkdir checkpoints
python3 export_onnx.py --origin_model efficientnet_b0.pth --output_model checkpoints/efficientnet_b0.onnx