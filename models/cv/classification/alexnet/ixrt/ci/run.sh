pip install -r requirements.txt
mkdir checkpoints
python3 export_onnx.py --origin_model alexnet.pth --output_model checkpoints/alexnet.onnx