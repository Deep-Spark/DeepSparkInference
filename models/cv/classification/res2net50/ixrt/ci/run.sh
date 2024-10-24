pip install -r requirements.txt
mkdir checkpoints
python3 export_onnx.py --origin_model res2net50.pth --output_model checkpoints/res2net50.onnx