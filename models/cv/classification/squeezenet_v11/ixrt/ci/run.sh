pip install -r requirements.txt
mkdir checkpoints
python3 export_onnx.py --origin_model squeezenet_v11.pth --output_model checkpoints/squeezenet_v11.onnx