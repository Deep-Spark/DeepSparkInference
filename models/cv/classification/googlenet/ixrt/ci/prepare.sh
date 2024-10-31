pip install -r requirements.txt
mkdir checkpoints
python3 export_onnx.py --origin_model /root/data/checkpoints/googlenet.pth --output_model checkpoints/googlenet.onnx