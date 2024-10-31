pip install -r requirements.txt
mkdir checkpoints
python3 export_onnx.py --origin_model /root/data/checkpoints/alexnet.pth --output_model checkpoints/alexnet.onnx