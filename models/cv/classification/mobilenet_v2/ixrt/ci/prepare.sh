pip install -r requirements.txt
mkdir checkpoints
python3 export_onnx.py --origin_model /root/data/checkpoints/mobilenet_v2.pth --output_model checkpoints/mobilenet_v2.onnx