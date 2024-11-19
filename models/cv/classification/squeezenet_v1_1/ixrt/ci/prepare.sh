pip install -r requirements.txt
mkdir checkpoints
python3 export_onnx.py --origin_model /root/data/checkpoints/squeezenet_v1_1.pth --output_model checkpoints/squeezenet_v1_1.onnx