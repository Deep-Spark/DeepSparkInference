pip install -r requirements.txt
mkdir checkpoints
python3 export_onnx.py --origin_model /root/data/checkpoints/vgg16.pth --output_model checkpoints/resnet18.onnx