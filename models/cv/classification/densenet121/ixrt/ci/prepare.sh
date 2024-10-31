pip install -r requirements.txt
mkdir checkpoints
mkdir -p /root/.cache/torch/hub/checkpoints/
ln -s /root/data/checkpoints/densenet121.pth /root/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth
python3 export_onnx.py --output_model ./checkpoints/densenet121.onnx