pip install -r requirements.txt
mkdir -p checkpoints
unzip /root/data/repos/mmpretrain-0.24.0.zip -d ./checkpoints/
python3 export_onnx.py   \
--config_file ./checkpoints/mmpretrain/configs/shufflenet_v1/shufflenet-v1-1x_16xb64_in1k.py  \
--checkpoint_file  /root/data/checkpoints/shufflenet_v1.pth \
--output_model ./checkpoints/shufflenet_v1.onnx