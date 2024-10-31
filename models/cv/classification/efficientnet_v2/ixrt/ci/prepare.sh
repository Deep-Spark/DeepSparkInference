pip install -r requirements.txt
mkdir -p checkpoints
unzip /root/data/repos/pytorch-image-models-a852318b636a8.zip -d ./
cp ./export_onnx.py pytorch-image-models/timm/models
rm pytorch-image-models/timm/models/_builder.py
mv ./_builder.py pytorch-image-models/timm/models
mkdir -p /root/.cache/torch/hub/checkpoints/
ln -s /root/data/checkpoints/efficientnet_v2.pth /root/.cache/torch/hub/checkpoints/efficientnetv2_t_agc-3620981a.pth
cd pytorch-image-models/timm/ && python3 -m models.export_onnx --output_model ../../checkpoints/efficientnet_v2.onnx
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python