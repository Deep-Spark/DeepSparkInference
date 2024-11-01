pip install -r requirements.txt
mkdir -p checkpoints
ln -s /root/data/checkpoints/yolov8.pt yolov8.pt
python3 export.py --weight yolov8.pt --batch 32
onnxsim ./yolov8.onnx ./checkpoints/yolov8.onnx