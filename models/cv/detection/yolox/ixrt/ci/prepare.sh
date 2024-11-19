pip install -r requirements.txt
unzip /root/data/repos/yolox-f00a798c8bf59f43ab557a2f3d566afa831c8887.zip -d ./
ln -s /root/data/checkpoints/yolox_m.pth ./YOLOX/
cd YOLOX && python3 setup.py develop && python3 tools/export_onnx.py --output-name ../yolox.onnx -n yolox-m -c yolox_m.pth --batch-size 32
if [ "$1" = "nvidia" ]; then
    cd ../plugin && mkdir -p build && cd build && cmake .. -DUSE_TRT=1 && make -j12
else
    cd ../plugin && mkdir -p build && cd build && cmake .. -DIXRT_HOME=/usr/local/corex && make -j12
fi