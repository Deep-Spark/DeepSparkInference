# CenterNet (IGIE)

## Model Description

Three-dimensional objects are commonly represented as 3D boxes in a point-cloud. This representation mimics the well-studied image-based 2D bounding-box detection but comes with additional challenges. Objects in a 3D world do not follow any particular orientation, and box-based detectors have difficulties enumerating all orientations or fitting an axis-aligned bounding box to rotated objects. In this paper, we instead propose to represent, detect, and track 3D objects as points. Our framework, CenterPoint, first detects centers of objects using a keypoint detector and regresses to other attributes, including 3D size, 3D orientation, and velocity. In a second stage, it refines these estimates using additional point features on the object. In CenterPoint, 3D object tracking simplifies to greedy closest-point matching. The resulting detection and tracking algorithm is simple, efficient, and effective. CenterPoint achieved state-of-the-art performance on the nuScenes benchmark for both 3D detection and tracking, with 65.5 NDS and 63.8 AMOTA for a single model. On the Waymo Open Dataset, CenterPoint outperforms all previous single model method by a large margin and ranks first among all Lidar-only submissions.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 25.09 |

## Model Preparation

### Prepare Resources

模型的导出参考官方[https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/04-supported-codebases/mmdet3d.md](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/04-supported-codebases/mmdet3d.md)

这里使用openmmlab生态中的centerpoint模型，配置openmmlab环境比较复杂，主要是各packages之间的版本依赖关系。
默认使用python3.10，使用mim命令进行模型的导出。
```
# 安装mim命令
python3 -m pip install -U openmim

python3 -m mim install "mmdet==3.2.0"
python3 -m mim install "mmdet3d==1.3.0"
python3 -m mim install "mmengine==0.10.7"

# clone mmdeploy，切到1.3.0后，源码安装
git clone  https://github.com/open-mmlab/mmdeploy.git 
cd mmdeploy 
git checkout v1.3.0 --force 
python3 setup.py install 

# 下载模型产物，注意，当前在mmdeploy目录下面。
mim download mmdet3d --config centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d --dest .
export MODEL_CONFIG=centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py
export MODEL_PATH=centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth
export TEST_DATA=tests/data/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151612397179.pcd.bin
```

注意，这里要安装1.15.0的onnx及onnxruntime，不然导出时有报错：
```
pip3 install onnx==1.15.0
pip3 install onnxruntime==1.15.0 

# 最后一步，导出centerpoint end2end模型文件
python3 tools/deploy.py configs/mmdet3d/voxel-detection/voxel-detection_onnxruntime_dynamic.py $MODEL_CONFIG $MODEL_PATH $TEST_DATA --work-dir centerpoint
```
这样，我们就在centerpoint目录中得到end2end.onnx模型文件。

对end2end.onnx模型做onnxsim优化：
```
onnxsim end2end.onnx centerpoint_e2e_opt.onnx
```



