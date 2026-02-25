# CenterNet (IGIE)

## Model Description

Three-dimensional objects are commonly represented as 3D boxes in a point-cloud. This representation mimics the well-studied image-based 2D bounding-box detection but comes with additional challenges. Objects in a 3D world do not follow any particular orientation, and box-based detectors have difficulties enumerating all orientations or fitting an axis-aligned bounding box to rotated objects. In this paper, we instead propose to represent, detect, and track 3D objects as points. Our framework, CenterPoint, first detects centers of objects using a keypoint detector and regresses to other attributes, including 3D size, 3D orientation, and velocity. In a second stage, it refines these estimates using additional point features on the object. In CenterPoint, 3D object tracking simplifies to greedy closest-point matching. The resulting detection and tracking algorithm is simple, efficient, and effective. CenterPoint achieved state-of-the-art performance on the nuScenes benchmark for both 3D detection and tracking, with 65.5 NDS and 63.8 AMOTA for a single model. On the Waymo Open Dataset, CenterPoint outperforms all previous single model method by a large margin and ranks first among all Lidar-only submissions.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 25.09 |

## Model Preparation

参考了[tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint.git)及[CarkusL/CenterPoint](https://github.com/CarkusL/CenterPoint.git)实现——其中前者为Centerpoint作者，并在此基础上，做了基于IGIE的推理适配。

由于在IGIE适配的过程中，对Centerpoint相关文件做了必要的改动，因此直接以改动后的源码形式给出。

# Env

运行环境配置。
```
cd ./CenterPoint
pip3 install -r requirements.txt

export PATH=/usr/local/corex-4.4.0/bin:$PATH
export CUDA_PATH=/usr/local/corex-4.4.0
export CUDA_HOME=/usr/local/corex-4.4.0
export LD_LIBRARY_PATH=/usr/local/corex-4.4.0/lib64:$LD_LIBRARY_PATH
bash setup.sh 
```


代码中用到det3d库，配置一下pythonpath，同时，要用到NuScenesEval，也要配置一下。
```
cd CenterPoint 
git clone https://github.com/tianweiy/nuscenes-devkit

export PYTHONPATH=/path/CenterPoint:${PYTHONPATH}
export PYTHONPATH=/path/CenterPoint/nuscenes-devkit:${PYTHONPATH}
```

# Dataset

这里使用nuscenes数据集，为了减少下载量，选择使用其v1.0-mini版本。

将上面解压后的数据集，放到CenterPoint/data/nuscenes目录下，如下：
```
CenterPoint# tree -L 1 data/nuscenes/
data/nuscenes/
├── LICENSE
├── maps
├── nuscenes_gt_database
├── samples
├── sweeps
└── v1.0-mini

5 directories, 1 file
```

使用*.pkl文件组织有用的数据信息，由于这里下载的是v1.0-mini版本，这里需要将version信息配置显式地配置一下，运行如下命令：
```
python3 tools/create_data.py nuscenes_data_prep --root_path=data/nuscenes --version="v1.0-mini" --nsweeps=10
```
处理完后，会多出几个*.pkl文件：
```
centerpoint# tree -L 1 data/nuscenes/
data/nuscenes/
├── dbinfos_train_10sweeps_withvelo.pkl
├── gt_database_10sweeps_withvelo
├── infos_train_10sweeps_withvelo_filter_True.pkl
├── infos_val_10sweeps_withvelo_filter_True.pkl
├── LICENSE
├── maps
├── nuscenes_gt_database
├── samples
├── sweeps
└── v1.0-mini

6 directories, 4 files
```

同时，需要配置一下nuscenes数据集位置
```
export NUSCENES_PATH=/path/CenterPoint/data/nuscenes
```

# Export Model

下载模型[trained model(latest.pth)](https://drive.google.com/drive/folders/1f8EHYqfHtP6kyNDlsTIG9Nbz_pJ0Cal9?usp=sharing)，需要注册账户。

这个latest.pth文件是可以加载推理的，可作为baseline供参考:
```
python3 tools/dist_test.py configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo_mini.py --work_dir dataset/nuscenes  --checkpoint ./latest.pth
```

CarkusL给出了导onnx的脚本以及onnx文件，但batch=1，且是服务于trt的，不能直接使用，需要重新走一遍导onnx模型的流程。
```
cd CenterPoint
python3 tools/export_pointpillars_onnx.py
onnxsim onnx_model/rpn.onnx onnx_model/rpn_opt.onnx
```

# build engine 

这里使用igie-exec编译，命令如下:
```
igie-exec --model_path onnx_model/rpn_opt.onnx --input input.1:4,64,512,512 --precision fp16 --engine_path rpn_opt.so --just_export True
```

# run inference 

沿用config配置文件形式，新添加了个`nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo_mini_igie.py`，并将igie engine配置到该文件中。运行如下命令完成基于数据集的e2e推理:
```
python3 tools/igie_test.py configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo_mini_igie.py --work_dir dataset/nuscenes  --checkpoint ./latest.pth
```

推理精度如下:
```
Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE   
car                     0.887   0.189   0.161   0.231   0.131   0.076 
truck                   0.658   0.157   0.182   0.335   0.120   0.066 
bus                     0.964   0.232   0.162   0.025   0.595   0.290 
trailer                 0.000   1.000   1.000   1.000   1.000   1.000 
construction_vehicle    0.000   1.000   1.000   1.000   1.000   1.000 
pedestrian              0.888   0.155   0.249   0.390   0.202   0.134 
motorcycle              0.502   0.237   0.264   0.814   0.049   0.000 
bicycle                 0.214   0.274   0.189   0.262   0.418   0.000 
traffic_cone            0.028   0.158   0.318   nan     nan     nan   
barrier                 0.000   1.000   1.000   1.000   nan     nan   
Evaluation nusc: Nusc v1.0-mini Evaluation
car Nusc dist AP@0.5, 1.0, 2.0, 4.0
77.82, 89.52, 92.78, 94.69 mean AP: 0.8870004403165079
truck Nusc dist AP@0.5, 1.0, 2.0, 4.0
61.80, 65.28, 66.69, 69.26 mean AP: 0.6575730264909234
construction_vehicle Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00, 0.00, 0.00, 0.00 mean AP: 0.0
bus Nusc dist AP@0.5, 1.0, 2.0, 4.0
87.92, 99.24, 99.24, 99.24 mean AP: 0.9641387276088109
trailer Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00, 0.00, 0.00, 0.00 mean AP: 0.0
barrier Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00, 0.00, 0.00, 0.00 mean AP: 0.0
motorcycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
44.41, 49.92, 52.62, 53.94 mean AP: 0.5022015213907609
bicycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
21.42, 21.42, 21.42, 21.42 mean AP: 0.21423507290205296
pedestrian Nusc dist AP@0.5, 1.0, 2.0, 4.0
85.16, 87.88, 89.93, 92.06 mean AP: 0.8876045828912109
traffic_cone Nusc dist AP@0.5, 1.0, 2.0, 4.0
2.76, 2.76, 2.76, 2.76 mean AP: 0.027558950834248364
```

# Scripts 

在./scripts目录中提供了验证精度和性能的脚本。
```
cd scripts

bash infer_centerpoint_fp16_accuracy.sh  
bash infer_centerpoint_fp16_performance.sh
```

# Model Results

| Model       | BatchSize | Precision | FPS  | mAP    | mATE   |
| ----------- | --------- | --------- | ---- | ------ | ------ |
| CenterPoint | 4         | fp16      | 14.9 | 0.4155 | 0.4432 |

