# CenterPoint (IGIE)

## Model Description

CenterPoint is a 3D object detection framework that represents objects as points. This entry runs **PointPillars** as a single IGIE FP16 engine (`pp_e2e_fp16_ixinfer.so`) on nuScenes mini.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.5.0 | 26.06 |

## Model Preparation

### Prepare Resources

**Dataset (public):** register at [nuScenes](https://www.nuscenes.org/nuscenes), download `v1.0-mini.tgz` (~4GB), extract so that:

```text
data/nuScenes/
├── samples/
├── sweeps/
├── maps/
└── v1.0-mini/
```

**Checkpoint (public):** download PointPillars `latest.pth` from

- <https://drive.google.com/drive/folders/1K_wHrBo6yRSG7H7UUjKI4rPnyEA8HvOp>

and place it at `CenterPoint/latest.pth`.

**IGIE:** install / build IGIE so that `import tvm` works, then set `TVM_HOME` to the IGIE root (directory that contains `python/` and `build/libtvm.so`). IXUCA SDK environments typically already provide CoreX runtime via `source /opt/sw_home/enable`.

### Install Dependencies

```bash
# Always work from this directory:
cd models/cv/object_detection/centerpoint/igie
source /opt/sw_home/enable

# Clone upstream (do not vendor the full tree in git)
git clone https://github.com/tianweiy/CenterPoint.git

# Apply DeepSpark IGIE patches (use rsync; plain `cp -a` may prompt on overwrite)
rsync -a adapt/ CenterPoint/
cd CenterPoint
bash apply_compat.sh   # Py3.10 / torch.load compatibility

pip3 install -r requirements.txt
bash setup.sh   # builds det3d CUDA ops: dcn, iou3d_nms
```

> PointPillars does not need `spconv`. Related warnings can be ignored.

### Build annotations

```bash
# still inside CenterPoint/
export PYTHONPATH=${PWD}:${PYTHONPATH}
export NUSCENES_PATH=${PWD}/data/nuScenes

python3 tools/create_data.py nuscenes_data_prep \
  --root_path=${NUSCENES_PATH} \
  --version="v1.0-mini" \
  --nsweeps=10
```

Required file:

```text
data/nuScenes/infos_val_10sweeps_withvelo_filter_True.pkl
```

### Model Conversion

```bash
# inside CenterPoint/
export TVM_HOME=/path/to/igie          # REQUIRED
export PYTHONPATH=${PWD}:${TVM_HOME}/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${TVM_HOME}/build:/opt/sw_home/local/corex/lib64:${LD_LIBRARY_PATH}
export NUSCENES_PATH=${PWD}/data/nuScenes
export CUDA_VISIBLE_DEVICES=0          # set to your GPU

./run_igie.sh build
# -> pp_e2e_fp16_ixinfer.so
```

`./run_igie.sh` will also `source /opt/sw_home/enable` and append PyTorch `lib/` to `LD_LIBRARY_PATH` (needed for `iou3d_nms`).

## Model Inference

Keep the same env vars as in **Model Conversion** (`TVM_HOME` is required).

### FP16

From `models/cv/object_detection/centerpoint/igie/`:

```bash
export TVM_HOME=/path/to/igie
export CUDA_VISIBLE_DEVICES=0
bash scripts/infer_centerpoint_fp16_accuracy.sh
bash scripts/infer_centerpoint_fp16_performance.sh
```

Or inside `CenterPoint/`:

```bash
./run_igie.sh accuracy      # nuScenes mini mAP / NDS (+ dataset throughput in log)
./run_igie.sh performance   # single-frame IGIE latency / FPS
```

## Model Results

nuScenes **v1.0-mini** val (**81** samples), FP16 single SO:

| Model       | BatchSize | Precision | Dataset FPS | Single-frame FPS (wall) | mAP    | NDS    |
| :---------: | :-------: | :-------: | :---------: | :---------------------: | :----: | :----: |
| CenterPoint | 1         | FP16      | ~6.8–7.0    | ~55–60                  | 0.4165 | 0.4855 |

- **Dataset FPS** comes from `./run_igie.sh accuracy` log line `inference done: 81 samples in ...`.
- **Single-frame FPS** comes from `./run_igie.sh performance` (depends on `num_pillars` / GPU).
- Acceptable drift vs table: about ±0.003 on mAP/NDS.

## References

- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [nuScenes](https://www.nuscenes.org/nuscenes)
- [Paper](https://arxiv.org/abs/2006.11275)
