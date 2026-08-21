# BEVFormer (IGIE)

## Model Description

BEVFormer learns unified Bird's-Eye-View features with spatiotemporal transformers for camera-only 3D detection. This entry runs a **three-stage FP16 IGIE stack** (backbone / encoder / decoder) on nuScenes mini.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.5.0 | 26.06 |

> IGIE must include Modulated DCN / multi-scale deform attention rewrite passes (IXUCA 4.5+ with matching IGIE). After setup run `bash tools/igie/check_repro_prereqs.sh` (expect PASS).

## Model Preparation

### Prepare Resources

**Dataset (public):** register at [nuScenes](https://www.nuscenes.org/nuscenes), download `v1.0-mini.tgz` (~4GB), extract to:

```text
data/nuscenes/
├── samples/
├── sweeps/
├── maps/
└── v1.0-mini/
```

**Checkpoint (public):**

```bash
mkdir -p ckpts
wget https://github.com/michigan-traffic-lab/nuCarla/releases/download/v1.0/bevformer-base.pth \
  -O ckpts/bevformer-base.pth
```

> Weight is trained on **trainval**. Absolute mAP/NDS on **mini** is low by design; use **Model Results** as the acceptance target.

**IGIE:** set `TVM_HOME` to an IGIE root that contains `python/` and `build/libtvm.so`. CoreX runtime: `source /opt/sw_home/enable`.

### Install Dependencies

```bash
# From: models/cv/object_detection/bevformer/igie/
source /opt/sw_home/enable

git clone https://github.com/michigan-traffic-lab/nuCarla.git
# use rsync (plain `cp -a` may prompt on overwrite)
rsync -a adapt/ nuCarla/BEVFormer/
cd nuCarla/BEVFormer

# Public clone may vendor an incomplete mmcv/ tree — do not use it as the runtime mmcv.
# Prefer an Iluvatar mmcv-*.whl (contact admin), or build OpenMMLab mmcv v1.7.0 with ops:
#   git clone -b v1.7.0 https://github.com/open-mmlab/mmcv.git /tmp/mmcv && cd /tmp/mmcv
#   MMCV_WITH_OPS=1 pip3 install -e . --no-build-isolation
# If a local mmcv/ directory exists in this repo, rename it so it cannot shadow site-packages:
[[ -d mmcv ]] && mv mmcv mmcv-src

pip3 install --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2.git' || true
pip3 install -r requirements.txt
# Install plugin package (skip if dependency resolver fails; PYTHONPATH below is enough)
python3 setup.py develop --user 2>/dev/null \
  || python3 setup.py install --user 2>/dev/null \
  || true
export PYTHONPATH=${PWD}:${PYTHONPATH}
```

### Build annotations

```bash
# inside nuCarla/BEVFormer/
export PYTHONPATH=${PWD}:${PYTHONPATH}
python3 tools/create_data.py --version v1.0-mini
```

Required:

```text
data/nuscenes/nuscenes_infos_temporal_val.pkl
```

### Model Conversion

```bash
export TVM_HOME=/path/to/igie          # REQUIRED
export PYTHONPATH=${PWD}:${TVM_HOME}/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${TVM_HOME}/build:/opt/sw_home/local/corex/lib64:${LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0

bash tools/igie/check_repro_prereqs.sh
./run_igie.sh build
# installs export-friendly mmcv ops from tools/igie/patches/ into site-packages,
# then exports pt2 + builds three .so + numeric compare
```

Artifacts (local only):

- `bevformer_backbone_fp16_conv_only_ixinfer_NHWC_mdcn.so`
- `bevformer_encoder_fp16_ixinfer_NHWC_msdeform.so`
- `bevformer_decoder_fp16_ixinfer_NHWC_msdeform.so`

## Model Inference

Keep the same env (`TVM_HOME` required).

### FP16

From `models/cv/object_detection/bevformer/igie/`:

```bash
export TVM_HOME=/path/to/igie
export CUDA_VISIBLE_DEVICES=0
bash scripts/infer_bevformer_fp16_accuracy.sh
bash scripts/infer_bevformer_fp16_performance.sh
```

Or inside `nuCarla/BEVFormer/`:

```bash
./run_igie.sh accuracy       # mAP / NDS (+ dataset FPS in log)
./run_igie.sh performance    # single-frame IGIE latency (ms)
```

## Model Results

nuScenes **v1.0-mini** val (**81** samples), three-stage FP16:

| Model     | BatchSize | Precision | Dataset FPS | Single-frame (e2e) | mAP    | NDS    |
| :-------: | :-------: | :-------: | :---------: | :----------------: | :----: | :----: |
| BEVFormer | 1         | FP16      | ~0.9–1.0    | ~700 ms (~1.4 fps) | 0.0739 | 0.1166 |

- **Dataset FPS** from `./run_igie.sh accuracy` log: `inference done: 81 samples in ...`.
- **Single-frame** from `./run_igie.sh performance` (`igie_e2e_ms`).
- Tolerance: about ±0.001 on mAP/NDS.

## References

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [nuCarla (weights)](https://github.com/michigan-traffic-lab/nuCarla)
- [nuScenes](https://www.nuscenes.org/nuscenes)
- [Paper](https://arxiv.org/abs/2203.17270)
