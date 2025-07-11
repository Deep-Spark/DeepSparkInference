# Conformer (IGIE)

## Model Description

Conformer is a speech recognition model proposed by Google in 2020. It combines the advantages of CNN and Transformer.
CNN efficiently extracts local features, while Transformer is more effective in capturing long sequence dependencies.
Conformer applies convolution to the Encoder layer of Transformer, enhancing the performance of Transformer in the ASR
(Automatic Speech Recognition) domain.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20211025_conformer_exp.tar.gz>

Dataset: <https://www.openslr.org/33/> to download the Aishell dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install sox sox-devel -y
## Ubuntu
apt install sox libsox-fmt-all -y

pip3 install -r requirements.txt
cd ctc_decoder/swig && bash setup.sh
cd ../../
```

### Model Conversion

```bash
tar -zxvf 20211025_conformer_exp.tar.gz

export PYTHONPATH=`pwd`/wenet:$PYTHONPATH

# Get Onnx Model
cd wenet
python3 wenet/bin/export_onnx_gpu.py                          \
    --config ../20211025_conformer_exp/train.yaml             \
    --checkpoint ../20211025_conformer_exp/final.pt           \
    --batch_size 24                                           \
    --seq_len 384                                             \
    --beam 4                                                  \
    --cmvn_file ../20211025_conformer_exp/global_cmvn         \
    --output_onnx_dir ../
cd ..

# Use onnxsim optimize onnx model
onnxsim encoder_bs24_seq384_static.onnx encoder_bs24_seq384_static_opt.onnx
python3 alter_onnx.py --batch_size 24 --path encoder_bs24_seq384_static_opt.onnx
```

## Model Inference

```bash
# Need to unzip aishell to the current directory. For details, refer to data.list
tar -zxvf aishell.tar.gz
```

### FP16

```bash
# Accuracy
bash scripts/infer_conformer_fp16_accuracy.sh
# Performance
bash scripts/infer_conformer_fp16_performance.sh
```

## Model Results

| Model     | BatchSize | Precision | FPS      | ACC   |
| :----: | :----: | :----: | :----: | :----: |
| Conformer | 32        | FP16      | 1940.759 | 95.29 |
