# CosyVoice2 (ixRT)

## Model Description

CosyVoice2-0.5B is a small speech model designed to understand and generate human-like speech. It can be used for tasks like voice assistants, text-to-speech, or voice cloning. With 0.5 billion parameters, it is lightweight and works well on devices with limited computing power. It focuses on natural-sounding voices and easy customization.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 | release/26.03 |
| MR-V100 | 4.3.0 | 25.12 | release/25.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.03`

## Model Preparation

### Prepare Resources

Pretrained model: <https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B>

### Install Dependencies

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ 
pip install onnxsim
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# If you failed to clone the submodule due to network failures, please run the following command until success
cd CosyVoice
git checkout 1fc843514689daa61b471f1bc862893b3a5035a7
git submodule update --init --recursive

# cp modify files
cp -rf ../cosyvoice ./
cp ../asset/zero_shot_reference.wav ./asset/
cp -r ../scripts ./
cp ../build_dynamic_engine.py ./
cp ../inference.py ./

mkdir -p pretrained_models
# download CosyVoice2-0.5B model into pretrained_models dir

onnxsim ./pretrained_models/CosyVoice2-0.5B/flow.decoder.estimator.fp32.onnx ./pretrained_models/CosyVoice2-0.5B/flow.decoder.estimator.fp32_sim.onnx

# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

## Model Inference

```bash
bash scripts/infer_cosyvoice2_fp16.sh
```

## References

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice/commit/1fc843514689daa61b471f1bc862893b3a5035a7)