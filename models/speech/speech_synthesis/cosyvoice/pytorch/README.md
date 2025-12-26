# CosyVoice2 (pytorch)

## Model Description

CosyVoice2-0.5B is a small speech model designed to understand and generate human-like speech. It can be used for tasks like voice assistants, text-to-speech, or voice cloning. With 0.5 billion parameters, it is lightweight and works well on devices with limited computing power. It focuses on natural-sounding voices and easy customization.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B>

### Install Dependencies

```bash
pip3 install -r requirements.txt
pip3 install onnxruntime==1.18.0
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# If you failed to clone the submodule due to network failures, please run the following command until success
cd CosyVoice
git checkout 2db78e705835e56778f69b80cae51e4314d635b0
git submodule update --init --recursive

mkdir -p pretrained_models
# download CosyVoice2-0.5B model into pretrained_models dir

# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

## Model Inference

```bash
# make sure run cosyvoice2_example() in example.py 
python3 example.py
```

## Model Results

## References

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice/commit/2db78e705835e56778f69b80cae51e4314d635b0)