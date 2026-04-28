# IndexTTS-2 (indextts)

## Model Description

IndexTTS-2 is a breakthrough in emotionally expressive and duration-controlled auto-regressive zero-shot text-to-speech. It proposes a novel method for speech duration control and achieves state-of-the-art performance in emotional fidelity.

Key features:
- **Duration Control**: Supports explicit token count specification or free autoregressive generation
- **Emotion Disentanglement**: Independent control over timbre and emotion
- **Soft Instruction Mechanism**: Natural language descriptions for emotion control via fine-tuned Qwen3
- **GPT Latent Representations**: Improves speech clarity in highly emotional expressions

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/IndexTeam/IndexTTS-2>
- Encoder: <https://huggingface.co/facebook/w2v-bert-2.0>
- Codec: <https://huggingface.co/amphion/MaskGCT>
- FunASR: <https://huggingface.co/funasr/campplus>
- BigVGAN: <https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x>


### Install Dependencies

IndexTTS-2 requires the official repository to be cloned and dependencies installed:

```bash
pip install bentoml \
    accelerate==1.8.1 \
    transformers==4.52.1 \
    cn2an==0.5.22 \
    ffmpeg-python==0.2.0 \
    Cython==3.0.7 \
    g2p-en==2.1.0 \
    jieba==0.42.1 \
    keras==2.9.0 \
    numba \
    numpy==1.26.2 \
    pandas==2.1.3 \
    matplotlib==3.8.2 \
    opencv-python==4.9.0.80 \
    vocos==0.1.0 \
    omegaconf \
    sentencepiece \
    munch==4.0.0 \
    librosa==0.10.2.post1 \
    descript-audiotools==0.7.2 \
    textstat==0.7.10 \
    tokenizers==0.21.0 \
    json5==0.10.0 \
    pydub \
    tqdm \
    wetext \
    WeTextProcessing

# Clone the repository
git clone https://github.com/index-tts/index-tts.git
cd index-tts
git lfs pull
```

## Model Inference

```bash
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints/IndexTTS-2", use_fp16=False, use_cuda_kernel=False, use_deepspeed=True)

#如果需要生成不同的音色和音频内容，请替换spk_audio_prompt所需的参考音频以及text的输出内容。
text = "快躲起来！是他要来了！他要来抓我们了！"
tts.infer(spk_audio_prompt='examples/emo_sad.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)
```