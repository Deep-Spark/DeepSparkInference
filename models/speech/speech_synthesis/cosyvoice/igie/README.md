# CosyVoice2 (ixRT)

## Model Description

CosyVoice2-0.5B is a small speech model designed to understand and generate human-like speech. It can be used for tasks like voice assistants, text-to-speech, or voice cloning. With 0.5 billion parameters, it is lightweight and works well on devices with limited computing power. It focuses on natural-sounding voices and easy customization.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | dev-only    |  26.06  |

## Model Preparation

! hint, the IGIE implement of cosyvoice is based on the following official cosyvoice, but we added ort and igie backend.
we rebased on the main branch with hash=ace7c47.
### Prepare Resources

Pretrained model: <https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B>

### Install Dependencies

```bash 
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

cd CosyVoice
git checkout ace7c47
git submodule update --init --recursive

pip3 install -r requirements.txt 


mkdir -p pretrained_models
# download CosyVoice2-0.5B model into pretrained_models dir

pip3 install onnxsim==0.4.36
cd pretrained_models/CosyVoice2-0.5B
onnxsim flow.decoder.estimator.fp32.onnx flow.decoder.estimator.fp32.opt.onnx
```

## Model Inference 

we got two ways to perform cosyvoice inference.
you can directly run example.py to get sythesis simple wavs, you also can get more formal style scores(based on dataset) with official cv3-eval tool.

way1:
```
python3 example.py
```

way2:
we using cv3-eval to mark cosyvoice scores, but there were some bugs in cv3-eval, so we modified some files.
since the original cv3-eval is too heavy(many *.wavs), we lighten it by only keep the modified files.

now we evaluate the zero_shot case with en/zh dataset.
```
# git clone https://github.com/FunAudioLLM/CV3-Eval

cd cosyvoice
python3 examples/cv3_eval/infer_cv3_eval.py \
  --cv3-eval-dir ../cv3-eval \
  --model-dir ./pretrained_models/CosyVoice2-0.5B \
  --task zero_shot \
  --subsets zh en\
  --load-igie \
  --fp16 \
  --output-dir ../cv3-eval/results/zero_shot \
```

after output wavs, we use cv3-eval to get the CER/WER/SS scores.
```
cd cv3-eval
bash run_infer_cv3_eval_zh.sh

bash run_infer_cv3_eval_en.sh
```

zero_shot zh:
CER 4.262%
SS 77.69

zero_shot en:
WER 6.846%
SS 70.65