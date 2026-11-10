# MOSS-Image (ComfyUI)

## Model Description

MOSS-Image is a text-to-image model family from OpenMOSS. MOSS-Image-SFT is the supervised fine-tuned release, derived from MOSS-Image-Base with the same architecture and inference pipeline. It accepts natural-language prompts, tag sequences, and JSON layouts for controlling text placement and object arrangement, and is particularly strong at rendering legible text inside the generated image. This page uses the ComfyUI repack, which ships the diffusion model, the text encoder, and the VAE as separate single-file weights.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| BI-V150 | 5.0.0 | 26.12 | release/26.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.12`

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/Comfy-Org/MOSS-Image/file/view/master/diffusion_models/moss_image_sft_bf16.safetensors>
- Encoder: <https://modelscope.cn/models/Comfy-Org/MOSS-Image/file/view/master/text_encoders/qwen3.5_9b_bf16.safetensors>
- VAE: <https://modelscope.cn/models/Comfy-Org/MOSS-Image/file/view/master/vae/flux2-vae_small.safetensors>

> 仓库同时提供 `int8_convrot` / `w4a8_gptq` 量化权重。CoreX 没有对应的量化算子，请只取上面三个 BF16 文件。

### Install Dependencies

```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
git checkout v0.44.0
pip3 install --no-deps torchsde trampoline alembic SQLAlchemy Mako greenlet av simpleeval
pip3 download --no-deps --platform any --only-binary=:all: -d wheels comfy-aimdo==0.6.0 comfy-kitchen==0.2.40
pip3 install --no-deps wheels/*.whl
mkdir -p web
python3 main.py --listen 0.0.0.0 --port 8188 --disable-dynamic-vram --front-end-root ./web
```

> `--no-deps` 保留镜像自带的 CoreX PyTorch。`comfy-aimdo` 与 `comfy-kitchen` 在启动时被导入，这里用
> `--platform any --only-binary=:all:` 只取纯 Python wheel，避免装进带 NVIDIA CUDA kernel 的二进制包；
> 两者都会回落到普通 PyTorch 实现。`--front-end-root ./web` 指向一个空目录，省掉前端资源包。

## Model Inference

- Download moss_image_sft_bf16.safetensors and put it in your ComfyUI/models/diffusion_models directory.

- Download qwen3.5_9b_bf16.safetensors and put it in your ComfyUI/models/text_encoders directory.

- Download flux2-vae_small.safetensors and put it in your ComfyUI/models/vae directory.

Then queue the workflow this run used. `inference.py` posts `prompt.json` to the running
server, waits for the run to finish, and prints where the image was written:

```bash
python3 inference.py --server http://127.0.0.1:8188 --workflow prompt.json
```

The positive prompt and the step count can be overridden without editing the workflow:

```bash
python3 inference.py --prompt "a calligraphy brush on rice paper" --steps 20
```

You can also load up the following workflow in ComfyUI:

https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_moss_image_t2i.json

## Model Results

| Model | Precision | Resolution | Steps | Latency |
| :----: | :----: | :----: | :----: | :----: |
| MOSS-Image | BF16 | 1024x1024 | 20 | 42.10 s |

> 以上是权重已驻留显存后的采样耗时。服务刚起来时第一次排队要先把 diffusion model、
> text encoder 和 VAE 读进来，用上面的命令实测端到端 112.4 s；之后再排同一个工作流即回到上表的量级。

## References

- [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Workflow template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_moss_image_t2i.json)
