# Stable Diffusion 1.5

## Model description

Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.

## Model Preparation

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/diffusers-0.31.0-py3-none-any.whl
pip3 install -r requirements.txt
```

### Prepare Resources

Download the runwayml/stable-diffusion-v1-5 from [huggingface page](https://huggingface.co/runwayml/stable-diffusion-v1-5).

```bash
cd stable-diffusion
mkdir -p data/
ln -s /path/to/stable-diffusion-v1-5 ./data/
```

## Model Inference

```bash
export ENABLE_IXFORMER_INFERENCE=1
python3 demo.py
```

## References

- [diffusers](https://github.com/huggingface/diffusers)
