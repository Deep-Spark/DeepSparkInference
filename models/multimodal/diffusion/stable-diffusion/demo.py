import random
import time
import numpy as np
import torch
from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(20)

pipe = StableDiffusionPipeline.from_pretrained(f"data/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.safety_checker = None
pipe = pipe.to("cuda")
prompt = "A raccoon wearing formal clothes, wearing a tophat and holding a cane"
print("Warming up GPU...")
wh = 1024
batch_size=1
print(f'height={wh}, width={wh}, prompt={prompt}, batch_size={batch_size}')
start_time = time.time()
image = pipe(
    prompt, output_type="pt", return_dict=True, height=wh, width=wh, num_images_per_prompt=batch_size, num_inference_steps=50, guidance_scale=7.0
).images[0]
use_time = time.time() - start_time
print("time: {:.2f} seconds".format(use_time))
save_image(image, "demo.png")