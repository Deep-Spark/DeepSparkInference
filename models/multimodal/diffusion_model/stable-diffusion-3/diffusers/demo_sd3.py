import torch
from diffusers import StableDiffusion3Pipeline
import time
from PIL import Image
import numpy as np
import random
import os
dtype=torch.float16
  
      
os.environ["ENABLE_IXFORMER_INFERENCE"] = "1" 
os.environ["USE_NHWC_GN"] = "1"
def ixformer_accelerate(pipe):
    pipe.transformer.fuse_qkv_projections()
    if int(os.environ.get("USE_NHWC_GN", 0)):        
        pipe.vae.to(memory_format=torch.channels_last)
    # pipe.text_encoder=torch.compile(pipe.text_encoder)
    # pipe.text_encoder_2=torch.compile(pipe.text_encoder_2)
    # pipe.text_encoder_3=torch.compile(pipe.text_encoder_3)
 
     
pipe = StableDiffusion3Pipeline.from_pretrained(f"stablediffusion3/stable-diffusion-3-medium-diffusers", torch_dtype=dtype)
pipe = pipe.to("cuda")
ixformer_accelerate(pipe) 
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

prompt="A cat holding a sign that says hello world"
resolution=[512,1024]
num_inference_steps=20
guidance_scale=7.0
for item in resolution:
    # 设置随机数种子
    setup_seed(20)
    width=height=item
    #warm up
    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height
    ).images[0]
    image_name = 'sd3_'+str(item)+'x'+str(item)+'.png'
    image.save(image_name)

    iter=2
    #performence
    torch.cuda.synchronize()
    start_time = time.time()
    torch.cuda.profiler.start()
    for _ in range(iter):     
        pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,width=width,
        height=height)
    torch.cuda.profiler.stop()
    torch.cuda.synchronize()
    use_time = time.time() - start_time
    print(f"resolution: {item}x{item}, num_inference_steps: {num_inference_steps}, guidance_scale: {guidance_scale}, time: {use_time/iter:.2f} seconds")


