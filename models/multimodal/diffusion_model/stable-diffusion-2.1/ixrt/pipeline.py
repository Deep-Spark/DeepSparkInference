import argparse
from engine import Engine
import gc
import torch
from cuda import cudart
import os
from diffusers import DDIMScheduler
from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
)
from type import PIPELINE_TYPE
import tensorrt as trt
import numpy as np
import inspect
import os
import random
from PIL import Image
import time
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

def make_scheduler(cls, version, pipeline, hf_token, framework_model_dir, subfolder="scheduler"):
    scheduler_dir = os.path.join(
        framework_model_dir, version, pipeline.name, next(iter({cls.__name__})).lower(), subfolder
    )
    if not os.path.exists(scheduler_dir):
        scheduler = cls.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder=subfolder, token=hf_token)
        scheduler.save_pretrained(scheduler_dir)
    else:
        print(f"[I] Load Scheduler {cls.__name__} from: {scheduler_dir}")
        scheduler = cls.from_pretrained(scheduler_dir)
    return scheduler

def load_scheduler(cls, scheduler_dir):
    scheduler = cls.from_pretrained(scheduler_dir)
    return scheduler

def make_tokenizer(version, pipeline, hf_token, framework_model_dir, subfolder="tokenizer", tokenizer_type="clip"):
    if tokenizer_type == "clip":
        tokenizer_class = CLIPTokenizer
    elif tokenizer_type == "t5":
        tokenizer_class = T5TokenizerFast
    else:
        raise ValueError(
            f"Unsupported tokenizer_type {tokenizer_type}. Only tokenizer_type clip and t5 are currently supported"
        )
    tokenizer_model_dir = 'pytorch_model/2.1/TXT2IMG/tokenizer'
    if not os.path.exists(tokenizer_model_dir):
        model = tokenizer_class.from_pretrained(
            'stabilityai/stable-diffusion-2-1', subfolder=subfolder, use_safetensors=False, token=hf_token
        )
        model.save_pretrained(tokenizer_model_dir)
    else:
        print(f"[I] Load {tokenizer_class.__name__} model from: {tokenizer_model_dir}")
        model = tokenizer_class.from_pretrained(tokenizer_model_dir)
    return model

def load_tokenizer(tokenizer_model_dir, tokenizer_type="clip"):
    if tokenizer_type == "clip":
        tokenizer_class = CLIPTokenizer
    elif tokenizer_type == "t5":
        tokenizer_class = T5TokenizerFast
    else:
        raise ValueError(
            f"Unsupported tokenizer_type {tokenizer_type}. Only tokenizer_type clip and t5 are currently supported"
        )
    print(f"[I] Load {tokenizer_class.__name__} model from: {tokenizer_model_dir}")
    model = tokenizer_class.from_pretrained(tokenizer_model_dir)
    return model

class StableDiffusionPipeline:
    def __init__(self, parser, prompt):
        self.prompt = prompt.splitlines()
        self.denoising_steps = parser.denoising_steps
        self.max_batch_size = parser.batch_size
        self.image_height = parser.height
        self.image_width = parser.width
        self.stages = ['clip','unet','vae']
        self.engine_path = parser.engine_path
        self.seed = parser.seed
        self.version = parser.version
        self.pipeline_type = PIPELINE_TYPE.TXT2IMG
        self.device = 'cuda'
        self.guidance_scale = 7.5
        self.vae_scaling_factor = 0.18215
        self.warmup_time = parser.num_warmup_runs
        self.correct_image_path = parser.correct_image_path
        self.clip_tokenizer_path = parser.clip_tokenizer_path
        self.gen_image_path = ''
        self.engine = {}
        self.events = {}
        self.negative_prompt = ['']
        self.shared_device_memory = None
        self.stream = None
        self.generator = None
        self.hf_token = None
        self.exec_time = -1

        # self.scheduler = make_scheduler(DDIMScheduler, self.version, self.pipeline_type, None, 'tmp')
        self.scheduler = load_scheduler(DDIMScheduler, parser.ddim_scheduler_path)

    def loadEngines(self):
        # self.tokenizer = make_tokenizer(self.version, self.pipeline_type, self.hf_token, 'pytorch_model')
        self.tokenizer = load_tokenizer(self.clip_tokenizer_path)

        self.engine['clip'] = Engine(self.engine_path + "/clip.engine")
        self.engine['clip'].load()
        self.engine['vae'] = Engine(self.engine_path + "/vae.engine")
        self.engine['vae'].load()
        self.engine['unet'] = Engine(self.engine_path + "/unet.engine")
        self.engine['unet'].load()
        print("load engine down.")
        # Release temp GPU memory during onnx export to avoid OOM.
        gc.collect()
        torch.cuda.empty_cache()

    def calculateMaxDeviceMemory(self):
        max_device_memory = 0
        for model_name, engine in self.engine.items():
            max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
        return max_device_memory

    def activateEngines(self, shared_device_memory):
        self.shared_device_memory = shared_device_memory
        # Load and activate TensorRT engines
        for engine in self.engine.values():
            engine.activate(device_memory=self.shared_device_memory)\

    def teardown(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e[0])
            cudart.cudaEventDestroy(e[1])

        for engine in self.engine.values():
            del engine

        if self.shared_device_memory:
            cudart.cudaFree(self.shared_device_memory)

        cudart.cudaStreamDestroy(self.stream)
        del self.stream

    def loadResources(self):
        # Initialize noise generator
        if self.seed:
            self.generator = torch.Generator(device="cuda").manual_seed(self.seed)

        # Create CUDA events and stream
        for stage in ['clip', 'denoise', 'vae']:
            self.events[stage] = [cudart.cudaEventCreate()[1], cudart.cudaEventCreate()[1]]
        self.stream = cudart.cudaStreamCreate()[1]

        # Allocate TensorRT I/O buffers
        self.engine['clip'].allocate_buffers({'input_ids': (self.max_batch_size, 77), 'text_embeddings': (self.max_batch_size, 77, 1024)})
        self.engine['unet'].allocate_buffers({'sample': (self.max_batch_size * 2, 4, 64, 64), 'encoder_hidden_states': (self.max_batch_size * 2, 77, 1024), 'latent': (self.max_batch_size * 2, 4, 64, 64)})
        self.engine['vae'].allocate_buffers({'latent': (self.max_batch_size, 4, 64, 64), 'images': (self.max_batch_size, 3, 512, 512)})

    def initialize_latents(self, batch_size, unet_channels, latent_height, latent_width, latents_dtype=torch.float32):
        latents_dtype = latents_dtype # text_embeddings.dtype
        latents_shape = (batch_size, unet_channels, latent_height, latent_width)
        latents = torch.randn(latents_shape, device='cuda', dtype=latents_dtype, generator=self.generator)
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream)

    def encode_prompt(self, prompt, negative_prompt, encoder='clip', pooled_outputs=False, output_hidden_states=False):
        self.profile_start('clip', color='green')

        tokenizer = self.tokenizer

        def tokenize(prompt, output_hidden_states):
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.type(torch.int32).to('cuda')

            text_hidden_states = None

            # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
            outputs = self.runEngine(encoder, {'input_ids': text_input_ids})
            text_embeddings = outputs['text_embeddings'].clone()
            if output_hidden_states:
                text_hidden_states = outputs['hidden_states'].clone()
            return text_embeddings, text_hidden_states

        # Tokenize prompt
        text_embeddings, text_hidden_states = tokenize(prompt, output_hidden_states)


        # Tokenize negative prompt
        uncond_embeddings, uncond_hidden_states = tokenize(negative_prompt, output_hidden_states)

        # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)

        if pooled_outputs:
            pooled_output = text_embeddings

        if output_hidden_states:
            text_embeddings = torch.cat([uncond_hidden_states, text_hidden_states]).to(dtype=torch.float16)

        self.profile_stop('clip')
        if pooled_outputs:
            return text_embeddings, pooled_output
        return text_embeddings

    def preprocess_controlnet_images(self, batch_size, images=None):
        '''
        images: List of PIL.Image.Image
        '''
        if images is None:
            return None
        self.profile_start('preprocess', color='pink')
        images = [(np.array(i.convert("RGB")).astype(np.float32) / 255.0)[..., None].transpose(3, 2, 0, 1).repeat(batch_size, axis=0) for i in images]
        # do_classifier_free_guidance
        images = [torch.cat([torch.from_numpy(i).to(self.device).float()] * 2) for i in images]
        images = torch.cat([image[None, ...] for image in images], dim=0)
        self.profile_stop('preprocess')
        return images

    def denoise_latent(self,
        latents,
        text_embeddings,
        denoiser='unet',
        timesteps=None,
        step_offset=0,
        mask=None,
        masked_image_latents=None,
        image_guidance=1.5,
        controlnet_imgs=None,
        controlnet_scales=None,
        text_embeds=None,
        time_ids=None):

        assert image_guidance > 1.0, "Image guidance has to be > 1.0"

        controlnet_imgs = self.preprocess_controlnet_images(latents.shape[0], controlnet_imgs)

        with torch.autocast('cuda', enabled=False):
            self.profile_start('denoise', color='blue')
            for step_index, timestep in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
                if isinstance(mask, torch.Tensor):
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # Predict the noise residual
                timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

                params = {"sample": latent_model_input, "timestep": timestep_float, "encoder_hidden_states": text_embeddings}
                if controlnet_imgs is not None:
                    params.update({"images": controlnet_imgs, "controlnet_scales": controlnet_scales})
                if text_embeds != None:
                    params.update({'text_embeds': text_embeds})
                if time_ids != None:
                    params.update({'time_ids': time_ids})
                noise_pred = self.runEngine(denoiser, params)['latent']

                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # from diffusers (prepare_extra_step_kwargs)
                extra_step_kwargs = {}
                if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
                    # TODO: configurable eta
                    eta = 0.0
                    extra_step_kwargs["eta"] = eta
                if "generator" in set(inspect.signature(self.scheduler.step).parameters.keys()):
                    extra_step_kwargs["generator"] = self.generator

                latents = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]

            latents = 1. / self.vae_scaling_factor * latents
            latents = latents.to(dtype=torch.float32)

        self.profile_stop('denoise')
        return latents

    def decode_latent(self, latents):
        self.profile_start('vae', color='red')
        cast_to = torch.float16
        latents = latents.to(dtype=cast_to)

        images = self.runEngine('vae', {'latent': latents})['images']
        self.profile_stop('vae')
        return images

    def save_image_impl(self, images, image_path_dir, image_name_prefix, image_name_suffix):
        """
        Save the generated images to png files.
        """
        for i in range(images.shape[0]):
            image_path = os.path.join(
                image_path_dir,
                f"{image_name_prefix}{i + 1}-{random.randint(1000, 9999)}-{image_name_suffix}.png",
            )
            print(f"Saving image {i+1} / {images.shape[0]} to: {image_path}")
            Image.fromarray(images[i]).save(image_path)
            self.gen_image_path = image_path

    def save_image(self, images, pipeline, prompt, seed):
        # Save image
        image_name_prefix = pipeline+''.join(set(['-'+prompt[i].replace(' ','_')[:10] for i in range(len(prompt))]))+'-'+str(seed)+'-'
        image_name_suffix = 'trt'
        self.save_image_impl(images, '.', image_name_prefix, image_name_suffix)

    def run(self):
        self.prompt = self.prompt * self.max_batch_size
        self.negative_prompt = self.negative_prompt * self.max_batch_size

        if self.warmup_time > 0:
            for _ in range(self.warmup_time):
                self.infer(True)
        
        print("[I] Running StableDiffusion pipeline")
        self.infer()

    def infer(self, is_warmp_up = False):
        print(f"prompt{self.prompt}, negative_prompt{self.negative_prompt}")
        assert len(self.prompt) == len(self.negative_prompt)
        batch_size = len(self.prompt)
        # Spatial dimensions of latent tensor
        latent_height = self.image_height // 8
        latent_width = self.image_width // 8

        if self.generator and self.seed:
            self.generator.manual_seed(self.seed)
        
        num_inference_steps = self.denoising_steps

        with torch.inference_mode(), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            start_time = time.time()
            self.scheduler.set_timesteps(self.denoising_steps, device='cuda')
            timesteps = self.scheduler.timesteps.to(device='cuda')

            # Initialize latents
            latents = self.initialize_latents(batch_size=batch_size,
                unet_channels=4,
                latent_height=latent_height,
                latent_width=latent_width)
            
            text_embeddings = self.encode_prompt(self.prompt, self.negative_prompt)

            timesteps = self.scheduler.timesteps.to(self.device)
            latents = self.denoise_latent(latents, text_embeddings, 'unet', timesteps)

            images = self.decode_latent(latents)

            torch.cuda.synchronize()
            end_time = time.time()
            self.exec_time = end_time - start_time # s
            if not is_warmp_up:
                self.print_summary(self.denoising_steps, self.exec_time * 1000, self.max_batch_size)
                print("Pipeline time: {} s".format(self.exec_time))

            images = ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
            self.save_image(images, self.pipeline_type.name.lower(), self.prompt, self.seed)

    def profile_start(self, name, color='blue'):
        if name in self.events:
            cudart.cudaEventRecord(self.events[name][0], 0)

    def profile_stop(self, name):
        if name in self.events:
            cudart.cudaEventRecord(self.events[name][1], 0)

    def print_summary(self, denoising_steps, walltime_ms, batch_size):
        print('|-----------------|--------------|')
        print('| {:^15} | {:^12} |'.format('Module', 'Latency'))
        print('|-----------------|--------------|')
        if 'vae_encoder' in self.stages:
            print('| {:^15} | {:>9.2f} ms |'.format('VAE-Enc', cudart.cudaEventElapsedTime(self.events['vae_encoder'][0], self.events['vae_encoder'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('CLIP', cudart.cudaEventElapsedTime(self.events['clip'][0], self.events['clip'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('UNet'+('+CNet' if self.pipeline_type.is_controlnet() else '')+' x '+str(denoising_steps), cudart.cudaEventElapsedTime(self.events['denoise'][0], self.events['denoise'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('VAE-Dec', cudart.cudaEventElapsedTime(self.events['vae'][0], self.events['vae'][1])[1]))
        print('|-----------------|--------------|')
        print('| {:^15} | {:>9.2f} ms |'.format('Pipeline', walltime_ms))
        print('|-----------------|--------------|')
        print('Throughput: {:.2f} image/s'.format(batch_size*1000./walltime_ms))

        metricResult = {"metricResult": {}}
        metricResult["metricResult"]["CLIP"] = cudart.cudaEventElapsedTime(self.events['clip'][0], self.events['clip'][1])[1]
        metricResult["metricResult"]["UNetx20"] = cudart.cudaEventElapsedTime(self.events['denoise'][0], self.events['denoise'][1])[1]
        metricResult["metricResult"]["VAE-Dec"] = cudart.cudaEventElapsedTime(self.events['vae'][0], self.events['vae'][1])[1]
        metricResult["metricResult"]["Pipeline"] = walltime_ms
        metricResult["metricResult"]["Throughput"] = batch_size*1000./walltime_ms
        print(metricResult)

    def CheckImageResult(self):
        print(f"correct image path {self.correct_image_path}, generate image path {self.gen_image_path}")
        if self.correct_image_path == '' or self.gen_image_path == '':
            return False

        try:
            # 读取图片
            img_ref = Image.open(self.correct_image_path)
            img_gen = Image.open(self.gen_image_path)
            
            # 转换为 numpy 数组
            arr_ref = np.array(img_ref)
            arr_gen = np.array(img_gen)
            
            # 确保图像尺寸相同
            if arr_ref.shape != arr_gen.shape:
                img_gen = img_gen.resize(img_ref.size, Image.Resampling.LANCZOS)
                arr_gen = np.array(img_gen)
            
            print("=" * 60)
            print("图像比较结果:")
            
            # 1. 像素级差异（基础比较）
            diff_pixels = np.sum(arr_ref != arr_gen)
            total_pixels = arr_ref.size
            diff_percentage = (diff_pixels / total_pixels) * 100
            
            if len(arr_ref.shape) == 3:  # 彩色图像
                mean_diff = np.mean(np.abs(arr_ref.astype(np.float32) - arr_gen.astype(np.float32)))
            else:  # 灰度图像
                mean_diff = np.mean(np.abs(arr_ref.astype(np.float32) - arr_gen.astype(np.float32)))
            
            print(f"不同像素数量: {diff_pixels:,} / {total_pixels:,}")
            print(f"差异百分比: {diff_percentage:.6f}%")
            print(f"平均像素差异: {mean_diff:.4f}")
            
            # 2. 结构相似性比较（更适合内容相似的图片）
            from skimage.metrics import structural_similarity as ssim
            
            if len(arr_ref.shape) == 3:
                # 转换为灰度图计算SSIM
                gray_ref = np.mean(arr_ref, axis=2).astype(np.uint8)
                gray_gen = np.mean(arr_gen, axis=2).astype(np.uint8)
                ssim_score = ssim(gray_ref, gray_gen)
            else:
                ssim_score = ssim(arr_ref, arr_gen)
            
            print(f"结构相似性(SSIM): {ssim_score:.4f}")
            
            # 3. 直方图比较（颜色分布相似性）
            if len(arr_ref.shape) == 3:
                hist_ref = cv2.calcHist([arr_ref], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist_gen = cv2.calcHist([arr_gen], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist_corr = cv2.compareHist(hist_ref, hist_gen, cv2.HISTCMP_CORREL)
                print(f"直方图相关性: {hist_corr:.4f}")
            else:
                hist_corr = 1.0
            
            # 4. 智能判断逻辑
            if ssim_score > 0.6:  # 结构相似性较高
                print("✓ 图片内容结构相似（都是景色）")
                return True
            elif hist_corr > 0.7:  # 颜色分布相似
                print("✓ 图片颜色分布相似")
                return True
            elif mean_diff < 10.0 and diff_percentage < 30.0:  # 像素级差异较小
                print("✓ 像素级差异在可接受范围内")
                return True
            else:
                print("✗ 图片差异过大")
                print(f"mean_diff: {mean_diff}")
                return False
                
        except Exception as e:
            print(f"图像比较出错: {str(e)}")
            return False