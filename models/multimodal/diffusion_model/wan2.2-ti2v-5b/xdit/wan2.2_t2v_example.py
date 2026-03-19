import logging
import time
import torch
import torch.distributed
from diffusers import AutoencoderKLTemporalDecoder
from xfuser import xFuserWanPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
    get_world_group
)
from diffusers import WanPipeline

from xfuser.core.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from diffusers.utils import export_to_video
from xfuser.model_executor.cache.teacache.backend import TeaCacheBackend
from xfuser.model_executor.cache.data import DiffusionCacheConfig


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)

    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."

    pipe = xFuserWanPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    if engine_args.use_teacache:
        config = DiffusionCacheConfig(rel_l1_thresh = 0.2,
                                      coefficients = [
                                       6.85271205e+04,
                                       -9.88214072e+03,
                                       5.08858742e+02,
                                       -7.39731467e+00,
                                        1.22746295e-01,])
        backend = TeaCacheBackend(config)
        backend.enable(pipe,transformer_key = "transformer_2")
        backend.refresh(pipe, input_config.num_inference_steps, transformer_key = "transformer_2")
    
    
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)

    if args.enable_tiling:
        pipe.vae.enable_tiling()

    if args.enable_slicing:
        pipe.vae.enable_slicing()
        
    if  engine_args.use_w8a8_linear:  
        from w8a8_linear import apply_quant_linear_i8w8o16
        pipe.transformer=apply_quant_linear_i8w8o16(pipe.transformer)   

    # warmup
    # output = pipe(
    #     height=input_config.height,
    #     width=input_config.width,
    #     num_frames=input_config.num_frames,
    #     prompt=input_config.prompt,
    #     num_inference_steps=1,
    #     generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    # ).frames

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()


    if args.use_easycache:
        cache_kwargs = {
        "use_easycache":True,
        "cache_thresh":0.02,  #easy eacch thresh
        #"ret_steps":10
       }
    else:
        cache_kwargs = None  

    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        negative_prompt = input_config.negative_prompt,
        num_inference_steps=input_config.num_inference_steps,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        cache_kwargs = cache_kwargs
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_reserved(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    # if is_dp_last_group():
    resolution = f"{input_config.width}x{input_config.height}"
    for i, frames in enumerate(output.frames):
        output_filename = f"results/wan2.2_t2v_{i}_{parallel_info}_{resolution}.mp4"
        export_to_video(frames, output_filename, fps=16)
        print(f"output saved to {output_filename}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
