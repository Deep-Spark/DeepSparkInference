import time
import os
import torch
import torch.distributed
from transformers import T5EncoderModel
from xfuser import xFuserStableDiffusion3Pipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    is_dp_last_group,
    get_data_parallel_rank,
    get_runtime_state,
)
from xfuser.core.distributed.parallel_state import get_data_parallel_world_size

from apex.normalization.fused_layer_norm import FusedRMSNorm
import torch
import torch.nn as nn


from ixformer.inference.functions.rms_norm import rms_norm


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # Compute variance without subtracting mean (RMSNorm)
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Cast back to half precision if needed
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
    

#https://github.com/huggingface/transformers/issues/20287  fix apex, from apex.  normalization import FusedRMSNorm 
def replace_fused_rmsnorm_with_t5(module):
    for name, child in module.named_children():
        if isinstance(child, FusedRMSNorm):
            hidden_size = child.weight.shape[0]
            eps = getattr(child, "eps", 1e-6)
            new_ln = T5LayerNorm(hidden_size, eps=eps)
            new_ln.weight.data = child.weight.data.clone()
            setattr(module, name, new_ln)
        else:
            replace_fused_rmsnorm_with_t5(child)

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank
    torch.cuda.set_device(local_rank)
    text_encoder_3 = T5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder_3", torch_dtype=torch.float16)
    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        print(f"rank {local_rank} quantizing text encoder 2")
        quantize(text_encoder_3, weights=qfloat8)
        freeze(text_encoder_3)

    pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
        text_encoder_3=text_encoder_3,
    ).to(f"cuda:{local_rank}")
    
    replace_fused_rmsnorm_with_t5(text_encoder_3)
    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    import os
    if os.environ.get("ENABLE_IXFORMER_W8A8LINEAR", "0") == "1":
        from w8a8_linear import apply_quant_linear_i8w8o16
        pipe.transformer=apply_quant_linear_i8w8o16(pipe.transformer)
    
    pipe.prepare_run(input_config)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if pipe.is_dp_last_group():
            if not os.path.exists("results"):
                os.mkdir("results")
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image.save(
                    f"./results/stable_diffusion_3_result_{parallel_info}_{image_rank}.png"
                )
                print(
                    f"image {i} saved to ./results/stable_diffusion_3_result_{parallel_info}_{image_rank}.png"
                )

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, peak memory: {peak_memory/1e9:.2f} GB"
        )

    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
