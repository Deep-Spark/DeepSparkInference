import argparse
from cuda import cuda, cudart
from pipeline import StableDiffusionPipeline
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Generation")
    
    # 添加参数
    parser.add_argument("--prompt-file", type=str, help="File containing prompt text")
    parser.add_argument("--engine-path", type=str, help="File containing the engine path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--version", type=str, default="1.5", help="Stable Diffusion version")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--denoising-steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--num-warmup-runs", type=int, default=0, help="Number of warmup runs")
    parser.add_argument("--correct-image-path", type=str, default=0, help="the correct result image path")
    parser.add_argument("--perform-target", type=float, default=-1.0, help="the targer performs")
    parser.add_argument("--ddim-scheduler-path", type=str, default=0, help="the path of Scheduler DDIMScheduler")
    parser.add_argument("--clip-tokenizer-path", type=str, default=0, help="the path of CLIPTokenizer")
    
    return parser.parse_args()

def read_prompt_from_file(file_path):
    """从文件中读取prompt内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read().strip()
        return prompt
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def main():
    args = parse_arguments()
    
    # 处理prompt输入
    if args.prompt_file:
        prompt = read_prompt_from_file(args.prompt_file)
        if prompt is None:
            return
    elif args.prompt:
        prompt = args.prompt
    else:
        print("错误: 必须提供 --prompt-file 参数")
        return
    
    # 使用解析后的参数
    print("参数解析结果:")
    print(f"Prompt: {prompt}")
    print(f"engine-path: {args.engine_path}")
    print(f"Verbose: {args.verbose}")
    print(f"Version: {args.version}")
    print(f"Seed: {args.seed}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Width: {args.width}")
    print(f"Height: {args.height}")
    print(f"Denoising Steps: {args.denoising_steps}")
    print(f"Warmup Runs: {args.num_warmup_runs}")
    print(f"Correct image path: {args.correct_image_path}")

    demo = StableDiffusionPipeline(args, prompt)

    # Load TensorRT engines
    demo.loadEngines()
    
    # Load resources, free is in demo.teardown()
    _, shared_device_memory = cudart.cudaMalloc(demo.calculateMaxDeviceMemory())
    demo.activateEngines(shared_device_memory)
    demo.loadResources()
    demo.run()

    result = demo.CheckImageResult()
    exec_time = demo.exec_time

    demo.teardown()

    if args.perform_target > 0:
        current_throughput = 1 / exec_time
        print(f"Current throughput: {current_throughput}, Target accuracy: {args.perform_target}")
        # is performace mode
        if current_throughput < args.perform_target:

            print('Fail')
            sys.exit(1)
        else:
            print('pass')
            sys.exit(0)
    else:
        # is accuary mode
        if result == False:
            print('Fail')
            sys.exit(1)
        else:
            print('pass')
            sys.exit(0)


if __name__ == "__main__":
    main()