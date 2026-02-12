import argparse
import time
from openai import OpenAI

def main(args):
    # 初始化 OpenAI 客户端 |Initialize the OpenAI client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout
    )

    # 构造消息 |Construct message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": args.image_url
                    }
                },
                {
                    "type": "text",
                    "text": args.prompt
                }
            ]
        }
    ]

    print(f"Sending request to {args.base_url} with model {args.model_name}...")
    
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=args.model_name,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,

            extra_body={
                "skip_special_tokens": False,
                "vllm_xargs": {
                    "ngram_size": 30,
                    "window_size": 90,
                    # whitelist: <td>, </td>
                    "whitelist_token_ids": [128821, 128822],
                },
            },
        )
        elapsed_time = time.time() - start
        
        content = response.choices[0].message.content
        print(f"\nResponse costs: {elapsed_time:.2f}s")
        print("-" * 20 + " Generated Text " + "-" * 20)
        print(content)
        print("-" * 56)

    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Client for vLLM DeepSeek-OCR Online Inference')
    
    # 服务器配置 |Server configuration
    parser.add_argument('--base-url', type=str, default="http://localhost:8000/v1",
                        help='vLLM API server base URL.')
    parser.add_argument('--api-key', type=str, default="EMPTY",
                        help='API key for the server.')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Request timeout in seconds.')

    # 模型配置 |Model configuration
    parser.add_argument('--model-name', type=str, 
                        default="/mnt/app_auto/models_zoo/deepseek_models/DeepSeek-OCR/",
                        help='Name of the model served by vLLM.')

    # 推理参数 |Reasoning parameters
    parser.add_argument('--image-url', type=str, 
                        default="https://pic1.zhimg.com/v2-1b8eed3577af0c15d5bb9c78e09e7c18_1440w.jpg",
                        help='URL of the image to process.')
    parser.add_argument('--prompt', type=str, default="Free OCR.",
                        help='Prompt text.')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature.')
    parser.add_argument('--max-tokens', type=int, default=2048,
                        help='Maximum tokens to generate.')

    args = parser.parse_args()
    main(args)