import os
import shutil
import json
import argparse
from transformers import AutoTokenizer
from datasets import load_from_disk
from functools import partial

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", required=True, type=str, help="dataset-path")
parser.add_argument("--model", required=True, type=str, help="model-path")
parser.add_argument("--num-samples", type=int, default=512, help="The number of samples used for calibration")
parser.add_argument("--max-length", type=int, default=2048, help="max sequence length")
parser.add_argument("--model-type", type=str, required=True, choices=['llama', 'chatglm', "baichuan2", "qwen", "gpt-neox"], help="model type")
parser.add_argument("--smoothquant", type=bool, default=True, help="enable smoothquant")


args = parser.parse_args()


CHATGLM_SMOOTHQUANT_MAPPINGS = [
    [["re:.*query_key_value"], "re:.*input_layernorm"],
    [["re:.*dense_h_to_4h"], "re:.*post_attention_layernorm"],
]

BAICHUAN2_SMOOTHQUANT_MAPPINGS = [
    [["re:.*W_pack"], "re:.*input_layernorm"],
    [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
]

LLAMA_SMOOTHQUANT_MAPPINGS = [
    [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
    [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
]

GPT_NEOX_SMOOTHQUANT_MAPPINGS = [
    [["re:.*query_key_value"], "re:.*input_layernorm"],
    [["re:.*dense_h_to_4h"], "re:.*post_attention_layernorm"],
]

smooth_quant_mappings = {
    'chatglm': CHATGLM_SMOOTHQUANT_MAPPINGS,
    "baichuan2": BAICHUAN2_SMOOTHQUANT_MAPPINGS,
    'llama': LLAMA_SMOOTHQUANT_MAPPINGS,
    'qwen': LLAMA_SMOOTHQUANT_MAPPINGS,
    'gpt-neox': GPT_NEOX_SMOOTHQUANT_MAPPINGS,
}

ignore_layer = {
    "chatglm": ['transformer.output_layer'],
    "baichuan2":  ['lm_head'],
    "llama": ['lm_head'],
    "qwen": ['lm_head'],
    "gpt-neox": ['embed_out']
}


def preprocess(example, tokenizer, chat_template=None):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            chat_template=chat_template,
            tokenize=False,
        )
    }

# Tokenize inputs.
def tokenize(sample, tokenizer, max_length):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )

def save_tokenizer(model_path, save_path):
    tokenizer_files = [
        "special_tokens_map.json",
        "tokenization_chatglm.py", 
        "tokenizer_config.json", 
        "tokenizer.model",
        "tokenizer_config_default.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
    ]
    for file_name in tokenizer_files:
        src_file = os.path.join(model_path, file_name)
        des_file = os.path.join(save_path, file_name)
        if os.path.isfile(src_file):
            shutil.copy(src_file, des_file)

def preprocess_for_llama(model_path):
    generation_config_file = os.path.join(model_path, "generation_config.json")
    if os.path.isfile(generation_config_file):
        # add do_sample 
        shutil.copy(generation_config_file, generation_config_file+"_bk")
        with open(generation_config_file,'r') as f:
            config = json.load(f)
        if "do_sample" not in config:
            config['do_sample'] = True
            with open(generation_config_file,'w') as f:
                json.dump(config, f)

def main(args):
    if args.model_type == "llama":
        preprocess_for_llama(args.model)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)

    dataset = load_from_disk(args.dataset_path)['train_sft']
    dataset = dataset.shuffle(seed=42).select(range(args.num_samples))
    if tokenizer.chat_template is None:
        chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    else:
        chat_template = None

    dataset = dataset.map(partial(preprocess, tokenizer=tokenizer, chat_template=chat_template))
    dataset = dataset.map(partial(tokenize, tokenizer=tokenizer, max_length=args.max_length), remove_columns=dataset.column_names)

    if args.smoothquant:
        smooth_modifier = SmoothQuantModifier(smoothing_strength=0.8)
        smooth_modifier.mappings = smooth_quant_mappings[args.model_type]
        recipe = [
            smooth_modifier,
            QuantizationModifier(targets="Linear", scheme="W8A8", ignore=ignore_layer[args.model_type]),
        ]
    else:
        recipe = [
            QuantizationModifier(targets="Linear", scheme="W8A8", ignore=ignore_layer[args.model_type]),
        ]

    model = SparseAutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )

    # Apply algorithms.
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_length,
        num_calibration_samples=args.num_samples,
    )

    # Save to disk compressed.
    SAVE_DIR = args.model + "-W8A8-Dynamic-Per-Token"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    if args.model_type == "chatglm" or args.model_type == "gpt-neox":
        save_tokenizer(args.model, SAVE_DIR)
    else:
        tokenizer.save_pretrained(SAVE_DIR)

if __name__ == "__main__":
    main(args)
