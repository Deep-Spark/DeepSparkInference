import json
class TransformerBaseConfig:
    def __init__(self, config_path, use_fp16=True):
        with open(config_path, "r") as f:
            data = json.load(f)
            self.num_attention_heads = data["num_attention_heads"]
            self.hidden_size = data["hidden_size"]
            self.intermediate_size = data["intermediate_size"]
            self.num_hidden_layers = data["num_hidden_layers"]
            self.head_size = self.hidden_size // self.num_attention_heads
            self.tgt_vocab_size = data["tgt_vocab_size"]
            self.max_sequence_length = data["max_sequence_length"]
            self.sos_token_id = data["sos_token_id"]
            self.eos_token_id = data["eos_token_id"]
            self.use_fp16 = use_fp16