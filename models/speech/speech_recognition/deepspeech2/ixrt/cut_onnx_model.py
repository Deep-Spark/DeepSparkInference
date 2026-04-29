import os
import onnx

base_path = "../../../../../data/checkpoints/deepspeech2"

raw_path = os.path.join(base_path, "deepspeech2_all.onnx")
save_path = os.path.join(base_path, "deepspeech2_part.onnx")

input_names = ["input"]
output_names = ["layer_norm_9.tmp_2"]

onnx.utils.extract_model(raw_path, save_path, input_names, output_names)
