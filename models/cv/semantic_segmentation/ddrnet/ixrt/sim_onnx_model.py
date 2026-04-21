import onnx
import argparse
from onnxsim import simplify

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_model_path", type=str)
    parser.add_argument("--sim_model_path", type=str)
    args = parser.parse_args()
    return args


args = parse_args()
onnx_model = onnx.load(args.raw_model_path)
model_simp, check = simplify(onnx_model)
onnx.save(model_simp, args.sim_model_path)
print('Simplify onnx Done.')
