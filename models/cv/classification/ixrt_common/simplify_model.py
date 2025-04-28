import onnx
import argparse
from onnxsim import simplify

# Simplify
def simplify_model(args):
    onnx_model = onnx.load(args.origin_model)
    model_simp, check = simplify(onnx_model)
    model_simp = onnx.shape_inference.infer_shapes(model_simp)
    onnx.save(model_simp, args.output_model)
    print("  Simplify onnx Done.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_model", type=str)
    parser.add_argument("--output_model", type=str)
    parser.add_argument("--reshape", action="store_true")
    args = parser.parse_args()
    return args

args = parse_args()
simplify_model(args)
    



