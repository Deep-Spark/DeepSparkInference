import onnx
import argparse
from onnxsim import simplify

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str)
    parser.add_argument("--output_model", type=str)
    parser.add_argument("--input_names", nargs='+', type=str)
    parser.add_argument("--output_names", nargs='+', type=str)
    args = parser.parse_args()
    return args

args = parse_args()
onnx.utils.extract_model(args.input_model, args.output_model, args.input_names, args.output_names)
print("  Cut Model Done.")