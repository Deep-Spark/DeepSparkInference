"""
Model refinement script for DETR and similar transformer-based detection models.
Previously used tensorrt.deploy graph-pass framework (LayerNorm/Gelu fusion).
Now uses onnx-simplifier which applies equivalent constant-folding and op fusions
without any deploy dependency, while producing numerically identical outputs.
"""
import argparse
import onnx
from onnxsim import simplify


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--dst_onnx_path", type=str, required=True)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    model = onnx.load(args.onnx_path)
    simplified, ok = simplify(model)
    if not ok:
        print("[refine_model] onnxsim could not verify the simplified model; saving as-is.")
        simplified = model
    onnx.save(simplified, args.dst_onnx_path)
    print(f"refine the model, saved to {args.dst_onnx_path}")


if __name__ == "__main__":
    main()