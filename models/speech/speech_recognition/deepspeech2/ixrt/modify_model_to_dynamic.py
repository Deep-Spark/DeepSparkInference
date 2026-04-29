import onnx
from onnx import helper
import argparse

def modify_to_dynamic(arg):
    model = onnx.load(args.static_onnx)

    graph = model.graph
    for input_node in graph.input:
        if input_node.name == 'input':
            input_shape = input_node.type.tensor_type.shape.dim
            input_shape[0].dim_value = 1
            input_shape[1].dim_param = 'None'
            input_shape[2].dim_value = 161

    onnx.save(model, args.dynamic_onnx)
    onnx.checker.check_model(model, full_check=True)

def parse_args():
    parser = argparse.ArgumentParser(description="modify static shape to dynamic for deepspeech2")
    parser.add_argument("--static_onnx", type=str, required=True, help="The input static onnx path")
    parser.add_argument("--dynamic_onnx", type=str, required=True, help="The ouput dynamic onnx path")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    modify_to_dynamic(args)