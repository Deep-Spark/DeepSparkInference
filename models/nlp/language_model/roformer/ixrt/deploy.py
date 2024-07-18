import onnx
import argparse

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--output_path", default="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    model = onnx.load(args.model_path)
    for input in model.graph.input:
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name == input.name:
                    node.input[i] =name.replace(':',"")
        input.name=input.name.replace(':',"")# 保存修改后的模型
    onnx.save(model, args.output_path)