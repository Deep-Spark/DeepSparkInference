import argparse
import onnx

def change_input_output_dim(model):
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = "batch"
    # sym_batch_dim = -1

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        dim1.dim_param = sym_batch_dim

        if input.name == "new_categorical_placeholder:0":
            input.type.tensor_type.shape.dim[1].dim_value = int(2)
        elif input.name == "new_numeric_placeholder:0":
            input.type.tensor_type.shape.dim[1].dim_value = int(13)
        elif input.name == "import/head/predictions/zeros_like:0":
            input.type.tensor_type.shape.dim[1].dim_value = int(1)

        # or update it to be an actual value:
        # dim1.dim_value = actual_batch_dim

    outputs = model.graph.output

    for output in outputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = output.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        dim1.dim_param = sym_batch_dim
        
def change_input_node_name(model, input_names):
    for i,input in enumerate(model.graph.input):
        input_name = input_names[i]
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name == input.name:
                    node.input[i] = input_name
        input.name = input_name


def change_output_node_name(model, output_names):
    for i,output in enumerate(model.graph.output):
        output_name = output_names[i]
        for node in model.graph.node:
            for i, name in enumerate(node.output):
                if name == output.name:
                    node.output[i] = output_name
        output.name = output_name


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
    change_input_output_dim(model)
    model = onnx.load(args.model_path)
    for input in model.graph.input:
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name == input.name:
                    node.input[i] =name.replace(':',"")
        input.name=input.name.replace(':',"")# 保存修改后的模型
    onnx.save(model, args.output_path)