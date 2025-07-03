import onnx
import argparse
import copy
import numpy as np

def change_input_dim(model, bsz):
    batch_size = bsz

    # The following code changes the first dimension of every input to be batch_size
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_size
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        if isinstance(batch_size, str):
            # set dynamic batch size
            dim1.dim_param = batch_size
        elif (isinstance(batch_size, str) and batch_size.isdigit()) or isinstance(batch_size, int):
            # set given batch size
            dim1.dim_value = int(batch_size)
        else:
            # set batch size of 1
            dim1.dim_value = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--origin_model", type=str)
    parser.add_argument("--output_model", type=str)
    args = parser.parse_args()
    return args

def modify_resize_nodes(model, bsz):
    print("modify resize")
    for node in model.graph.node:
        if node.op_type == "Resize":
            if len(node.input) >= 4 and node.input[3]:
                sizes_name = node.input[3]
                for initializer in model.graph.initializer:
                    if initializer.name == sizes_name:
                        shape = copy.deepcopy(onnx.numpy_helper.to_array(initializer))
                        shape[0] = shape[0] * bsz
                        new_sizes = np.array(shape, dtype=np.int64)
                        initializer.CopyFrom(onnx.numpy_helper.from_array(new_sizes, name=initializer.name))
                        break
    
args = parse_args()
model = onnx.load(args.origin_model)
change_input_dim(model, args.batch_size)
modify_resize_nodes(model, args.batch_size)
onnx.save(model, args.output_model)
