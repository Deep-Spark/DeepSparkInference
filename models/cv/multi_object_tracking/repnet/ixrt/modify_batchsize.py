import onnx
import argparse
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

    # Modify Reshape params: (1, -1)--->(batch_size, -1)
    shape_edges = []
    for node in model.graph.node:
        if node.op_type == "Reshape":
            shape_name = node.input[-1]
            shape_edges.append(shape_name)

    shape_edges = list(set(shape_edges))
    for data in model.graph.initializer:
        if data.name in shape_edges:
            raw_data = np.frombuffer(data.raw_data, np.int64).copy()
            raw_data[0] = batch_size
            data.raw_data = raw_data.tobytes()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--origin_model", type=str)
    parser.add_argument("--output_model", type=str)
    args = parser.parse_args()
    return args

args = parse_args()
model = onnx.load(args.origin_model)
change_input_dim(model, args.batch_size)
onnx.save(model, args.output_model)
