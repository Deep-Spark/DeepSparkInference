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
        if isinstance(batch_size, str):
            # set dynamic (symbolic) batch size — clear any existing static value first
            dim1.ClearField("dim_value")
            dim1.dim_param = batch_size
        elif isinstance(batch_size, int):
            # set given static batch size — clear any existing symbolic name first
            dim1.ClearField("dim_param")
            dim1.dim_value = int(batch_size)
        else:
            # set batch size of 1
            dim1.ClearField("dim_param")
            dim1.dim_value = 1

def change_reshape_batch(model, bsz):
    batch_size = int(bsz) if not isinstance(bsz, int) else bsz
    initializer_map = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == 'Reshape' and len(node.input) >= 2:
            shape_name = node.input[1]
            if shape_name not in initializer_map:
                continue
            init = initializer_map[shape_name]
            shape_val = np.array(onnx.numpy_helper.to_array(init))
            if len(shape_val) >= 1 and shape_val[0] > 0 and shape_val[0] != batch_size:
                old_val = shape_val[0]
                shape_val[0] = batch_size
                new_init = onnx.numpy_helper.from_array(shape_val, name=shape_name)
                init.CopyFrom(new_init)
                print(f"  Reshape {node.name}: shape[0] {old_val} -> {batch_size}")

def infer_node_shape(model):
    # remove old shape of the node
    for value_info in model.graph.value_info:
        tensor_type = value_info.type.tensor_type
        if tensor_type.HasField('shape'):
            tensor_type.ClearField('shape')

    for output_info in model.graph.output:
        tensor_type = output_info.type.tensor_type
        if tensor_type.HasField('shape'):
            tensor_type.ClearField('shape')

    from onnx import checker, shape_inference
    model = shape_inference.infer_shapes(model)
    try:
        checker.check_model(model)
    except Exception as e:
        print(f"[modify_batchsize] check_model warning (non-fatal): {e}")

    return model

def parse_args():
    parser = argparse.ArgumentParser()
    # Accept either an integer string ("32") or a symbolic name ("batch_size") for dynamic batch.
    parser.add_argument("--batch_size", type=str)
    parser.add_argument("--origin_model", type=str)
    parser.add_argument("--output_model", type=str)
    parser.add_argument("--strict_mode", action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    model = onnx.load(args.origin_model)
    # Convert to int if the value is a pure digit string; keep as str for symbolic/dynamic batch.
    bsz = int(args.batch_size) if args.batch_size.lstrip('-').isdigit() else args.batch_size
    change_input_dim(model, bsz)
    change_reshape_batch(model, bsz)

    if args.strict_mode:
        model = infer_node_shape(model)

    onnx.save(model, args.output_model)
