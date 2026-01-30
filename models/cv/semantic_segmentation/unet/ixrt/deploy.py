import argparse
import json
import os

import numpy as np
import onnx
import collections
from onnxsim import simplify

def onnx_sim(onnx_name, save_name=None):
    #  simplify onnx
    if save_name:
        sim_onnx_name = save_name
    else:
        sim_onnx_name = onnx_name[0:-5] + "_tmp.onnx"
        if os.path.exists(sim_onnx_name):
            os.remove(sim_onnx_name)
    
    onnx_model = onnx.load(onnx_name) # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, sim_onnx_name)
    print("[info] onnxsim done!")
    return sim_onnx_name


def create_graph_json(onnx_name):
    # create graph json and weights
    graph_path = onnx_name[0:-5] + "_graph.json"
    weight_path = onnx_name[0:-5] + ".weights"
    # if os.path.exists(graph_path) and os.path.exists(weight_path):
    #     print("file exists, can run infer scripts")
    #     exit(0)
    model = onnx.load(onnx_name)
    graph = model.graph
    nodes = graph.node
    initializer = graph.initializer
    value_info = graph.value_info  # Infer shape info

    model_inputs = [tensor.name for tensor in graph.input]
    model_outputs = [tensor.name for tensor in graph.output]

    model = {}
    model["nodes"] = {}
    model["tensors"] = {}
    model["edges"] = {}
    model["output"] = {}
    data_type_table = {
        1: "float32",
        2: "uint8",
        3: "int8",
        4: "uint16",
        5: "int16",
        6: "int32",
        7: "int64",
        9: "bool",
        10: "float16",
        11: "double",
        12: "uint32",
        13: "uint64",
    }
    input_cache = []
    for item in graph.input:
        if item.type.tensor_type.elem_type in data_type_table:
            cache = {
                "name": item.name,
                "type": data_type_table[item.type.tensor_type.elem_type],
            }
        else:
            cache = {"name": item.name}
        input_cache.append(cache)
    model["input"] = input_cache

    output_cache = []
    for item in graph.output:
        if item.type.tensor_type.elem_type in data_type_table:
            cache = {
                "name": item.name,
                "type": data_type_table[item.type.tensor_type.elem_type],
            }
        else:
            cache = {"name": item.name}
        output_cache.append(cache)
    model["output"] = output_cache

    # find cast dict
    input_cast_dict = {}
    output_cast_dict = {}
    for i, item in enumerate(nodes):
        node_name = item.name
        input_edge_list = list(item.input)
        output_edge_list = list(item.output)
        # remove input and output cast op
        if item.op_type == "Cast":
            if input_edge_list[0] in model_inputs:
                input_cast_dict[output_edge_list[0]] = input_edge_list[0]
            if output_edge_list[0] in model_outputs:
                output_cast_dict[input_edge_list[0]] = output_edge_list[0]

    for i, item in enumerate(nodes):
        node_name = item.name
        input_edge_list = list(item.input)
        output_edge_list = list(item.output)
        # remove input and output cast op
        if item.op_type == "Cast":
            if input_edge_list[0] in model_inputs:
                continue
            if output_edge_list[0] in model_outputs:
                continue

        for idx, edge_name in enumerate(input_edge_list):
            if edge_name in input_cast_dict.keys():
                input_edge_list[idx] = input_cast_dict[edge_name]

        for idx, edge_name in enumerate(output_edge_list):
            if edge_name in output_cast_dict.keys():
                output_edge_list[idx] = output_cast_dict[edge_name]

        # remove mask in EmbedLayerNormalization
        if item.op_type == "EmbedLayerNormalization":
            no_attention_mask_in_Embed = True
            for input_edge in input_edge_list:
                if "attention_mask" in input_edge:
                    input_edge_list.remove(input_edge)
                    no_attention_mask_in_Embed = False
            if no_attention_mask_in_Embed:
                for tensor_name in model_inputs:
                    if "attention_mask" in tensor_name:
                        output_edge_list[1] = tensor_name

        node_dict = {"inputs": input_edge_list, "outputs": output_edge_list}
        node_dict["op_type"] = item.op_type
        attribute_dict = {}

        if node_name == "":
            for input_edge in input_edge_list:
                node_name += input_edge + "_"
            node_name += "to"
            for output_edge in output_edge_list:
                node_name += "_" + output_edge

        for attr in item.attribute:

            if attr.type == onnx.AttributeProto().AttributeType.FLOAT:
                attribute_dict[attr.name] = attr.f
            if attr.type == onnx.AttributeProto().AttributeType.FLOATS:
                attribute_dict[attr.name] = [x for x in attr.floats]
            if attr.type == onnx.AttributeProto().AttributeType.INT:
                attribute_dict[attr.name] = attr.i
            if attr.type == onnx.AttributeProto().AttributeType.INTS:
                attribute_dict[attr.name] = [x for x in attr.ints]
            if attr.type == onnx.AttributeProto().AttributeType.STRING:
                attribute_dict[attr.name] = str(attr.s.decode("UTF-8"))
            if attr.type == onnx.AttributeProto().AttributeType.STRINGS:
                attribute_dict[attr.name] = [str(x.decode("UTF-8")) for x in attr.strings]

        node_dict["attrbiute"] = attribute_dict
        model["nodes"][node_name] = node_dict

    for i, item in enumerate(initializer):
        tensor_name = item.name
        tensor_dict = {}
        if item.data_type in data_type_table:
            tensor_dict["data_type"] = data_type_table[item.data_type]
        else:
            print(
                tensor_name,
                " use unsupport data type: ",
                item.data_type,
                ", data info will not be saved",
            )
            continue
        tensor_dict["dims"] = list(item.dims)

        model["tensors"][tensor_name] = tensor_dict

    with open(graph_path, "w") as fh:
        json.dump(model, fh, indent=4)


    """
    Export weight
    """
    byte_string = "".encode()

    weight_file_postfix = ".weights"
    for item in initializer:
        tensor_name = item.name

        np_data = None
        if len(item.raw_data):
            np_data = np.frombuffer(item.raw_data, dtype=np.byte)
        elif item.data_type == 1 and len(item.float_data):
            np_data = np.array(list(item.float_data), dtype=np.float32)
        elif item.data_type == 2 and len(item.int32_data):
            np_data = np.array(list(item.int32_data), dtype=np.uint8)
        elif item.data_type == 6 and len(item.int32_data):
            np_data = np.array(list(item.int32_data), dtype=np.int32)
        elif item.data_type == 7 and len(item.int64_data):
            np_data = np.array(list(item.int64_data), dtype=np.int64)
        elif item.data_type == 10 and len(item.int32_data):
            np_data = (
                np.asarray(item.int32_data, dtype=np.uint16)
                .reshape(item.dims)
                .view(np.float16)
            )
        else:
            print(
                "tensor name: ",
                tensor_name,
                ", type: ",
                item.data_type,
                ", len: ",
                len(item.raw_data),
                len(item.float_data),
                len(item.int32_data),
                len(item.int64_data),
                ", will not save into weights file",
            )

        if np_data is not None:
            byte_string += np.uint64(len(tensor_name)).tobytes()
            byte_string += tensor_name.encode()
            np_bytes = np_data.tobytes()
            byte_string += np.uint64(len(np_bytes)).tobytes()
            byte_string += np_bytes


    # Export weight values as bin file
    with open(weight_path, "wb") as fh:
        fh.write(byte_string)
    print("----------------------------")
    print("[OK] graph and weights file save at :")
    print(graph_path)
    print(weight_path)
    return graph_path, weight_path

def parse_args():
    parser = argparse.ArgumentParser("Extract onnx graph weights")
    parser.add_argument("--onnx_name", default="./unet_export.onnx",help="onnx filepath")
    parser.add_argument("--save_dir", default="./unet", help="directory to save output")
    parser.add_argument("--data_type", default="float16", type=str, choices=["float16", "int8"], help="int8 float16")
    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()
    save_dir, onnx_name = config.save_dir, config.onnx_name
    save_name = f"{save_dir}/unet.onnx"
    onnx_sim_name = onnx_sim(onnx_name, save_name)
    if config.data_type == "int8":
        print("Error: not support int8 deploy yet.")
    
    if not os.path.exists(save_name):
        print("Error: convert onnx..")
    else:
        create_graph_json(save_name)


    print("Deploy Unet done..")
