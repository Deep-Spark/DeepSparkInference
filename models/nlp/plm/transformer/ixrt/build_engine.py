import os
import argparse
import tensorrt
from tensorrt import Dims


def build_engine_trtapi_staticshape(config):
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)

    precision = tensorrt.BuilderFlag.INT8 if config.precision == "int8" else tensorrt.BuilderFlag.FP16
    if precision == tensorrt.BuilderFlag.INT8:
        parser.parse_from_files(config.model, config.quant_file)
    else:
        parser.parse_from_file(config.model)

    build_config.set_flag(precision)

    plan = builder.build_serialized_network(network, build_config)
    engine_file_path = config.engine
    with open(engine_file_path, "wb") as f:
        f.write(plan)
    print("Build static shape engine done!")


def build_engine_trtapi_dynamicshape(config):
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()

    profile = builder.create_optimization_profile()
    profile.set_shape("src_tokens", Dims([1, 1]), Dims([56, 43]), Dims([128, 102]))
    build_config.add_optimization_profile(profile)

    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)

    precision = tensorrt.BuilderFlag.INT8 if config.precision == "int8" else tensorrt.BuilderFlag.FP16
    if precision == tensorrt.BuilderFlag.INT8:
        parser.parse_from_files(config.model, config.quant_file)
    else:
        parser.parse_from_file(config.model)

    build_config.set_flag(precision)

    # set dynamic
    num_inputs = network.num_inputs
    for i in range(num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.shape = Dims([-1, -1])

    plan = builder.build_serialized_network(network, build_config)
    engine_file_path = config.engine
    with open(engine_file_path, "wb") as f:
        f.write(plan)
    print("Build dynamic shape engine done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--quant_file", type=str, default=None)
    parser.add_argument("--precision", type=str, choices=["float16", "int8", "float32"], default="float16",
            help="The precision of datatype")
    parser.add_argument("--engine", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    build_engine_trtapi_dynamicshape(args)
