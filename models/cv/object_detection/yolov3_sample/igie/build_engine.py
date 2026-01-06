import argparse
import numpy as np
import tvm
from tvm import relay
from tvm.relay.import_model import import_model_to_igie


def main(config):
    target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")
    device = tvm.device(target.kind.name, 0)
    precision = config.precision
    
    inputs_info = {"images": ([config.bsz, 3, 416, 416], "float32")}
    mod, params = import_model_to_igie(config.model, inputs_info, precision=precision, backend="tensorrt")
    lib = relay.build(mod, target=target, params=params, precision=precision, device=device)
    lib.export_library(config.engine)
    print("Build engine done!")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--precision", type=str, choices=["float16", "int8", "float32"], default="int8",
            help="The precision of datatype")
    parser.add_argument("--bsz", type=int)
    # engine args
    parser.add_argument("--engine", type=str, default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)