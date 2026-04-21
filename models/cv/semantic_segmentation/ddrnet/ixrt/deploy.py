import os
import cv2
import argparse
import numpy as np
import torch

from utils import input_transform

from tensorrt import IxRT
from ixrt.common import RuntimeConfig, RuntimeContext
from tensorrt.deploy.api import *


def create_runtime_from_model(args):
    model = args.model
    quant_file = args.quant_file
    precision = args.precision

    config = RuntimeConfig()
    config.input_shapes = [("inputx", [args.bsz, 3, args.imgsz_h, args.imgsz_w])]
    config.device_idx = args.device
    if precision == "int8":
        assert os.path.isfile(quant_file), "Quant file must provided for int8 inferencing"   

    config.runtime_context = RuntimeContext(
        precision,
        "nhwc",
        use_gpu=True,
        pipeline_sync=True,
        input_types={"inputx": "float32"},
        output_types={"outputy": "float32"}
    )
    runtime = IxRT.from_onnx(model, quant_file, config)
    runtime.Init(runtime.config)
    return runtime


def create_runtime_from_engine(engine):
    runtime = IxRT()
    runtime.LoadEngine(engine)
    return runtime


def pre_process(img_file):
    assert os.path.isfile(img_file), "The input file {img_file} must be existed!"
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = input_transform(
        img, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    return img


def main(args):
    print(args)
    img_file = args.img_file
    if args.engine is not None:
        runtime = create_runtime_from_engine(args.engine)
    else:
        runtime = create_runtime_from_model(args)

    input_map = runtime.GetInputShape()
    output_map = runtime.GetOutputShape()
    print(f"input map is: {input_map}")    
    print(f"output map is: {output_map}")    
 
    input_io_buffers = []
    output_io_buffers = []
    for name, shape in input_map.items():
        # 1. apply memory buffer for input of the shape, based on shape and padding
        _shape, _padding = shape.dims, shape.padding
        _shape = [i + j for i, j in zip(_shape, _padding)]
        _shape = [_shape[0], *_shape[2:4], _shape[1]]
        # currently we only support float32 as I/O
        buffer = np.zeros(_shape, dtype=np.float32)
        # 2. load image to the buffer, TODO batch load
        img = pre_process(img_file)
        print("image shape is:", img.shape)

        buffer[0, :, :, :3] = img
        print(f"Allocated input buffer:{_shape}")
    
        # 3. put the buffer to a list
        input_io_buffers.append([name, buffer, shape])

    for name, shape in output_map.items():
        # 1. apply memory buffer for output of the shape 
        # output_buffer = np.zeros(shape.dims, dtype=np.float32) 
        bs, c, h, w = shape.dims
        dims = [bs, h, w, c]

        output_buffer = np.zeros(dims, dtype=np.float32)
        # 2. put the buffer to a list
        output_io_buffers.append([name, output_buffer, shape])
    
    runtime.LoadInput(input_io_buffers)
    runtime.Execute()
    runtime.FetchOutput(output_io_buffers)
    
    print(f"Test Achieved!")    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,  default="ddrnet23.onnx")
    parser.add_argument("--quant_file", type=str, default=None, help="the json of quantization")
    parser.add_argument("--bsz", type=int, default=4, help="batch size")
    parser.add_argument("--precision", type=str, choices=["float16", "int8"], default="int8", help="The precision of datatype")
    parser.add_argument("--warm_up", type=int, default=5, help="warm_up count")
    parser.add_argument("--imgsz_h", type=int, default=1024, help="inference size h")
    parser.add_argument("--imgsz_w", type=int, default=2048, help="inference size w")
    # engine args
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--img_file", type=str, default=None)
    # device
    parser.add_argument(
        "--device", type=int, default=0, help="cuda device, i.e. 0 or 0,1,2,3,4"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
