import os
import argparse
import torch
import onnx
from onnx import helper
from onnx import TensorProto, numpy_helper
import tensorrt

from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin()
def create_onnx(args):
    nms = helper.make_node(
        "DetectionNMS_IxRT",
        name="NMS",
        inputs=["nms_input"],
        outputs=["nms_output0", "nms_output1"],
        nMaxKeep=args.max_box_pre_img,
        fIoUThresh=args.iou_thresh,
        fScoreThresh=args.score_thresh
    )
    graph = helper.make_graph(
        nodes=[nms],
        name="gpu_nms",
        inputs=[
            helper.make_tensor_value_info(
                "nms_input", onnx.TensorProto.FLOAT, (args.bsz, args.all_box_num, 6)
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "nms_output0", onnx.TensorProto.FLOAT, (args.bsz, args.max_box_pre_img, 6)
            ),
            helper.make_tensor_value_info(
                "nms_output1", onnx.TensorProto.INT32, (args.bsz,)
            )
        ],
        initializer=[]
    )

    op = onnx.OperatorSetIdProto()
    op.version = 13
    model = onnx.helper.make_model(graph)

    model = onnx.helper.make_model(graph, opset_imports=[op])
    onnx_path = args.path + "/nms.onnx"
    onnx.save(model, onnx_path)

def build_engine(args):
    onnx_path = args.path + "/nms.onnx"
    IXRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(IXRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    build_config = builder.create_builder_config()
    parser = tensorrt.OnnxParser(network, IXRT_LOGGER)
    parser.parse_from_file(onnx_path)
    plan = builder.build_serialized_network(network, build_config)

    engine_path = args.path + "/nms.engine"
    with open(engine_path, "wb") as f:
        f.write(plan)

def main(args):
    create_onnx(args)
    build_engine(args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=1, help="batch size")
    parser.add_argument("--path", type=str)
    parser.add_argument("--all_box_num", type=int, default=25200)
    parser.add_argument("--max_box_pre_img", type=int, default=1000)
    parser.add_argument("--iou_thresh", type=float, default=0.6)
    parser.add_argument("--score_thresh", type=float, default=0.001)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)