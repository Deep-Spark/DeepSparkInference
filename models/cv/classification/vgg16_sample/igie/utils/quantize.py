import os
import psutil
import numpy as np
import logging
import time

import tvm
from tvm import relay
from tvm.relay.quantize import ort_quantize
from tvm.relay.transform import InferType, ToMixedPrecision

from utils.datasets import get_dataloader
import onnx
from onnxruntime.quantization import CalibrationDataReader

import platform
cached_architecture = None

def get_architecture():
    global cached_architecture

    if cached_architecture is not None:
        return cached_architecture

    arch = platform.machine()
    if arch == 'x86_64':
        cached_architecture = 'x86'
    elif arch == 'arm64' or arch == 'aarch64':
        cached_architecture = 'ARM'
    else:
        cached_architecture = 'Unknown'
    
    return cached_architecture

def igie_calibrate_dataset(data_loader, input_name, calibrate_num = 50):
    logging.info('Calibration from dataset... ')
    num = 0
    count = 0
    calibration_data_list = []
    for batch in data_loader:
        if num >= calibrate_num or count > 2:
            break
        data = tvm.nd.array(batch[0])
        num += data.shape[0]
        count += 1
        calibration_data_list.append({input_name: data})

    return calibration_data_list

def igie_calibrate_dataset_from_npy(input_name, calibrate_num = 50):
    logging.info('Calibration from npy file... ')
    
    ### calibrate data npy file lists
    calibrate_file_npy_lists = ["./batch_0.npy"]
    
    calibrate_data = []
    for calibrate_file_npy in calibrate_file_npy_lists:
        if os.path.isfile(calibrate_file_npy):
            calibrate_data.append(np.load(calibrate_file_npy))
    
    num = 0
    calibration_data_list = []
    for i in range(calibrate_num):
        if num >= calibrate_num:
            break
        data = tvm.nd.array(calibrate_data[i])
        num += data.shape[0]
        calibration_data_list.append({input_name: data})
    
    return calibration_data_list

 
def igie_quantize_model(mod, params, args, data_aware=True, scale_file_path=""):
    input_name = args.input_name
    precision = args.precision
    use_ixinfer = args.use_ixinfer
    model_name = args.model_name
    model_format = args.model_format
    
    ### 设置保存量化系数的目录（绝对路径）
    if scale_file_path == "":
        scale_file_dir = os.getcwd() + "/quantize_scale/"
        if not os.path.exists(scale_file_dir):
            os.makedirs(scale_file_dir)
        scale_file_path = scale_file_dir + "quantize_scale_file_%s_%s_%s.npy" % (model_format, model_name, "ixinfer" if use_ixinfer else "igie")
    
    if precision == "fp16" or precision == "float16":
        with tvm.transform.PassContext(opt_level=3):
            mod = InferType()(mod)
            mod = ToMixedPrecision(mixed_precision_type="float16")(mod)
            return mod, params
    elif precision == "int8":
        ori_workers = args.workers
        if get_architecture() == "ARM":
            args.workers = 0
        dataloader = get_dataloader(args)

        calibrate_dataset_func = igie_calibrate_dataset(dataloader, input_name)
        # calibrate_dataset_func = igie_calibrate_dataset_from_npy(quat_data, input_name)
        if use_ixinfer:
            skip_conv_layers=None
        else:
            skip_conv_layers=[0]

        with tvm.transform.PassContext(opt_level=3):
            if data_aware:
                with relay.quantize.qconfig(calibrate_mode="percentile", weight_scale="max", skip_conv_layers=skip_conv_layers, 
                                            skip_dense_layer=False, calibrate_chunk_by=-1, import_scale_file=scale_file_path, skip_group_conv_layers=False):
                    mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset_func)
            else:
                with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0, skip_conv_layers=skip_conv_layers, skip_dense_layer=False):
                    mod = relay.quantize.quantize(mod, params)

        args.workers = ori_workers
        return mod, params
    else:
        logging.info("Model inference percsion is FP32...")
        return mod, params


### ONNXRuntime Quantization
def get_memory_info():
    info = psutil.virtual_memory()
    cur_used = psutil.Process(os.getpid()).memory_info().rss
    total = info.total
    cur_percent = cur_used / total
    total_percent = info.percent
    return cur_used, cur_percent, total_percent


def exceed_memory_upper_bound(upper_bound=90):
    # upper_bound in [0, 100]
    if get_memory_info()[-1] >= upper_bound:
        return True
    return False


class BaseDataReader(CalibrationDataReader):

    def __init__(self, dataloader, input_name="images", cnt_limit=100):
        self.dataloader = dataloader
        self.input_name = input_name
        self.cnt = 0
        self.cnt_limit = cnt_limit
        self.rewind()

    def get_next(self):
        raise NotImplementedError

    def reset_dataloader(self):
        self.dataloader_iter = iter(self.dataloader)

    def rewind(self):
        self.reset_dataloader()
        self.cnt = 0

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader
        self.rewind()

    def should_stop(self, memory_upper_bound=90):
        # avoid OOM
        if self.cnt + 1 > self.cnt_limit:
            return True 
        if exceed_memory_upper_bound(upper_bound=memory_upper_bound):
            print(f"memory usage >= 90%, Retuen None image data.")
            return True 
        self.cnt += 1
        return False

    def get_next_data(self):
        data = next(self.dataloader_iter, None)
        if data is None:
            self.reset_dataloader()
            data = next(self.dataloader_iter, None)
        return data

class ONNXDataReader(BaseDataReader):

    def get_next(self):
        if self.should_stop(memory_upper_bound=90):
            return None
        all_input = self.get_next_data()
        print(f"cnt = {self.cnt}, input_shape = {np.array(all_input[0]).shape}")
        return {self.input_name: all_input[0]}

def get_file_name(path):
    return os.path.splitext(os.path.realpath(path))[0]

def contain_qlinear_opearator(onnx_model):
    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    nodes = onnx_model.graph.node

    for i in nodes:
        if i.op_type.startswith("QLinear"):
            return True
    return False


def get_exclude_node(model_path, model_name, exclude_op_names=[], skip_group_conv=False):
    if isinstance(model_path, str):
        onnx_model = onnx.load(model_path)

    exclude_nodes = []
    if "yolov3" in model_path or "yolov3" in model_name:
        # yolov3 detect部分不做量化
        exclude_nodes = [
            "conv_lbbox/BiasAdd__251", "conv_mbbox/BiasAdd__304", "conv_sbbox/BiasAdd__357", 
            "pred_lbbox/Reshape", "pred_lbbox/strided_slice_2", "pred_lbbox/add", "pred_lbbox/mul", 
            "pred_lbbox/strided_slice_3", "pred_lbbox/Exp", "pred_lbbox/mul_2", "pred_lbbox/strided_slice_4", 
            "pred_lbbox/strided_slice_5", "pred_lbbox/concat_2", "yolo_reshape_2", "pred_mbbox/Reshape",
            "pred_mbbox/strided_slice_2", "pred_mbbox/add", "pred_mbbox/mul", "pred_mbbox/strided_slice_3", "pred_mbbox/Exp",
            "pred_mbbox/mul_2", "pred_mbbox/strided_slice_4", "pred_mbbox/strided_slice_5", "pred_mbbox/concat_2",
            "yolo_reshape_1", "pred_sbbox/Reshape", "pred_sbbox/strided_slice_2",  "pred_sbbox/add", "pred_sbbox/mul",
            "pred_sbbox/strided_slice_3", "pred_sbbox/Exp", "pred_sbbox/mul_2", "pred_sbbox/strided_slice_4",
            "pred_sbbox/strided_slice_5", "pred_sbbox/concat_2", "yolo_reshape_0", "yolo_concat"
        ]
    
    elif "yolov5" in model_path or "yolov5" in model_name:
        # yolov5 detect部分不做量化
        exclude_nodes = [
            "Mul_278", "Mul_284", "Mul_297", "Mul_303", "Mul_316", "Mul_322", 
            "Mul_282", "Mul_288", "Mul_301", "Mul_307", "Mul_320", "Mul_326",
            "Add_280", "Add_299", "Add_318",
            "Concat_289", "Concat_308", "Concat_327", "Reshape_290", "Reshape_309", "Reshape_328",
            "Concat_329"
        ]
    
    elif "yolov7" in model_path or "yolov7" in model_name:
        # yolov7 detect部分不做量化
        exclude_nodes = [
            'Mul_315', 'Mul_321', 'Mul_345', 'Mul_351',   'Mul_375',  'Mul_381',
            'Add_317', 'Add_347', 'Add_377',
            'Concat_322', 'Concat_352', 'Concat_382',
            'Reshape_325', 'Reshape_355', 'Reshape_385', 
            'Concat_386'
        ]
    
    nodes = onnx_model.graph.node
    for op_name in exclude_op_names:
        for node in nodes:
            node_name = node.name
            node_op_type = node.op_type
            if op_name in node_name and node_op_type in op_name:
                exclude_nodes.append(node_name)
                
    if skip_group_conv:
        for index, node in enumerate(nodes):
            node_name = node.name
            node_op_type = node.op_type
            if node_op_type == 'Conv':
                attrs = node.attribute
                for attr in attrs:
                    if attr.name == 'group' and attr.i > 1:
                        exclude_nodes.append(node_name)
     
    return exclude_nodes
                

def onnx_quantize_model(model_path, model_name, precision, calibration_data_reader):
    if precision == "int8":
        dst_onnx_model_path = f"{get_file_name(model_path)}_{precision}.onnx"
        if not os.path.exists(dst_onnx_model_path):
            print(f"\nStart ONNXRuntime Quantization...")
            # check if quantized
            if contain_qlinear_opearator(model_path):
                raise ValueError(f"mode: {model_path} has contains Quant Operator already, set use_onnx_runtime_quant to False plz.")

            nodes_to_exclude = get_exclude_node(model_path,
                                                model_name, 
                                                exclude_op_names=["Softmax", "Gemm", "Pow"], 
                                                skip_group_conv=False)
            
            print(f"ONNXRunitme Quantization nodes_to_exclude: \n{nodes_to_exclude}.\n")
            
            qconfig = ort_quantize.QuantizeConfig(calibration_data_reader, calibrate_method='percentile', nodes_to_exclude=nodes_to_exclude)
            ort_quantize.ort_static_quantize(model_input=model_path, model_output=dst_onnx_model_path, qconfig=qconfig)
            print(f"Get quantized model: {dst_onnx_model_path}")
            return dst_onnx_model_path
        else:
            print(f"Using quantized model: {dst_onnx_model_path}")
            return dst_onnx_model_path
    else:
        logging.info("onnx support INT8 quantization only. Current precision is %s\n", precision)
        raise ValueError("onnx support INT8 quantization only. Current precision is %s\n", precision)