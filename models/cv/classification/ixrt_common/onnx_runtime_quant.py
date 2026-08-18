import os
import argparse
import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision import transforms as T
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    CalibrationDataReader,
    QuantFormat,
    CalibrationMethod
)


# Patch: handle silent C++ failures in infer_shapes_path (e.g. for dynamic-batch models).
# When infer_shapes_path succeeds, the patch is transparent.
# When it doesn't produce an output file, fall back to in-memory shape inference.
_orig_infer_shapes_path = onnx.shape_inference.infer_shapes_path
def _robust_infer_shapes_path(input_path: str, output_path: str, **kwargs):
    try:
        _orig_infer_shapes_path(input_path, output_path, **kwargs)
    except Exception:
        pass
    if not Path(output_path).exists():
        print(f"[ort_quant patch] infer_shapes_path did not produce {output_path}, using in-memory fallback")
        model = onnx.load(input_path)
        inferred = onnx.shape_inference.infer_shapes(model)
        onnx.save(inferred, output_path)
onnx.shape_inference.infer_shapes_path = _robust_infer_shapes_path

class CalibrationImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(CalibrationImageNet, self).__init__(*args, **kwargs)
        img2label_path = os.path.join(self.root, "val_map.txt")
        if not os.path.exists(img2label_path):
            raise FileNotFoundError(f"Not found label file `{img2label_path}`.")

        self.img2label_map = self.make_img2label_map(img2label_path)

    def make_img2label_map(self, path):
        with open(path) as f:
            lines = f.readlines()

        img2lable_map = dict()
        for line in lines:
            line = line.lstrip().rstrip().split("\t")
            if len(line) != 2:
                continue
            img_name, label = line
            img_name = img_name.strip()
            if img_name in [None, ""]:
                continue
            label = int(label.strip())
            img2lable_map[img_name] = label
        return img2lable_map

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        img_name = os.path.basename(path)
        target = self.img2label_map[img_name]

        return sample, target

def create_dataloaders(data_path, num_samples=1024, img_sz=224, batch_size=2, workers=0, drop_last=False):
    resize_sz = int(round(img_sz * 256 / 224))
    dataset = CalibrationImageNet(
        data_path,
        transform=T.Compose(
            [
                T.Resize(resize_sz),
                T.CenterCrop(img_sz),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    calibration_dataset = dataset
    if num_samples is not None:
        calibration_dataset = torch.utils.data.Subset(
            dataset, indices=range(num_samples)
        )

    calibration_dataloader = DataLoader(
        calibration_dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=workers,
    )

    verify_dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=workers,
    )

    return calibration_dataloader, verify_dataloader


def getdataloader(dataset_dir, step=32, batch_size=1, workers=4, img_sz=224, total_sample=100, drop_last=False):
    num_samples = total_sample
    if step < 0:
        num_samples = None
    calibration_dataloader, _ = create_dataloaders(
        dataset_dir,
        img_sz=img_sz,
        batch_size=batch_size,
        workers=workers,
        num_samples=num_samples,
        drop_last=drop_last,
    )
    return calibration_dataloader

class TorchCalibrationDataReader(CalibrationDataReader):
    """基于PyTorch DataLoader的校准数据读取器"""
    def __init__(self, dataloader, input_name):
        """
        初始化校准数据读取器
        :param dataloader: PyTorch DataLoader对象
        :param input_name: ONNX模型输入节点名称
        """
        self.dataloader = dataloader
        self.input_name = input_name
        self.iterator = iter(dataloader)
        self._reset()
    
    def _reset(self):
        """重置迭代器"""
        self.iterator = iter(self.dataloader)
    
    def get_next(self):
        """获取下一个校准样本（转换为numpy格式）"""
        try:
            # 获取数据并转换为numpy
            data, _ = next(self.iterator)  # 忽略标签
            if isinstance(data, torch.Tensor):
                data_np = data.cpu().numpy()  # 转换为numpy数组
                return {self.input_name: data_np}
            elif isinstance(data, dict):
                # 处理字典类型的输入
                return {k: v.cpu().numpy() for k, v in data.items() if k == self.input_name}
            return None
        except StopIteration:
            return None  # 数据耗尽

def create_torch_dataloader(data_dir, input_shape, batch_size=1, num_workers=4, drop_last=False, total_sample=100):
    """
    创建PyTorch DataLoader
    :param data_dir: 数据目录
    :param input_shape: 模型输入形状 (C, H, W)
    :param batch_size: 批次大小（需与模型静态 batch 一致）
    :param drop_last: 是否丢弃不完整的最后一个 batch
    :param total_sample: 校准数据集总样本数（实际使用图像数）
    :return: DataLoader对象
    """
    img_sz = input_shape[1] if len(input_shape) >= 2 else 224
    # 使用 step=-1 绕过 getdataloader 内部 min(total_sample, step*batch_size) 截断，
    # 然后在 create_dataloaders 中直接以 total_sample 作为 num_samples。
    dataloader = getdataloader(data_dir, batch_size=batch_size, workers=num_workers, drop_last=drop_last, img_sz=img_sz, total_sample=total_sample)
    return dataloader

def get_onnx_input_info(onnx_model_path):
    """获取ONNX模型输入信息"""
    model = onnx.load(onnx_model_path)
    input_tensor = model.graph.input[0]
    input_name = input_tensor.name
    
    # 解析输入形状 (假设格式为 [batch, channel, height, width])
    input_shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        input_shape.append(dim.dim_value if dim.dim_value != 0 else 1)
    
    return input_name, tuple(input_shape)

def quantize_with_torch_data(input_model, output_model, calibration_dir, num_samples=100, batch_size=1, per_channel=True, op_types_to_quantize=None, nodes_to_exclude=None, calibrate_method='Entropy'):
    """使用PyTorch DataLoader加载数据进行量化"""
    # 获取模型输入信息
    input_name, input_shape = get_onnx_input_info(input_model)
    print(f"模型输入: {input_name}, 形状: {input_shape}")
    
    # 提取输入维度 (C, H, W)
    if len(input_shape) == 4:
        _, c, h, w = input_shape
    else:
        raise ValueError(f"不支持的输入形状: {input_shape}")
    
    # 当 batch_size > 1 时启用 drop_last，避免最后一个不完整 batch 与静态输入维度不匹配
    # num_samples 表示校准批次数（batches），因此数据集总样本数 = num_samples * batch_size，
    # 确保 drop_last=True 时有足够的完整 batch 可用。
    dataloader = create_torch_dataloader(
        data_dir=calibration_dir,
        input_shape=(c, h, w),
        batch_size=batch_size,
        num_workers=4,
        drop_last=(batch_size > 1),
        total_sample=num_samples * batch_size,
    )
    
    # 限制校准批次数量（每次 __next__ 返回一个 batch）
    class LimitedDataLoader:
        def __init__(self, dataloader, max_batches):
            self.dataloader = dataloader
            self.max_batches = max_batches
            self.count = 0
            self._iter = None

        def __iter__(self):
            self.count = 0
            self._iter = iter(self.dataloader)
            return self

        def __next__(self):
            if self.count >= self.max_batches:
                raise StopIteration
            try:
                item = next(self._iter)
            except StopIteration:
                raise StopIteration
            self.count += 1
            return item
    
    limited_dataloader = LimitedDataLoader(dataloader, num_samples)
    
    # 创建校准数据读取器
    calib_reader = TorchCalibrationDataReader(
        dataloader=limited_dataloader,
        input_name=input_name
    )
    
    # 执行静态量化
    # per_channel=True 适合 CNN（Conv 权重有明确的 output_channel 轴）
    # per_channel=False 适合 Transformer（MatMul 权重的轴约定与 IxRT 期望可能不符）
    # op_types_to_quantize=None 时量化所有支持的算子类型；
    #   对 Transformer 模型建议只量化 ['MatMul']，避免 IxRT 解析 LayerNorm/Softmax 的 QDQ 出错
    calib_method_map = {
        'Entropy': CalibrationMethod.Entropy,
        'MinMax': CalibrationMethod.MinMax,
        'Percentile': CalibrationMethod.Percentile,
    }
    calib_method = calib_method_map.get(calibrate_method, CalibrationMethod.Entropy)
    quantize_static(
        model_input=input_model,
        model_output=output_model,
        calibration_data_reader=calib_reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        op_types_to_quantize=op_types_to_quantize,
        nodes_to_exclude=nodes_to_exclude or [],
        calibrate_method=calib_method,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "ZeroPoint": 0,
            "QuantizeBias": False,
            "EnableSubgraph": True,
            "Axis": False,
        }
    )
    
    print(f"量化完成，模型保存至: {output_model}")
    return output_model

def verify_quantization(original_model, quantized_model, test_dataloader):
    """使用PyTorch DataLoader验证量化前后的精度差异"""
    import onnxruntime as ort
    
    # 创建ONNX Runtime会话
    sess_original = ort.InferenceSession(original_model, providers=["CPUExecutionProvider"])
    sess_quantized = ort.InferenceSession(quantized_model, providers=["CPUExecutionProvider"])
    
    input_name = sess_original.get_inputs()[0].name
    output_name = sess_original.get_outputs()[0].name
    
    total = 0
    correct_original = 0
    correct_quantized = 0
    
    for data, labels in test_dataloader:
        # 转换为numpy
        input_data = data.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # 原始模型推理
        outputs_original = sess_original.run([output_name], {input_name: input_data})[0]
        preds_original = np.argmax(outputs_original, axis=1)
        
        # 量化模型推理
        outputs_quantized = sess_quantized.run([output_name], {input_name: input_data})[0]
        preds_quantized = np.argmax(outputs_quantized, axis=1)
        
        # 统计正确数
        correct_original += np.sum(preds_original == labels_np)
        correct_quantized += np.sum(preds_quantized == labels_np)
        total += len(labels_np)
        
        # 打印进度
        if total % 100 == 0:
            print(f"已验证 {total} 个样本...")
    
    # 计算精度
    acc_original = correct_original / total if total > 0 else 0
    acc_quantized = correct_quantized / total if total > 0 else 0
    
    print(f"\n原始模型精度: {acc_original:.4f}")
    print(f"量化模型精度: {acc_quantized:.4f}")
    print(f"精度损失: {acc_original - acc_quantized:.4f}")

def convert_fc_weights_to_per_tensor(model_path, output_path, op_types=("Gemm", "MatMul")):
    """Convert per-channel weight quantization of the given op types back to per-tensor.

    The previous ``tensorrt.deploy.static_quantize`` flow quantized Conv weights
    per-channel but kept fully-connected (Gemm/MatMul) weights per-tensor. ONNX
    Runtime's ``per_channel=True`` is a global switch that also makes FC weights
    per-channel, for which IxRT has no efficient fused INT8 GEMM kernel -- this
    regresses FPS on FC-dominated models such as VGG16. Re-quantizing the FC
    weights per-tensor restores the original performance.
    """
    model = onnx.load(model_path)
    graph = model.graph
    init_idx = {init.name: i for i, init in enumerate(graph.initializer)}
    dq_by_out = {n.output[0]: n for n in graph.node if n.op_type == "DequantizeLinear"}

    for node in graph.node:
        if node.op_type not in op_types:
            continue
        if len(node.input) < 2:
            continue
        dq = dq_by_out.get(node.input[1])
        if dq is None:
            continue
        q_name, s_name = dq.input[0], dq.input[1]
        zp_name = dq.input[2] if len(dq.input) > 2 else None
        if s_name not in init_idx or q_name not in init_idx:
            continue
        scale = numpy_helper.to_array(graph.initializer[init_idx[s_name]])
        if scale.size <= 1:
            continue  # already per-tensor
        q = numpy_helper.to_array(graph.initializer[init_idx[q_name]]).astype(np.float32)
        if zp_name in init_idx:
            zp = numpy_helper.to_array(graph.initializer[init_idx[zp_name]]).astype(np.float32)
        else:
            zp = np.zeros_like(scale)
        axis = next((a.i for a in dq.attribute if a.name == "axis"), 0)
        shape = [1] * q.ndim
        shape[axis] = scale.size
        w_float = (q - zp.reshape(shape)) * scale.reshape(shape)
        amax = float(np.max(np.abs(w_float)))
        s_pt = amax / 127.0 if amax > 0 else 1.0
        q_new = np.clip(np.round(w_float / s_pt), -127, 127).astype(np.int8)

        graph.initializer[init_idx[q_name]].CopyFrom(numpy_helper.from_array(q_new, q_name))
        graph.initializer[init_idx[s_name]].CopyFrom(
            numpy_helper.from_array(np.array(s_pt, dtype=np.float32), s_name)
        )
        if zp_name in init_idx:
            graph.initializer[init_idx[zp_name]].CopyFrom(
                numpy_helper.from_array(np.array(0, dtype=np.int8), zp_name)
            )
        for i in reversed([i for i, a in enumerate(dq.attribute) if a.name == "axis"]):
            del dq.attribute[i]
        print(f"Converted {node.op_type} weight '{q_name}' to per-tensor quantization")
    onnx.save(model, output_path)


def remove_quantize_axis_attribute(model_path, output_path):
    model = onnx.load(model_path)

    quantize_node_types = {
        "QuantizeLinear",
        "DequantizeLinear",
        "DynamicQuantizeLinear"
    }

    # Build the set of constant initializer tensor names (= weight/bias tensors).
    # DequantizeLinear nodes whose first input (the quantized tensor X) is a
    # constant initializer are *weight* DQ nodes; their per-channel axis=0 is
    # semantically correct and must be preserved so that ONNX Runtime (and IxRT)
    # can broadcast the scale along the output-channel dimension.
    #
    # Removing axis from weight DQ nodes causes the runtime to fall back to the
    # ONNX default axis=1 (input-channel dim), which does not match the scale
    # size when out_ch != in_ch (e.g. the stem Conv [32, 3, 3, 3] has
    # scale[32] but in_ch=3), leading to a runtime error or wrong scale mapping.
    #
    # Activation QDQ nodes never carry a per-channel axis with op_types_to_quantize
    # restricted to Conv (activations are always per-tensor), so removing axis
    # from them is a no-op.  We keep the loop as a safety net for future changes.
    init_names = {init.name for init in model.graph.initializer}

    for node in model.graph.node:
        if node.op_type not in quantize_node_types:
            continue

        # Skip weight DequantizeLinear nodes – preserve their axis attribute.
        if (node.op_type == "DequantizeLinear"
                and len(node.input) > 0
                and node.input[0] in init_names):
            continue

        to_remove = [i for i, attr in enumerate(node.attribute) if attr.name == "axis"]
        for i in reversed(to_remove):
            del node.attribute[i]
            print(f"Removed 'axis' attribute from {node.op_type} node (name: {node.name})")

    onnx.save(model, output_path)
    print(f"Modified model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='使用PyTorch DataLoader的ONNX量化工具')
    parser.add_argument('--input', required=True, help='输入ONNX模型路径')
    parser.add_argument('--calibration_dir', required=True, help='校准数据目录')
    parser.add_argument('--model_name', required=True, help='模型名称')
    parser.add_argument('--test_dir', help='测试数据目录（用于精度验证）')
    parser.add_argument('--save_dir')
    parser.add_argument('--num_samples', type=int, default=100, 
                      help='校准样本数量')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='校准 DataLoader 的 batch size，需与模型静态 batch 一致（默认 1）')
    parser.add_argument('--per_channel', type=lambda x: x.lower() != 'false', default=True,
                      help='是否使用 per-channel 量化（CNN 模型用 True，Transformer 模型用 False，默认 True）')
    parser.add_argument('--op_types_to_quantize', nargs='*', default=None,
                      help='限制量化的算子类型列表，例如 --op_types_to_quantize MatMul Conv；'
                           '不传则量化所有支持的算子（默认）')
    parser.add_argument('--nodes_to_exclude', nargs='*', default=None,
                      help='从量化中排除的具体节点名称列表，例如 --nodes_to_exclude Conv_95 Conv_98；'
                           '常用于跳过对精度敏感的 SE 模块等节点（默认不排除任何节点）')
    parser.add_argument('--calibrate_method', type=str, default='Entropy',
                      choices=['Entropy', 'MinMax', 'Percentile'],
                      help='校准方法（默认 Entropy；SiLU 激活模型可尝试 MinMax 或 Percentile）')
    parser.add_argument('--fc_per_tensor', action='store_true', default=False,
                      help='将全连接层（Gemm/MatMul）权重的量化由 per-channel 转回 per-tensor。'
                           'IxRT 无高效的 per-channel INT8 GEMM kernel，FC 占比高的模型（如 VGG16）'
                           '在 per-channel 下会走慢路径导致 FPS 下降；仅对这类模型开启（默认关闭）。')

    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.input):
        print(f"错误：输入模型不存在 - {args.input}")
        return
    
    if not os.path.isdir(args.calibration_dir):
        print(f"错误：校准数据目录不存在 - {args.calibration_dir}")
        return
    # onnx runtime 老版本不支持低于opset 13的 onnx模型qdq量化
    model = onnx.load(args.input)
    opset = model.opset_import[0].version
    if opset < 13:
        print(f"opset version: {opset}, convert to 13")
        model = onnx.version_converter.convert_version(model, 13)
        # Re-run shape inference after opset conversion to re-populate batch-size
        # dependent shape annotations that the conversion may have reset.
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, args.input)

    # 执行量化
    quantized_model = quantize_with_torch_data(
        input_model=args.input,
        output_model=f"{args.save_dir}/quantized_{args.model_name}.onnx",
        calibration_dir=args.calibration_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        per_channel=args.per_channel,
        op_types_to_quantize=args.op_types_to_quantize if args.op_types_to_quantize else None,
        nodes_to_exclude=args.nodes_to_exclude if args.nodes_to_exclude else None,
        calibrate_method=args.calibrate_method,
    )
    # Opt-in only (via --fc_per_tensor): match the legacy deploy flow by keeping
    # Conv weights per-channel while making fully-connected (Gemm/MatMul) weights
    # per-tensor, avoiding the IxRT per-channel GEMM slow path that regresses FPS
    # on FC-heavy models such as VGG16. Off by default so other models are unaffected.
    if args.fc_per_tensor:
        convert_fc_weights_to_per_tensor(quantized_model, quantized_model)
    remove_quantize_axis_attribute(quantized_model, f"{args.save_dir}/quantized_{args.model_name}.onnx")
    
    # 验证精度（如果提供测试数据）
    if args.test_dir and os.path.isdir(args.test_dir):
        print("\n开始验证量化模型精度...")
        _, input_shape = get_onnx_input_info(args.input)
        _, c, h, w = input_shape
        
        # 创建测试数据加载器
        test_dataloader = create_torch_dataloader(
            data_dir=args.test_dir,
            input_shape=(c, h, w),
            batch_size=1,
            num_workers=4
        )
        
        verify_quantization(args.input, quantized_model, test_dataloader)

if __name__ == "__main__":
    main()
    
