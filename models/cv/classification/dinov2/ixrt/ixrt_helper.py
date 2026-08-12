"""ixRT 通用辅助：模块导入、插件加载、engine IO 绑定。

ixRT 的 Python 包同时提供 `ixrt` 与 TensorRT 兼容的 `tensorrt` 两个模块名，
DeepSparkInference 的示例统一用 `tensorrt`，这里做一次兼容导入。
"""

import ctypes
import os

import numpy as np

_PLUGIN_LOADED = False


def import_trt():
    try:
        import tensorrt as trt
    except ImportError:
        import ixrt as trt
    return trt


def load_ixrt_plugin(trt, logger, dynamic_path=""):
    global _PLUGIN_LOADED
    if _PLUGIN_LOADED:
        return
    path = dynamic_path or os.path.join(os.path.dirname(trt.__file__), "lib", "libixrt_plugin.so")
    if not os.path.exists(path):
        print(f"[warn] 未找到 ixrt plugin: {path}，跳过加载（DINOv2 不依赖自定义 plugin）")
        _PLUGIN_LOADED = True
        return
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(logger, "")
    print(f"[ok] 已加载 plugin: {path}")
    _PLUGIN_LOADED = True


def make_logger(trt, verbose=False):
    level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    return trt.Logger(level)


class EngineRunner:
    """加载 ixRT engine，按 binding 顺序分配显存并执行推理。"""

    def __init__(self, engine_path, device=0, verbose=False):
        from cuda import cuda, cudart

        self.cuda, self.cudart = cuda, cudart
        err, = cudart.cudaSetDevice(device)
        assert err == cudart.cudaError_t.cudaSuccess, err

        trt = import_trt()
        self.trt = trt
        self.logger = make_logger(trt, verbose)
        load_ixrt_plugin(trt, self.logger)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine, f"反序列化 engine 失败: {engine_path}"
        self.context = self.engine.create_execution_context()
        assert self.context

        self.inputs, self.outputs, self.allocations = self._alloc_bindings()
        self.batch_size = self.inputs[0]["shape"][0]

    def _iter_binding_meta(self):
        trt, engine = self.trt, self.engine
        if hasattr(engine, "num_bindings"):
            for i in range(engine.num_bindings):
                yield (
                    i,
                    engine.get_binding_name(i),
                    engine.binding_is_input(i),
                    np.dtype(trt.nptype(engine.get_binding_dtype(i))),
                    tuple(engine.get_binding_shape(i)),
                )
        else:
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                yield (
                    i,
                    name,
                    engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
                    np.dtype(trt.nptype(engine.get_tensor_dtype(name))),
                    tuple(engine.get_tensor_shape(name)),
                )

    def _alloc_bindings(self):
        inputs, outputs, allocations = [], [], []
        for index, name, is_input, dtype, shape in self._iter_binding_meta():
            nbytes = int(dtype.itemsize * np.prod(shape))
            err, allocation = self.cudart.cudaMalloc(nbytes)
            assert err == self.cudart.cudaError_t.cudaSuccess, err
            binding = {
                "index": index,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "nbytes": nbytes,
            }
            kind = "input " if is_input else "output"
            print(f"binding {index} [{kind}] {name} dtype={dtype} shape={list(shape)}")
            allocations.append(allocation)
            (inputs if is_input else outputs).append(binding)
        assert inputs, "engine 没有输入 binding"
        return inputs, outputs, allocations

    def set_input(self, data, idx=0):
        binding = self.inputs[idx]
        data = np.ascontiguousarray(data.astype(binding["dtype"]))
        assert data.nbytes == binding["nbytes"], (
            f"输入字节数不匹配: 传入 {data.shape}{data.dtype} vs engine {binding['shape']}{binding['dtype']}"
        )
        err, = self.cuda.cuMemcpyHtoD(binding["allocation"], data, data.nbytes)
        assert err == self.cuda.CUresult.CUDA_SUCCESS, err

    def run(self):
        self.context.execute_v2(self.allocations)

    def sync(self):
        err, = self.cudart.cudaDeviceSynchronize()
        assert err == self.cudart.cudaError_t.cudaSuccess, err

    def fetch_outputs(self):
        results = []
        for binding in self.outputs:
            host = np.zeros(binding["shape"], binding["dtype"])
            err, = self.cuda.cuMemcpyDtoH(host, binding["allocation"], binding["nbytes"])
            assert err == self.cuda.CUresult.CUDA_SUCCESS, err
            results.append(host)
        return results

    def infer(self, data):
        self.set_input(data)
        self.run()
        self.sync()
        return self.fetch_outputs()

    def close(self):
        for allocation in self.allocations:
            self.cudart.cudaFree(allocation)
        self.allocations = []
