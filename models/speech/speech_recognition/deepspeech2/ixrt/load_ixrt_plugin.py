from os.path import join, dirname, exists
import tensorrt as trt
import ctypes

def load_ixrt_plugin(logger=trt.Logger(trt.Logger.WARNING), namespace="", dynamic_path=""):
    if not dynamic_path:
        dynamic_path = join(dirname(trt.__file__), "lib", "libixrt_plugin.so")
    if not exists(dynamic_path):
        raise FileNotFoundError(
            f"The ixrt_plugin lib {dynamic_path} is not existed, please provided effective plugin path!")
    ctypes.CDLL(dynamic_path, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(logger, namespace)
    print(f"Loaded plugin from {dynamic_path}")