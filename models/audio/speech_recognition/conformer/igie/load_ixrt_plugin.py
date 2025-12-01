import ctypes
import tensorrt
from os.path import join, dirname, exists
def load_ixrt_plugin(logger=tensorrt.Logger(tensorrt.Logger.INFO), namespace="", dynamic_path=""):
    if not dynamic_path:
        dynamic_path = join(dirname(tensorrt.__file__), "lib", "libixrt_plugin.so")
    if not exists(dynamic_path):
        raise FileNotFoundError(
            f"The ixrt_plugin lib {dynamic_path} is not existed, please provided effective plugin path!")
    ctypes.CDLL(dynamic_path)
    tensorrt.init_libnvinfer_plugins(logger, namespace)
    print(f"Loaded plugin from {dynamic_path}")
