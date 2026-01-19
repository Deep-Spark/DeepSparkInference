import onnx 
import tvm 
from tvm import relax, tir 
from tvm.relax.frontend.onnx import from_onnx

import numpy as np 
import os 

target = tvm.target.iluvatar(model="MR", options="-libs=ixinfer")
device = tvm.iluvatar(0)

model_path = r"centerpoint_e2e_opt.onnx"

onnx_mod = onnx.load(model_path)

# be careful, only one voxels_num.
voxels_num = tir.Var("voxels_num", "int64")
shape_dict = {"voxels": [voxels_num, 20, 5], 
            "num_points": [voxels_num], 
            "coors": [voxels_num, 4]}

mod = from_onnx(onnx_mod, shape_dict=shape_dict)

# skip FuseOps.
ex = relax.build(mod, target=target, precision="fp16", disabled_pass=["FuseOps"])

ex.export_library("centerpoint_e2e_opt.so")