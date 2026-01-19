import mmengine
from mmdeploy.apis import build_task_processor
from mmdeploy.utils import Backend, Codebase, Task, load_config
from mmdeploy.core import RewriterContext
import tvm 
from tvm import relax 

centerpoint_model_cfg = load_config('centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py')[0]

backend_type=Backend.ONNXRUNTIME
deploy_cfg = mmengine.Config(
    dict(
        backend_config=dict(type=backend_type.value),
        onnx_config=dict(
            input_shape=None,
            opset_version=11,
            input_names=['voxels', 'num_points', 'coors'],
            output_names=['outputs']),
        codebase_config=dict(
            type=Codebase.MMDET3D.value, task=Task.VOXEL_DETECTION.value)))

task_processor = build_task_processor(centerpoint_model_cfg, deploy_cfg,
                                        'cpu')

preproc = task_processor.build_data_preprocessor()
collate_data, (voxels, num_points, coors) = task_processor.create_input(
    pcd=  # noqa: E251
    'mmdeploy/tests/test_codebase/test_mmdet3d/data/nuscenes/n008-2018-09-18-12-07-26-0400__LIDAR_TOP__1537287083900561.pcd.bin',  # noqa: E501
    data_preprocessor=preproc)

print(voxels.shape)
print(num_points.shape)
print(coors.shape)

# target = tvm.target.iluvatar(model="MR", options="-libs=ixinfer")
device = tvm.iluvatar(0)

ex = tvm.runtime.load_module("centerpoint_e2e_opt.so")
vm = relax.VirtualMachine(ex, device)
results = vm["main"](tvm.nd.array(voxels.numpy(), device), 
                    tvm.nd.array(num_points.numpy(), device), 
                    tvm.nd.array(coors.numpy(), device))

outputs = dict()
outputs['cls_score'] = results[0]
outputs['bbox_pred'] = results[1]
outputs['dir_cls_pred'] = results[2]
