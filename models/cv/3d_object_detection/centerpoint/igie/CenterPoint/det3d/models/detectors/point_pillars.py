from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy 
import torch 
import tvm, onnxruntime
import numpy as np
@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        export_onnx = False
    ):
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.export_onnx = export_onnx

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        preds = self.bbox_head(x)
        
        #for dump preds
        self.preds = preds

        if self.export_onnx:
            return preds
        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        bev_feature = x 
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, None 

@DETECTORS.register_module
class PointPillarsIGIE(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        export_onnx=False,
        engine_path=None
    ):
        super(PointPillarsIGIE, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.export_onnx = export_onnx

        self.target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")
        self.device = tvm.device(self.target.kind.name, 0)

        lib = tvm.runtime.load_module(test_cfg["engine_path"])
        self.igie_module = tvm.contrib.graph_executor.GraphModule(lib["default"](self.device))

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )        
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )

        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )        
        xx = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )

        igie_tensor = tvm.nd.from_dlpack(xx)
        igie_tensor.fix_DeviceType(self.device.device_type, self.device.device_id)
        
        is_padded = False
        if not igie_tensor.shape[0] == 4:
            is_padded = True
            igie_tensor_np = igie_tensor.numpy()
            igie_tensor_np = np.repeat(igie_tensor_np, 4, axis=0)
            igie_tensor_nd = tvm.nd.array(igie_tensor_np, self.device)
            self.igie_module.set_input('input.1', igie_tensor_nd)   
        else:
            self.igie_module.set_input('input.1', igie_tensor)   

        self.igie_module.run()

        preds = []
        key_list = ['reg', 'height', 'dim', 'rot', 'vel', 'hm']
        for i in range(6):
            pred = {}
            for j in range(len(key_list)):
                if is_padded:
                    pred[key_list[j]] = torch.Tensor(self.igie_module.get_output(i * 6 + j).asnumpy()[0:1]).cuda()
                else:
                    pred[key_list[j]] = torch.Tensor(self.igie_module.get_output(i * 6 + j).asnumpy()).cuda()

            preds.append(pred)

        #for dump preds
        self.preds = preds

        if self.export_onnx:
            return preds
        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        bev_feature = x 
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, None 