from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch
import tvm
from tvm import relax

# kDLCUDA — torch CUDA tensors appear as this after from_dlpack (see test_d2d.py).
_CUDA_DEVICE_TYPE = None


def _cuda_device_type():
    """Lazy-resolve torch CUDA's DLPack device_type (normally 2 = kDLCUDA)."""
    global _CUDA_DEVICE_TYPE
    if _CUDA_DEVICE_TYPE is None:
        probe = torch.empty(1, device="cuda")
        nd = tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(probe))
        _CUDA_DEVICE_TYPE = nd.device.device_type
    return _CUDA_DEVICE_TYPE


def torch_cuda_to_igie(tensor, igie_device):
    """Torch CUDA → IGIE NDArray (zero-copy D2D).

    Same pattern as ``igie/tests/test_iluvatar/test_function/test_d2d.py``::

        torch_dlpack = torch.utils.dlpack.to_dlpack(torch_tensor)
        igie_tensor = tvm.nd.from_dlpack(torch_dlpack)
        igie_tensor.fix_DeviceType(device.device_type, device.device_id)
    """
    if not tensor.is_cuda:
        raise TypeError(f"expected CUDA tensor, got device={tensor.device}")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    nd = tvm.nd.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))
    nd.fix_DeviceType(igie_device.device_type, igie_device.device_id)
    return nd


def igie_to_torch_cuda(nd, copy=False):
    """IGIE NDArray → Torch CUDA (D2D).

    * ``copy=False`` (default): zero-copy retag ``iluvatar → kDLCUDA`` then
      ``torch.utils.dlpack.from_dlpack``. Safe for ephemeral VM outputs.
      **Do not** use on GraphExecutor internal buffers without restoring
      device_type afterwards (engine asserts device_type==20).
    * ``copy=True``: allocate a torch CUDA buffer and ``copyto`` into it
      (always safe; matches test_d2d hand-off into a torch-owned buffer).
    """
    if copy:
        torch_dtype = getattr(torch, str(nd.dtype))
        out = torch.empty(tuple(nd.shape), dtype=torch_dtype, device="cuda")
        out_nd = torch_cuda_to_igie(out, nd.device)
        nd.copyto(out_nd)
        return out

    # Zero-copy: retag device_type so torch accepts the DLPack capsule.
    nd.fix_DeviceType(_cuda_device_type(), nd.device.device_id)
    return torch.utils.dlpack.from_dlpack(nd.to_dlpack())


def wrap_torch_buffer_as_igie(tensor, igie_device):
    """Bind a preallocated torch CUDA buffer as an IGIE NDArray (D2D sink)."""
    return torch_cuda_to_igie(tensor, igie_device)


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
        export_onnx=False,
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

        # for dump preds
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
    """PointPillars with IGIE PFE (Relax VM) + RPN (Relay GraphExecutor).

    Device I/O follows ``test_d2d.py`` (DLPack + ``fix_DeviceType``):
    - Torch→IGIE: zero-copy view (``to_dlpack`` → ``from_dlpack`` → retag).
    - PFE out: zero-copy retag back to kDLCUDA for torch.
    - RPN out: ``get_output(i, preallocated_nd)`` D2D into torch-owned buffers
      (must not retag GraphExecutor internal storage; engine asserts type 20).
    - batch=1 pads on GPU (``repeat``), never host ``np.repeat``.
    """

    _RPN_KEYS = ("reg", "height", "dim", "rot", "vel", "hm")
    _RPN_BATCH = 4

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
        engine_path=None,
    ):
        super(PointPillarsIGIE, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.export_onnx = export_onnx

        self.target = tvm.target.iluvatar(
            model="MR", options="-libs=cudnn,cublas,ixinfer"
        )
        self.device = tvm.device(self.target.kind.name, 0)

        pfe_engine_path = test_cfg.get("pfe_engine_path")
        if pfe_engine_path:
            pfe_lib = tvm.runtime.load_module(pfe_engine_path)
            self.pfe_vm = relax.VirtualMachine(pfe_lib, self.device)
        else:
            self.pfe_vm = None

        rpn_engine_path = test_cfg.get(
            "rpn_engine_path", test_cfg.get("engine_path")
        )
        rpn_lib = tvm.runtime.load_module(rpn_engine_path)
        self.rpn_module = tvm.contrib.graph_executor.GraphModule(
            rpn_lib["default"](self.device)
        )
        self._init_rpn_output_buffers()

    def _init_rpn_output_buffers(self):
        """Preallocate torch CUDA sinks; wrap as IGIE NDArrays (test_d2d)."""
        n_out = self.rpn_module.get_num_outputs()
        self._rpn_out_torch = []
        self._rpn_out_nd = []
        for i in range(n_out):
            shape = tuple(self.rpn_module.get_output(i).shape)
            dtype = str(self.rpn_module.get_output(i).dtype)
            buf = torch.empty(
                shape, dtype=getattr(torch, dtype), device="cuda"
            )
            self._rpn_out_torch.append(buf)
            self._rpn_out_nd.append(wrap_torch_buffer_as_igie(buf, self.device))

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

    def _run_pfe(self, data):
        decorated = self.reader.decorate_features(
            data["features"], data["num_voxels"], data["coors"]
        )
        # Torch CUDA → IGIE (zero-copy D2D, test_d2d.py)
        pfe_in = torch_cuda_to_igie(decorated, self.device)
        pfe_out = self.pfe_vm["main"](pfe_in)
        # IGIE → Torch CUDA (zero-copy retag; VM output is ephemeral)
        return igie_to_torch_cuda(pfe_out, copy=False)

    def _run_rpn(self, bev):
        """Run RPN engine; ``bev`` is torch CUDA ``[B,64,512,512]``."""
        batch_size = int(bev.shape[0])
        is_padded = batch_size != self._RPN_BATCH
        if is_padded:
            if batch_size != 1:
                raise ValueError(
                    f"RPN engine expects batch={self._RPN_BATCH} or 1, got {batch_size}"
                )
            # GPU pad (host numpy repeat was ~280ms/frame).
            bev = bev.repeat(self._RPN_BATCH, 1, 1, 1)

        rpn_in = torch_cuda_to_igie(bev, self.device)
        self.rpn_module.set_input("input.1", rpn_in)
        self.rpn_module.run()

        preds = []
        for i in range(6):
            pred = {}
            for j, key in enumerate(self._RPN_KEYS):
                idx = i * 6 + j
                # D2D write into torch-owned buffer (safe for GraphExecutor)
                self.rpn_module.get_output(idx, self._rpn_out_nd[idx])
                t = self._rpn_out_torch[idx]
                pred[key] = t[:batch_size] if is_padded else t
            preds.append(pred)
        return preds

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

        if self.pfe_vm is not None:
            input_features = self._run_pfe(data)
        else:
            input_features = self.reader(
                data["features"], data["num_voxels"], data["coors"]
            )

        bev = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        preds = self._run_rpn(bev)

        # for dump preds
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
class PointPillarsIGIEE2E(SingleStageDetector):
    """PointPillars with a single Relax e2e engine (PFN+Scatter+RPN+Head).

    Inputs to the SO are decorated features ``[N,20,10]`` and ``coors [N,4]``
    (decoration stays in PyTorch). Dynamic ``num_pillars``; batch fixed to 1.
    Device I/O follows ``test_d2d.py`` (DLPack + ``fix_DeviceType``).
    """

    _HEAD_KEYS = ("reg", "height", "dim", "rot", "vel", "hm")
    _HEAD_CHANS = {
        "reg": 2,
        "height": 1,
        "dim": 3,
        "rot": 2,
        "vel": 2,
    }

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
        engine_path=None,
    ):
        super(PointPillarsIGIEE2E, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.export_onnx = export_onnx
        self.target = tvm.target.iluvatar(
            model="MR", options="-libs=cudnn,cublas,ixinfer"
        )
        self.device = tvm.device(self.target.kind.name, 0)

        e2e_path = test_cfg.get(
            "e2e_engine_path", test_cfg.get("engine_path", engine_path)
        )
        if not e2e_path:
            raise ValueError("PointPillarsIGIEE2E requires test_cfg.e2e_engine_path")
        lib = tvm.runtime.load_module(e2e_path)
        self.e2e_vm = relax.VirtualMachine(lib, self.device)
        self._hm_chans = [len(names) for names in self.bbox_head.class_names]

    def _pack_preds(self, outs):
        """36 flat tensors → list[dict] matching CenterHead.forward."""
        preds = []
        idx = 0
        for task_id in range(len(self._hm_chans)):
            pred = {}
            for key in self._HEAD_KEYS:
                t = outs[idx]
                if not isinstance(t, torch.Tensor):
                    t = igie_to_torch_cuda(t, copy=False)
                c = (
                    self._hm_chans[task_id]
                    if key == "hm"
                    else self._HEAD_CHANS[key]
                )
                if t.shape[1] > c:
                    t = t[:, :c]
                pred[key] = t
                idx += 1
            preds.append(pred)
        return preds

    def _run_e2e(self, data):
        decorated = self.reader.decorate_features(
            data["features"], data["num_voxels"], data["coors"]
        )
        coors = data["coors"]
        if coors.dtype != torch.int32:
            coors = coors.to(torch.int32)
        if not coors.is_contiguous():
            coors = coors.contiguous()
        feats_nd = torch_cuda_to_igie(decorated, self.device)
        coors_nd = torch_cuda_to_igie(coors, self.device)
        outs = self.e2e_vm["main"](feats_nd, coors_nd)
        return self._pack_preds(outs)

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]
        batch_size = len(num_voxels)

        if batch_size != 1:
            raise ValueError(
                f"PointPillarsIGIEE2E expects batch_size=1, got {batch_size}"
            )

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )
        preds = self._run_e2e(data)
        self.preds = preds

        if self.export_onnx:
            return preds
        if return_loss:
            return self.bbox_head.loss(example, preds)
        return self.bbox_head.predict(example, preds, self.test_cfg)
