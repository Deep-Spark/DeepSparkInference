# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import List, Sequence, Tuple, Union

import torch
from torch import nn


logger = logging.getLogger("dinov2")


class OnnxDinoVisionTransformer(nn.Module):
    """ONNX-backed DINOv2 backbone for linear evaluation.

    Supports the full intermediate-layer export produced by ``export.py``
    (``cls_tokens`` + ``patch_tokens``), reproducing
    ``get_intermediate_layers(..., norm=True, return_class_token=True)``.
    Legacy single-output ONNX files (final CLS only) are still accepted.
    """

    bag_of_channels = False

    def __init__(self, onnx_path: str, patch_size: int = 14):
        super().__init__()
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "ONNX inference requires onnxruntime. "
                "Install with: pip install onnxruntime-gpu  (or onnxruntime for CPU-only)"
            ) from e

        providers = []
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.patch_size = patch_size
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.multi_output = len(self.output_names) >= 2

        logger.info(
            "Loaded ONNX model from %s (providers=%s, outputs=%s)",
            onnx_path,
            self.session.get_providers(),
            self.output_names,
        )
        if not self.multi_output:
            logger.warning(
                "Legacy single-output ONNX detected. Re-export with the updated export.py "
                "to get full intermediate-layer outputs for all linear eval configurations."
            )

    def _run_onnx(self, images: torch.Tensor) -> List[torch.Tensor]:
        images = images.float()
        device = images.device
        input_np = images.detach().cpu().numpy()
        outputs_np = self.session.run(self.output_names, {self.input_name: input_np})
        return [torch.from_numpy(output).to(device=device) for output in outputs_np]

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple:
        del norm  # ONNX export already applies LayerNorm per block.
        if reshape:
            raise NotImplementedError("reshape=True is not supported for ONNX backbone")

        n_blocks = n if isinstance(n, int) else len(n)

        if self.multi_output:
            cls_tokens, patch_tokens = self._run_onnx(x)
            cls_tokens = cls_tokens[:, -n_blocks:, :]
            patch_tokens = patch_tokens[:, -n_blocks:, :, :]
            output = tuple((patch_tokens[:, i], cls_tokens[:, i]) for i in range(n_blocks))
        else:
            cls_token = self._run_onnx(x)[0]
            batch_size, _, height, width = x.shape
            num_patches = (height // self.patch_size) * (width // self.patch_size)
            embed_dim = cls_token.shape[-1]
            patch_tokens = torch.zeros(
                batch_size,
                num_patches,
                embed_dim,
                device=cls_token.device,
                dtype=cls_token.dtype,
            )
            block = (patch_tokens, cls_token)
            output = tuple(block for _ in range(n_blocks))

        if return_class_token:
            return output
        return tuple(patch for patch, _ in output)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.multi_output:
            cls_tokens, _ = self._run_onnx(images)
            return cls_tokens[:, -1, :]
        return self._run_onnx(images)[0]
