# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn


logger = logging.getLogger("dinov2")


class TvmDinoVisionTransformer(nn.Module):
    """TVM Relay engine backed DINOv2 backbone for linear evaluation.

    The engine is built from the ONNX exported by ``export.py`` and is expected
    to output ``cls_tokens`` and ``patch_tokens``. TVM engines have static input
    shapes, so smaller runtime batches are padded to the compiled batch size and
    sliced back after inference.
    """

    bag_of_channels = False

    def __init__(self, engine_path: str):
        super().__init__()
        try:
            import tvm
            from tvm.contrib import graph_executor
        except ImportError as e:
            raise ImportError("TVM inference requires tvm to be importable in this environment") from e

        self.tvm = tvm
        self.target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")
        self.device = tvm.device(self.target.kind.name, 0)
        lib = tvm.runtime.load_module(engine_path)
        self.graph_module = graph_executor.GraphModule(lib["default"](self.device))

        input_nd = self.graph_module.get_input(0)
        self.input_name = "input"
        self.input_shape = tuple(int(dim) for dim in input_nd.shape)
        self.input_dtype = input_nd.dtype

        logger.info(
            "Loaded TVM engine from %s (input_shape=%s, input_dtype=%s)",
            engine_path,
            self.input_shape,
            self.input_dtype,
        )

    def _run_tvm(self, images: torch.Tensor) -> List[torch.Tensor]:
        if len(self.input_shape) != 4:
            raise ValueError(f"Expected 4D TVM input shape, got {self.input_shape}")

        compiled_batch, compiled_channels, compiled_height, compiled_width = self.input_shape
        batch_size, channels, height, width = images.shape
        if (channels, height, width) != (compiled_channels, compiled_height, compiled_width):
            raise ValueError(
                "TVM engine input shape is static. "
                f"Got {(batch_size, channels, height, width)}, "
                f"but engine expects {self.input_shape}."
            )
        if batch_size > compiled_batch:
            raise ValueError(f"Batch size {batch_size} exceeds TVM engine batch size {compiled_batch}")

        padded = images.float()
        if batch_size < compiled_batch:
            pad_shape = (compiled_batch - batch_size, channels, height, width)
            padded = torch.cat([padded, torch.zeros(pad_shape, device=padded.device, dtype=padded.dtype)], dim=0)

        input_np = padded.detach().cpu().numpy().astype(self.input_dtype, copy=False)
        data = self.tvm.nd.array(input_np, self.device)
        try:
            self.graph_module.set_input(self.input_name, data)
        except Exception:
            self.graph_module.set_input(0, data)

        self.graph_module.run()
        outputs = []
        for index in range(self.graph_module.get_num_outputs()):
            output_np = self.graph_module.get_output(index).asnumpy()[:batch_size]
            outputs.append(torch.from_numpy(output_np).to(device=images.device))
        return outputs

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple:
        del norm  # The exported graph already applies LayerNorm per block.
        if reshape:
            raise NotImplementedError("reshape=True is not supported for TVM backend")

        outputs = self._run_tvm(x)
        if len(outputs) < 2:
            raise ValueError("TVM engine must be built from the multi-output ONNX export")

        n_blocks = n if isinstance(n, int) else len(n)
        cls_tokens, patch_tokens = outputs[:2]
        cls_tokens = cls_tokens[:, -n_blocks:, :]
        patch_tokens = patch_tokens[:, -n_blocks:, :, :]
        output = tuple((patch_tokens[:, i], cls_tokens[:, i]) for i in range(n_blocks))

        if return_class_token:
            return output
        return tuple(patch for patch, _ in output)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cls_tokens = self._run_tvm(images)[0]
        return cls_tokens[:, -1, :]

    def benchmark(self, number: int = 100, repeat: int = 1) -> Tuple[float, float]:
        data_np = np.random.random(self.input_shape).astype(self.input_dtype)
        data = self.tvm.nd.array(data_np, self.device)
        try:
            self.graph_module.set_input(self.input_name, data)
        except Exception:
            self.graph_module.set_input(0, data)

        timer = self.graph_module.module.time_evaluator("run", self.device, number=number, repeat=repeat)
        prof_res = np.array(timer().results) * 1000
        mean_time_ms = float(np.mean(prof_res))
        fps = self.input_shape[0] * 1000 / mean_time_ms
        return mean_time_ms, fps
