# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Unility functions for Transformer."""

import queue
import random
from typing import List

import numpy as np
import torch

IGNORE_ID = -1

instruct_list = ["You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用东北话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用甘肃话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用贵州话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用河南话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用湖北话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用湖南话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用江西话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用闽南话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用宁夏话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用山西话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用陕西话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用山东话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用上海话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用四川话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用天津话表达。<|endofprompt|>",
                 "You are a helpful assistant. 请用云南话表达。<|endofprompt|>",
                 "You are a helpful assistant. Please say a sentence as loudly as possible.<|endofprompt|>",
                 "You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>",
                 "You are a helpful assistant. 请用尽可能慢地语速说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>",
                 "You are a helpful assistant. 我想体验一下小猪佩奇风格，可以吗？<|endofprompt|>",
                 "You are a helpful assistant. 你可以尝试用机器人的方式解答吗？<|endofprompt|>"]


def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(batchs,
                              max_len,
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 2:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 3:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              xs[0].shape[2],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res


def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# Repetition Aware Sampling in VALL-E 2
def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        weighted_scores[top_ids] = -float('inf')
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)].item()
    return top_ids


def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True).item()
    return top_ids


def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    if fade_in_mel.device == torch.device('cpu'):
        fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask


class TrtContextWrapper:
    def __init__(self, trt_engine, trt_concurrent=1, device='cuda:0'):
        self.trt_context_pool = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        for _ in range(trt_concurrent):
            trt_context = trt_engine.create_execution_context()
            trt_stream = torch.cuda.stream(torch.cuda.Stream(device))
            assert trt_context is not None, 'failed to create trt context, maybe not enough CUDA memory, try reduce current trt concurrent {}'.format(trt_concurrent)
            self.trt_context_pool.put([trt_context, trt_stream])
        assert self.trt_context_pool.empty() is False, 'no avaialbe estimator context'

    def acquire_estimator(self):
        return self.trt_context_pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self.trt_context_pool.put([context, stream])


class OrtContextWrapper:
    def __init__(self, onnx_model, device='cuda:0', providers=None):
        import onnxruntime

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1

        self.device = torch.device(device)
        self.device_id = self.device.index if self.device.index is not None else 0
        available_providers = onnxruntime.get_available_providers()
        if providers is None:
            if self.device.type == 'cuda' and 'CUDAExecutionProvider' in available_providers:
                providers = [('CUDAExecutionProvider', {'device_id': self.device_id}), 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

        self.session = onnxruntime.InferenceSession(onnx_model, sess_options=option, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.input_dtypes = {i.name: self._torch_dtype(i.type) for i in self.session.get_inputs()}
        self.output_name = self.session.get_outputs()[0].name
        self.output_dtype = self._torch_dtype(self.session.get_outputs()[0].type)
        self.use_cuda_io = self.device.type == 'cuda' and 'CUDAExecutionProvider' in self.session.get_providers()

    @staticmethod
    def _torch_dtype(ort_type):
        if ort_type == 'tensor(float16)':
            return torch.float16
        if ort_type == 'tensor(double)':
            return torch.float64
        return torch.float32

    @staticmethod
    def _numpy_dtype(torch_dtype):
        if torch_dtype == torch.float16:
            return np.float16
        if torch_dtype == torch.float64:
            return np.float64
        return np.float32

    def _prepare_inputs(self, x, mask, mu, t, spks, cond):
        inputs = {
            'x': x,
            'mask': mask,
            'mu': mu,
            't': t,
            'spks': spks,
            'cond': cond,
        }
        return {
            name: inputs[name].to(dtype=self.input_dtypes.get(name, torch.float32)).contiguous()
            for name in self.input_names
        }

    def run(self, x, mask, mu, t, spks, cond):
        inputs = self._prepare_inputs(x, mask, mu, t, spks, cond)
        if self.use_cuda_io and x.device.type == 'cuda':
            output = torch.empty(x.shape, device=x.device, dtype=self.output_dtype)
            device_id = x.device.index if x.device.index is not None else self.device_id
            io_binding = self.session.io_binding()
            for name, tensor in inputs.items():
                io_binding.bind_input(name,
                                      'cuda',
                                      device_id,
                                      self._numpy_dtype(tensor.dtype),
                                      tuple(tensor.shape),
                                      tensor.data_ptr())
            io_binding.bind_output(self.output_name,
                                   'cuda',
                                   device_id,
                                   self._numpy_dtype(output.dtype),
                                   tuple(output.shape),
                                   output.data_ptr())
            self.session.run_with_iobinding(io_binding)
            return output

        ort_inputs = {name: tensor.detach().cpu().numpy() for name, tensor in inputs.items()}
        output = self.session.run([self.output_name], ort_inputs)[0]
        return torch.from_numpy(output).to(device=x.device)


class IgieContextWrapper:
    def __init__(self, igie_model, device='iluvatar:0'):
        import tvm
        from tvm import relax

        self.tvm = tvm
        self.tvm_device = tvm.device(device.split(':')[0], int(device.split(':')[1]))
        assert hasattr(tvm, 'iluvatar'), 'igie inference requires tvm.iluvatar device support'
        ex = tvm.runtime.load_module(igie_model)
        self.vm = relax.VirtualMachine(ex, device=self.tvm_device)

    def _to_tvm_ndarray(self, tensor):
        tensor = tensor.detach().to(dtype=torch.float32).contiguous()
        tensor_dlpack = torch.utils.dlpack.to_dlpack(tensor)

        # D2D zero copy.
        tensor_tvm = self.tvm.nd.from_dlpack(tensor_dlpack)
        tensor_tvm.fix_DeviceType(self.tvm_device.device_type, self.tvm_device.device_id)
        return tensor_tvm

    @staticmethod
    def _first_output(output):
        if hasattr(output, 'numpy'):
            return output
        return output[0]

    def run(self, x, mask, mu, t, spks, cond):
        inputs = [
            self._to_tvm_ndarray(x),
            self._to_tvm_ndarray(mask),
            self._to_tvm_ndarray(mu),
            self._to_tvm_ndarray(t),
            self._to_tvm_ndarray(spks),
            self._to_tvm_ndarray(cond),
        ]
        output = self._first_output(self.vm['main'](*inputs))
        return torch.from_numpy(output.numpy()).to(device=x.device, dtype=x.dtype)
