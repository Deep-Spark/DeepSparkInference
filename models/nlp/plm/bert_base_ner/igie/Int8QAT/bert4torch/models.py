# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import copy
import json
import re
import warnings

import torch
import torch.nn as nn
from bert4torch.activations import get_activation
from bert4torch.layers import (
    AdaptiveEmbedding,
    BertEmbeddings,
    BertLayer,
    GatedAttentionUnit,
    Identity,
    LayerNorm,
    T5Layer,
    XlnetLayer,
    XlnetPositionsEncoding,
)
from bert4torch.snippets import (
    FGM,
    PGD,
    VAT,
    EarlyStopping,
    IterDataset,
    ProgbarLogger,
    delete_arguments,
    get_kw,
    insert_arguments,
    metric_mapping,
    search_layer,
    take_along_dim,
)


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        (
            self.global_step,
            self.local_step,
            self.total_steps,
            self.epoch,
            self.train_dataloader,
        ) = (0, 0, 0, 0, None)
        self.callbacks = []

    def compile(
        self,
        loss,
        optimizer,
        scheduler=None,
        max_grad_norm=None,
        use_amp=False,
        metrics=None,
        adversarial_train={"name": ""},
    ):
        self.criterion = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        if use_amp:
            assert adversarial_train["name"] not in {
                "vat",
                "gradient_penalty",
            }, "Amp and adversarial_train both run is not supported in current version"
            from torch.cuda.amp import autocast

            self.autocast = autocast
            self.scaler = torch.cuda.amp.GradScaler()

        if metrics is None:
            metrics = []
        self.metrics = ["loss"] + [i for i in metrics if i != "loss"]

        # 对抗训练
        self.adversarial = adversarial_train
        self.adversarial_initialize()

    def adversarial_initialize(self):
        assert self.adversarial["name"] in {
            "",
            "fgm",
            "pgd",
            "vat",
            "gradient_penalty",
        }, "adversarial_train support fgm, pgd, vat and gradient_penalty mode"
        self.adversarial["epsilon"] = self.adversarial.get("epsilon", 1.0)
        self.adversarial["emb_name"] = self.adversarial.get(
            "emb_name", "word_embeddings"
        )

        if self.adversarial["name"] == "fgm":
            self.ad_train = FGM(self)
        elif self.adversarial["name"] == "pgd":
            self.adversarial["K"] = self.adversarial.get("K", 3)  # 步数
            self.adversarial["alpha"] = self.adversarial.get("alpha", 0.3)  # 学习率
            self.ad_train = PGD(self)
        elif self.adversarial["name"] == "gradient_penalty":
            pass
        elif self.adversarial["name"] == "vat":
            self.adversarial["K"] = self.adversarial.get("K", 3)
            self.adversarial["noise_var"] = self.adversarial.get(
                "noise_var", 1e-5
            )
            self.adversarial["noise_gamma"] = self.adversarial.get(
                "noise_gamma", 1e-6
            )
            self.adversarial["adv_step_size"] = self.adversarial.get(
                "adv_step_size", 1e-3
            )
            self.adversarial["adv_alpha"] = self.adversarial.get(
                "adv_alpha", 1
            )
            self.adversarial["norm_type"] = self.adversarial.get(
                "norm_type", "l2"
            )
            self.ad_train = VAT(self, **self.adversarial)

    def adversarial_training(
        self, train_X, train_y, output, loss, loss_detail, grad_accumulation_steps
    ):
        """对抗训练"""
        if self.adversarial["name"] == "fgm":
            self.ad_train.attack(**self.adversarial)
            output, loss, loss_detail = self.train_step(
                train_X, train_y, grad_accumulation_steps
            )
            loss.backward()
            self.ad_train.restore(**self.adversarial)
        elif self.adversarial["name"] == "pgd":
            self.ad_train.backup_grad()
            for t in range(self.adversarial["K"]):
                self.ad_train.attack(**self.adversarial, is_first_attack=(t == 0))
                if t != self.adversarial["K"] - 1:
                    self.optimizer.zero_grad()
                else:
                    self.ad_train.restore_grad()
                output, loss, loss_detail = self.train_step(
                    train_X, train_y, grad_accumulation_steps
                )
                loss.backward()
            self.ad_train.restore(**self.adversarial)
        elif self.adversarial["name"] == "gradient_penalty":
            para = search_layer(self, self.adversarial["emb_name"], retrun_first=True)
            gp = (para.grad**2).sum()
            loss += 0.5 * gp * self.adversarial["epsilon"]
            loss.backward()
        elif self.adversarial["name"] == "vat":
            logit = output[0] if isinstance(output, (list, tuple)) else output
            adv_loss = self.ad_train.virtual_adversarial_training(train_X, logit)
            loss_detail.update({"loss_sup": loss.item(), "loss_unsup": adv_loss})
            loss += adv_loss if adv_loss else 0
            loss.backward()

        return loss, loss_detail

    def train_step(self, train_X, train_y, grad_accumulation_steps):

        def args_segmentate(train_X):
            if isinstance(train_X, torch.Tensor):
                pass
            elif isinstance(self, (BaseModelDP, BaseModelDDP)):
                if self.module.forward.__code__.co_argcount >= 3:
                    return True
            elif self.forward.__code__.co_argcount >= 3:
                return True
            return False

        if self.use_amp:
            with self.autocast():
                output = (
                    self.forward(*train_X)
                    if args_segmentate(train_X)
                    else self.forward(train_X)
                )
                loss_detail = self.criterion(output, train_y)
        else:
            output = (
                self.forward(*train_X)
                if args_segmentate(train_X)
                else self.forward(train_X)
            )
            loss_detail = self.criterion(output, train_y)

        if isinstance(loss_detail, torch.Tensor):
            loss = loss_detail
            loss_detail = {}
        elif isinstance(loss_detail, dict):
            loss = loss_detail["loss"]
            del loss_detail["loss"]
        elif isinstance(loss_detail, (tuple, list)):
            loss = loss_detail[0]
            loss_detail = {
                f"loss{i}": v for i, v in enumerate(loss_detail[1:], start=1)
            }
        else:
            raise ValueError("Return loss only support Tensor/dict/tuple/list format")
        # 梯度累积
        loss = loss / grad_accumulation_steps if grad_accumulation_steps > 1 else loss
        return output, loss, loss_detail

    def callback_fun(self, mode, logs={}):
        if (
            isinstance(self, BaseModelDDP)
            and self.master_rank != torch.distributed.get_rank()
        ):
            return

        if mode == "train_begin":
            for callback in self.callbacks:
                callback.on_train_begin()
        elif mode == "epoch_begin":
            for callback in self.callbacks:
                callback.on_epoch_begin(self.global_step, self.epoch, logs)
        elif mode == "batch_begin":
            for callback in self.callbacks:
                callback.on_batch_begin(self.global_step, self.local_step, logs)
        elif mode == "batch_end":
            for callback in self.callbacks:
                callback.on_batch_end(self.global_step, self.local_step, logs)
        elif mode == "epoch_end":
            for callback in self.callbacks:
                callback.on_epoch_end(self.global_step, self.epoch, logs)
        elif mode == "train_end":
            for callback in self.callbacks:
                callback.on_train_end()
        elif mode == "dataloader_end":
            for callback in self.callbacks:
                callback.on_dataloader_end()

    def fit(
        self,
        train_dataloader,
        steps_per_epoch=None,
        epochs=1,
        grad_accumulation_steps=1,
        callbacks=[],
    ):
        if isinstance(train_dataloader.dataset, IterDataset):
            assert (
                steps_per_epoch is not None
            ), "IterDataset should specify steps_per_epoch"
        steps_per_epoch = (
            len(train_dataloader) if steps_per_epoch is None else steps_per_epoch
        )
        self.total_steps = steps_per_epoch * epochs
        self.global_step = 0
        self.train_dataloader = train_dataloader
        train_dataloader_iter = iter(self.train_dataloader)

        self.callbacks = [ProgbarLogger(epochs, steps_per_epoch, self.metrics)] + (
            callbacks if isinstance(callbacks, (list, tuple)) else [callbacks]
        )
        self.callback_fun("train_begin")

        self.bti = 0
        for epoch in range(epochs):
            if isinstance(
                self.train_dataloader.sampler,
                torch.utils.data.distributed.DistributedSampler,
            ):
                self.train_dataloader.sampler.set_epoch(epoch)
            self.epoch = epoch
            self.callback_fun("epoch_begin")
            for local_step in range(steps_per_epoch):
                self.local_step = local_step
                try:
                    batch = next(train_dataloader_iter)
                except StopIteration:
                    self.callback_fun(
                        "dataloader_end"
                    )
                    train_dataloader_iter = iter(
                        self.train_dataloader
                    )
                    self.bti = 0
                    batch = next(train_dataloader_iter)
                train_X, train_y = batch

                if isinstance(train_X, (list, tuple)):
                    if isinstance(train_X[0], (list, tuple)):
                        btz = train_X[0][0].size(0)
                    else:
                        btz = train_X[0].size(0)
                elif isinstance(train_X, torch.Tensor):
                    btz = train_X.size(0)
                else:
                    raise ValueError("Input only support [list, tuple, tensor]")
                logs = {"batch": self.local_step, "size": btz}
                self.callback_fun("batch_begin", logs)

                self.train()
                output, loss, loss_detail = self.train_step(
                    train_X, train_y, grad_accumulation_steps
                )

                retain_graph = (
                    True
                    if self.adversarial["name"] in {"gradient_penalty", "vat"}
                    else False
                )
                if self.use_amp:
                    scale_before_step = self.scaler.get_scale()
                    self.scaler.scale(loss).backward(retain_graph=retain_graph)
                else:
                    loss.backward(retain_graph=retain_graph)

                loss, loss_detail = self.adversarial_training(
                    train_X, train_y, output, loss, loss_detail, grad_accumulation_steps
                )

                if (self.global_step + 1) % grad_accumulation_steps == 0:
                    skip_scheduler = False
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.parameters(), self.max_grad_norm
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        skip_scheduler = self.scaler.get_scale() != scale_before_step
                    else:
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.parameters(), self.max_grad_norm
                            )
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    if (self.scheduler is not None) and not skip_scheduler:
                        self.scheduler.step()

                # 添加log打印
                logs.update({"loss": loss.item()})
                logs_loss_detail = {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in loss_detail.items()
                }
                logs.update(logs_loss_detail)
                if self.global_step == 0:
                    self.callbacks[0].add_metrics(
                        list(logs_loss_detail.keys()), add_position=1
                    )
                for metric in self.metrics:
                    tmp = metric_mapping(metric, output, train_y)
                    if tmp is not None:
                        logs[metric] = tmp
                self.callback_fun("batch_end", logs)

                self.bti += 1
                self.global_step += 1
            self.callback_fun("epoch_end", logs)
            callback_tmp = [
                callback_tmp
                for callback_tmp in self.callbacks
                if isinstance(callback_tmp, EarlyStopping)
            ]
            if callback_tmp and callback_tmp[0].stopped_epoch > 0:
                break
        self.callback_fun("train_end", logs)

    @torch.no_grad()
    def predict(self, input_tensor_list, return_all=None):
        self.eval()
        if self.forward.__code__.co_argcount >= 3:
            output = self.forward(*input_tensor_list)
        else:
            output = self.forward(input_tensor_list)
        if return_all is None:
            return output
        elif (
            isinstance(output, (tuple, list))
            and isinstance(return_all, int)
            and return_all < len(output)
        ):
            return output[return_all]
        else:
            raise ValueError("Return format error")

    def load_weights(self, load_path, strict=True, prefix=None):
        state_dict = torch.load(load_path, map_location="cpu")
        if prefix is None:
            self.load_state_dict(state_dict, strict=strict)
        else:
            eval_str = (
                "self.variable_mapping()"
                if prefix == ""
                else f"self.{prefix}.variable_mapping()"
            )
            mapping = {v: k for k, v in eval(eval_str).items()}
            mapping = (
                mapping
                if prefix == ""
                else {k: f"{prefix}.{v}" for k, v in mapping.items()}
            )
            state_dict_raw = {}
            for k, v in state_dict.items():
                k = mapping.get(k, k)
                state_dict_raw[k] = v
            self.load_state_dict(state_dict_raw, strict=strict)

    def save_weights(self, save_path, prefix=None):
        if prefix is None:
            torch.save(self.state_dict(), save_path)
        else:
            eval_str = (
                "self.variable_mapping()"
                if prefix == ""
                else f"self.{prefix}.variable_mapping()"
            )
            mapping = eval(eval_str)
            mapping = (
                mapping
                if prefix == ""
                else {f"{prefix}.{k}": v for k, v in mapping.items()}
            )
            state_dict_raw = {}
            for k, v in self.state_dict().items():
                k = mapping.get(k, k)
                state_dict_raw[k] = v
            torch.save(state_dict_raw, save_path)


class BaseModelDP(BaseModel, nn.DataParallel):

    def __init__(self, *args, **kwargs):
        nn.DataParallel.__init__(self, *args, **kwargs)


class BaseModelDDP(BaseModel, nn.parallel.DistributedDataParallel):

    def __init__(self, *args, master_rank=0, **kwargs):
        self.master_rank = master_rank
        nn.parallel.DistributedDataParallel.__init__(self, *args, **kwargs)


class BERT_BASE(BaseModel):
    """模型基类"""

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        hidden_act,
        dropout_rate=None,
        attention_probs_dropout_prob=None,
        embedding_size=None,
        attention_head_size=None,
        attention_key_size=None,
        initializer_range=0.02,
        sequence_length=None,
        keep_tokens=None, 
        compound_tokens=None,
        residual_attention_scores=False, 
        ignore_invalid_weights=False,
        keep_hidden_layers=None,
        hierarchical_position=None,
        **kwargs,
    ):
        super(BERT_BASE, self).__init__()
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = (
            attention_head_size or self.hidden_size // self.num_attention_heads
        )
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.attention_probs_dropout_prob = attention_probs_dropout_prob or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.initializer_range = initializer_range
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.ignore_invalid_weights = ignore_invalid_weights
        self.keep_hidden_layers = (
            set(range(num_hidden_layers))
            if keep_hidden_layers is None
            else set(keep_hidden_layers)
        )
        self.hierarchical_position = hierarchical_position

    def build(
        self,
        attention_caches=None,
        layer_norm_cond=None,
        layer_norm_cond_hidden_size=None,
        layer_norm_cond_hidden_act=None,
        additional_input_layers=None,
        **kwargs,
    ):
        self.attention_caches = attention_caches or {}
        self.output_all_encoded_layers = kwargs.get("output_all_encoded_layers", False)

    def forward(self, inputs):
        # Embedding
        outputs = self.apply_embeddings(inputs)
        # Main
        outputs = self.apply_main_layers(outputs)
        # Final
        outputs = self.apply_final_layers(outputs)
        return outputs

    def init_model_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)) and (
            module.weight.requires_grad
        ):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if (
                hasattr(module, "bias") and module.bias.requires_grad
            ):
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight.requires_grad:
                module.weight.data.fill_(1.0)
        if (
            isinstance(module, nn.Linear)
            and (module.bias is not None)
            and (module.bias.requires_grad)
        ):
            module.bias.data.zero_()

    def variable_mapping(self):
        return {}

    def load_load_variable(self):
        raise NotImplementedError

    def load_embeddings(self, embeddings):
        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]

        if self.compound_tokens is not None:
            ext_embeddings = []
            for item in self.compound_tokens:
                try:
                    ext_embeddings.append(
                        torch.mean(embeddings[item], 0)
                        * torch.ones_like(embeddings[item])
                    )
                except IndexError:
                    ext_embeddings.append(torch.mean(embeddings, 0, keepdim=True))
                    warnings.warn(
                        f"Initialize ext_embeddings from compound_tokens not in embedding index"
                    )
            embeddings = torch.cat([embeddings] + ext_embeddings, 0)

        return embeddings

    def load_pos_embeddings(self, embeddings):
        if self.hierarchical_position is not None:
            alpha = (
                0.4
                if self.hierarchical_position is True
                else self.hierarchical_position
            )
            embeddings = embeddings - alpha * embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            position_index = torch.arange(self.max_position)[:, None]

            embeddings_x = take_along_dim(
                embeddings,
                torch.div(position_index, embeddings.size(0), rounding_mode="trunc"),
                dim=0,
            )
            embeddings_y = take_along_dim(
                embeddings, position_index % embeddings.size(0), dim=0
            )
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y

        return embeddings

    def load_weights_from_pytorch_checkpoint(self, checkpoint, mapping=None):
        file_state_dict = torch.load(checkpoint, map_location="cpu")
        mapping = mapping or self.variable_mapping()
        parameters_set = set([i[0] for i in self.named_parameters()])

        for layer_name in parameters_set:
            if (layer_name in file_state_dict) and (layer_name not in mapping):
                mapping.update({layer_name: layer_name})

        state_dict_new = {}
        for new_key, old_key in mapping.items():
            if new_key not in self.state_dict():
                continue
            elif old_key in file_state_dict:
                state_dict_new[new_key] = self.load_variable(file_state_dict, old_key)
            elif (old_key not in file_state_dict) and (not self.ignore_invalid_weights):
                print(f"[WARNIMG] {old_key} not found in pretrain models")
            if new_key in parameters_set:
                parameters_set.remove(new_key)

        if not self.ignore_invalid_weights:
            for key in parameters_set:
                print(f"[WARNIMG] Parameter {key} not loaded from pretrain models")
        del file_state_dict

        self.load_state_dict(state_dict_new, strict=False)


    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_main_layers(self, inputs):
        raise NotImplementedError

    def apply_final_layers(self, inputs):
        raise NotImplementedError

    def apply_on_layer_begin(self, l_i, inputs):

        return inputs

    def apply_on_layer_end(self, l_i, inputs):

        return inputs

    def compute_attention_bias(self, inputs=None):

        return self.attention_bias

    def compute_position_bias(self, inputs=None):

        return self.position_bias

    def set_outputs(self, outputs):

        if not isinstance(outputs, list):
            outputs = [outputs]

        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]


class LM_Mask(object):

    def compute_attention_bias(self, inputs=None):
        seq_len = inputs[0].shape[1]
        attention_bias = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.long, device=inputs[0].device),
            diagonal=0,
        )
        self.attention_bias = attention_bias.unsqueeze(0).unsqueeze(1)
        return self.attention_bias


def extend_with_language_model(InputModel):

    class LanguageModel(LM_Mask, InputModel):

        def __init__(self, *args, **kwargs):
            kwargs["with_mlm"] = kwargs.get("with_mlm") or True
            super(LanguageModel, self).__init__(*args, **kwargs)

    return LanguageModel


class UniLM_Mask(object):
    def compute_attention_bias(self, inputs=None):
        segment_ids = inputs[1]
        attention_bias = torch.cumsum(segment_ids, dim=1)
        attention_bias = (attention_bias.unsqueeze(1)) <= (attention_bias.unsqueeze(2))
        self.attention_bias = attention_bias.unsqueeze(1).long()

        return self.attention_bias


def extend_with_unified_language_model(InputModel):

    class UnifiedLanguageModel(UniLM_Mask, InputModel):

        def __init__(self, *args, **kwargs):
            kwargs["with_mlm"] = kwargs.get("with_mlm") or True
            super(UnifiedLanguageModel, self).__init__(*args, **kwargs)

    return UnifiedLanguageModel


class BERT(BERT_BASE):
    def __init__(
        self,
        max_position,
        segment_vocab_size=2,
        with_pool=False,
        with_nsp=False,
        with_mlm=False,
        custom_position_ids=False,
        custom_attention_mask=False, 
        shared_segment_embeddings=False,
        layer_norm_cond=None,
        layer_add_embs=None, 
        is_dropout=False,
        token_pad_ids=0,
        **kwargs,
    ):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.custom_position_ids = custom_position_ids
        self.custom_attention_mask = custom_attention_mask
        self.shared_segment_embeddings = shared_segment_embeddings
        self.is_dropout = is_dropout
        self.token_pad_ids = token_pad_ids
        if self.with_nsp and not self.with_pool:
            self.with_pool = True
        self.layer_norm_conds = layer_norm_cond
        self.layer_add_embs = layer_add_embs
        self.conditional_size = (
            layer_norm_cond.weight.size(1) if layer_norm_cond is not None else None
        )
        self.embeddings = BertEmbeddings(
            self.vocab_size,
            self.embedding_size,
            self.hidden_size,
            self.max_position,
            self.segment_vocab_size,
            self.shared_segment_embeddings,
            self.dropout_rate,
            self.conditional_size,
            **get_kw(BertEmbeddings, kwargs),
        )
        kwargs["max_position"] = self.max_position
        layer = BertLayer(
            self.hidden_size,
            self.num_attention_heads,
            self.dropout_rate,
            self.attention_probs_dropout_prob,
            self.intermediate_size,
            self.hidden_act,
            is_dropout=self.is_dropout,
            conditional_size=self.conditional_size,
            **get_kw(BertLayer, kwargs),
        )
        self.encoderLayer = nn.ModuleList(
            [
                copy.deepcopy(layer)
                if layer_id in self.keep_hidden_layers
                else Identity()
                for layer_id in range(self.num_hidden_layers)
            ]
        )
        if self.with_pool:

            self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
            self.pooler_activation = (
                nn.Tanh() if self.with_pool is True else get_activation(self.with_pool)
            )
            if self.with_nsp:

                self.nsp = nn.Linear(self.hidden_size, 2)
        else:
            self.pooler = None
            self.pooler_activation = None
        if self.with_mlm:
            self.mlmDense = nn.Linear(self.hidden_size, self.hidden_size)
            self.transform_act_fn = get_activation(self.hidden_act)
            self.mlmLayerNorm = LayerNorm(
                self.hidden_size, eps=1e-12, conditional_size=self.conditional_size
            )
            self.mlmDecoder = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            if kwargs.get("tie_emb_prj_weight") is True:
                self.mlmDecoder.weight = self.embeddings.word_embeddings.weight
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDecoder.bias = self.mlmBias


    def apply_embeddings(self, inputs):
        token_ids = inputs[0]
        index_ = 1
        if self.segment_vocab_size > 0:
            segment_ids = inputs[index_]
            index_ += 1
        else:
            segment_ids = None

        if self.custom_position_ids:
            position_ids = inputs[index_]
            index_ += 1
        else:
            position_ids = None

        if self.custom_attention_mask:
            attention_mask = inputs[index_].long().unsqueeze(1).unsqueeze(2)
            index_ += 1
        elif (not token_ids.requires_grad) and (
            token_ids.dtype in {torch.long, torch.int}
        ):
            attention_mask = (
                (token_ids != self.token_pad_ids).long().unsqueeze(1).unsqueeze(2)
            )
            if self.token_pad_ids < 0:
                token_ids = token_ids * attention_mask[:, 0, 0, :]
        else:
            attention_mask = self.attention_mask_cache
        self.attention_mask_cache = attention_mask

        self.compute_attention_bias([token_ids, segment_ids])
        if self.attention_bias is not None:
            attention_mask = attention_mask * self.attention_bias

        try:
            attention_mask = attention_mask.to(
                dtype=next(self.parameters()).dtype
            )
        except StopIteration:
            attention_mask = attention_mask.to(dtype=torch.float32)

        if self.layer_norm_conds is None:
            conditional_emb = None
        else:
            conditional_emb = self.layer_norm_conds(inputs[index_])
            index_ += 1


        if isinstance(self.layer_add_embs, nn.Module):
            additional_embs = [self.layer_add_embs(inputs[index_])]
            index_ += 1
        elif isinstance(self.layer_add_embs, (tuple, list)):
            additional_embs = []
            for layer in self.layer_add_embs:
                assert isinstance(
                    layer, nn.Module
                ), "Layer_add_embs element should be nn.Module"
                additional_embs.append(layer(inputs[index_]))
                index_ += 1
        else:
            additional_embs = None


        hidden_states = self.embeddings(
            token_ids, segment_ids, conditional_emb, additional_embs
        )
        return [hidden_states, attention_mask, conditional_emb] + inputs[index_:]

    def apply_main_layers(self, inputs):
        hidden_states, attention_mask, conditional_emb = inputs[:3]
        if len(inputs[3:]) >= 2:
            encoder_hidden_state, encoder_attention_mask = inputs[3], inputs[4]
        else:
            encoder_hidden_state, encoder_attention_mask = None, None

        encoded_layers = [hidden_states]
        layer_inputs = [
            hidden_states,
            attention_mask,
            conditional_emb,
            encoder_hidden_state,
            encoder_attention_mask,
        ]
        for l_i, layer_module in enumerate(self.encoderLayer):
            layer_inputs = self.apply_on_layer_begin(l_i, layer_inputs)
            hidden_states = layer_module(*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(l_i, layer_inputs)

            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]

    def apply_final_layers(self, inputs):
        encoded_layers, conditional_emb = inputs
        sequence_output = encoded_layers[-1]

        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]


        if self.with_pool:
            pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        else:
            pooled_output = None

        if self.with_pool and self.with_nsp:
            nsp_scores = self.nsp(pooled_output)
        else:
            nsp_scores = None

        if self.with_mlm:
            mlm_hidden_state = self.mlmDense(sequence_output)
            mlm_hidden_state = self.transform_act_fn(mlm_hidden_state)
            mlm_hidden_state = self.mlmLayerNorm((mlm_hidden_state, conditional_emb))
            mlm_scores = self.mlmDecoder(mlm_hidden_state)
            mlm_activation = get_activation(
                "linear" if self.with_mlm is True else self.with_mlm
            )
            mlm_scores = mlm_activation(mlm_scores)
        else:
            mlm_scores = None

        outputs = [
            value
            for value in [encoded_layers, pooled_output, mlm_scores, nsp_scores]
            if value is not None
        ]
        return outputs if len(outputs) > 1 else outputs[0]

    def load_variable(self, state_dict, name, prefix="bert"):
        variable = state_dict[name]
        if name in {
            f"{prefix}.embeddings.word_embeddings.weight",
            "cls.predictions.bias",
            "cls.predictions.decoder.weight",
            "cls.predictions.decoder.bias",
        }:
            return self.load_embeddings(variable)
        elif name == f"{prefix}.embeddings.position_embeddings.weight":
            return self.load_pos_embeddings(variable)
        elif name == "cls.seq_relationship.weight":
            return variable.T
        else:
            return variable

    def variable_mapping(self, prefix="bert"):
        mapping = {
            "embeddings.word_embeddings.weight": f"{prefix}.embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight": f"{prefix}.embeddings.position_embeddings.weight",
            "embeddings.segment_embeddings.weight": f"{prefix}.embeddings.token_type_embeddings.weight",
            "embeddings.layerNorm.weight": f"{prefix}.embeddings.LayerNorm.weight",
            "embeddings.layerNorm.bias": f"{prefix}.embeddings.LayerNorm.bias",
            "pooler.weight": f"{prefix}.pooler.dense.weight",
            "pooler.bias": f"{prefix}.pooler.dense.bias",
            "nsp.weight": "cls.seq_relationship.weight",
            "nsp.bias": "cls.seq_relationship.bias",
            "mlmDense.weight": "cls.predictions.transform.dense.weight",
            "mlmDense.bias": "cls.predictions.transform.dense.bias",
            "mlmLayerNorm.weight": "cls.predictions.transform.LayerNorm.weight",
            "mlmLayerNorm.bias": "cls.predictions.transform.LayerNorm.bias",
            "mlmBias": "cls.predictions.bias",
            "mlmDecoder.weight": "cls.predictions.decoder.weight",
            "mlmDecoder.bias": "cls.predictions.decoder.bias",
        }
        for i in range(self.num_hidden_layers):
            prefix_i = f"{prefix}.encoder.layer.%d." % i
            mapping.update(
                {
                    f"encoderLayer.{i}.multiHeadAttention.q.weight": prefix_i
                    + "attention.self.query.weight",
                    f"encoderLayer.{i}.multiHeadAttention.q.bias": prefix_i
                    + "attention.self.query.bias",
                    f"encoderLayer.{i}.multiHeadAttention.k.weight": prefix_i
                    + "attention.self.key.weight",
                    f"encoderLayer.{i}.multiHeadAttention.k.bias": prefix_i
                    + "attention.self.key.bias",
                    f"encoderLayer.{i}.multiHeadAttention.v.weight": prefix_i
                    + "attention.self.value.weight",
                    f"encoderLayer.{i}.multiHeadAttention.v.bias": prefix_i
                    + "attention.self.value.bias",
                    f"encoderLayer.{i}.multiHeadAttention.o.weight": prefix_i
                    + "attention.output.dense.weight",
                    f"encoderLayer.{i}.multiHeadAttention.o.bias": prefix_i
                    + "attention.output.dense.bias",
                    f"encoderLayer.{i}.layerNorm1.weight": prefix_i
                    + "attention.output.LayerNorm.weight",
                    f"encoderLayer.{i}.layerNorm1.bias": prefix_i
                    + "attention.output.LayerNorm.bias",
                    f"encoderLayer.{i}.feedForward.intermediateDense.weight": prefix_i
                    + "intermediate.dense.weight",
                    f"encoderLayer.{i}.feedForward.intermediateDense.bias": prefix_i
                    + "intermediate.dense.bias",
                    f"encoderLayer.{i}.feedForward.outputDense.weight": prefix_i
                    + "output.dense.weight",
                    f"encoderLayer.{i}.feedForward.outputDense.bias": prefix_i
                    + "output.dense.bias",
                    f"encoderLayer.{i}.layerNorm2.weight": prefix_i
                    + "output.LayerNorm.weight",
                    f"encoderLayer.{i}.layerNorm2.bias": prefix_i
                    + "output.LayerNorm.bias",
                }
            )

        return mapping


class ALBERT(BERT):
    def __init__(self, *args, **kwargs):
        super(ALBERT, self).__init__(*args, **kwargs)
        self.encoderLayer = nn.ModuleList([self.encoderLayer[0]])

    def apply_main_layers(self, inputs):
        hidden_states, attention_mask, conditional_emb = inputs[:3]
        if len(inputs[3:]) >= 2:
            encoder_hidden_state, encoder_attention_mask = inputs[3], inputs[4]
        else:
            encoder_hidden_state, encoder_attention_mask = None, None

        encoded_layers = [hidden_states]
        layer_inputs = [
            hidden_states,
            attention_mask,
            conditional_emb,
            encoder_hidden_state,
            encoder_attention_mask,
        ]
        for l_i in range(self.num_hidden_layers):
            layer_inputs = self.apply_on_layer_begin(l_i, layer_inputs)
            hidden_states = self.encoderLayer[0](*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(l_i, layer_inputs)

            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]

    def variable_mapping(self, prefix="albert"):
        mapping = {
            "embeddings.word_embeddings.weight": f"{prefix}.embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight": f"{prefix}.embeddings.position_embeddings.weight",
            "embeddings.segment_embeddings.weight": f"{prefix}.embeddings.token_type_embeddings.weight",
            "embeddings.layerNorm.weight": f"{prefix}.embeddings.LayerNorm.weight",
            "embeddings.layerNorm.bias": f"{prefix}.embeddings.LayerNorm.bias",
            "embeddings.embedding_hidden_mapping_in.weight": f"{prefix}.encoder.embedding_hidden_mapping_in.weight",
            "embeddings.embedding_hidden_mapping_in.bias": f"{prefix}.encoder.embedding_hidden_mapping_in.bias",
            "pooler.weight": f"{prefix}.pooler.weight",
            "pooler.bias": f"{prefix}.pooler.bias",
            "nsp.weight": "sop_classifier.classifier.weight",
            "nsp.bias": "sop_classifier.classifier.bias",
            "mlmDense.weight": "predictions.dense.weight",
            "mlmDense.bias": "predictions.dense.bias",
            "mlmLayerNorm.weight": "predictions.LayerNorm.weight",
            "mlmLayerNorm.bias": "predictions.LayerNorm.bias",
            "mlmBias": "predictions.bias",
            "mlmDecoder.weight": "predictions.decoder.weight",
            "mlmDecoder.bias": "predictions.decoder.bias",
        }
        i = 0
        prefix_i = f"{prefix}.encoder.albert_layer_groups.{i}.albert_layers.{i}."
        mapping.update(
            {
                f"encoderLayer.{i}.multiHeadAttention.q.weight": prefix_i
                + "attention.query.weight",
                f"encoderLayer.{i}.multiHeadAttention.q.bias": prefix_i
                + "attention.query.bias",
                f"encoderLayer.{i}.multiHeadAttention.k.weight": prefix_i
                + "attention.key.weight",
                f"encoderLayer.{i}.multiHeadAttention.k.bias": prefix_i
                + "attention.key.bias",
                f"encoderLayer.{i}.multiHeadAttention.v.weight": prefix_i
                + "attention.value.weight",
                f"encoderLayer.{i}.multiHeadAttention.v.bias": prefix_i
                + "attention.value.bias",
                f"encoderLayer.{i}.multiHeadAttention.o.weight": prefix_i
                + "attention.dense.weight",
                f"encoderLayer.{i}.multiHeadAttention.o.bias": prefix_i
                + "attention.dense.bias",
                f"encoderLayer.{i}.layerNorm1.weight": prefix_i
                + "attention.LayerNorm.weight",
                f"encoderLayer.{i}.layerNorm1.bias": prefix_i
                + "attention.LayerNorm.bias",
                f"encoderLayer.{i}.feedForward.intermediateDense.weight": prefix_i
                + "ffn.weight",
                f"encoderLayer.{i}.feedForward.intermediateDense.bias": prefix_i
                + "ffn.bias",
                f"encoderLayer.{i}.feedForward.outputDense.weight": prefix_i
                + "ffn_output.weight",
                f"encoderLayer.{i}.feedForward.outputDense.bias": prefix_i
                + "ffn_output.bias",
                f"encoderLayer.{i}.layerNorm2.weight": prefix_i
                + "full_layer_layer_norm.weight",
                f"encoderLayer.{i}.layerNorm2.bias": prefix_i
                + "full_layer_layer_norm.bias",
            }
        )

        return mapping

    def load_variable(self, state_dict, name):

        variable = state_dict[name]
        if name in {
            "albert.embeddings.word_embeddings.weight",
            "predictions.bias",
            "predictions.decoder.weight",
            "predictions.decoder.bias",
        }:
            return self.load_embeddings(variable)
        elif name == "albert.embeddings.position_embeddings.weight":
            return self.load_pos_embeddings(variable)
        elif name == "sop_classifier.classifier.weight":
            return variable.T
        else:
            return variable


class ALBERT_Unshared(ALBERT):
    def __init__(self, *args, **kwargs):
        super(ALBERT_Unshared).__init__(*args, **kwargs)
        self.encoderLayer = nn.ModuleList(
            [copy.deepcopy(self.encoderLayer[0]) for _ in range(self.num_hidden_layers)]
        )

    def apply_main_layers(self, inputs):

        hidden_states, attention_mask, conditional_emb = inputs
        if len(inputs[3:]) >= 2:
            encoder_hidden_state, encoder_attention_mask = inputs[3], inputs[4]
        else:
            encoder_hidden_state, encoder_attention_mask = None, None

        encoded_layers = [hidden_states]  # 添加embedding的输出
        layer_inputs = [
            hidden_states,
            attention_mask,
            conditional_emb,
            encoder_hidden_state,
            encoder_attention_mask,
        ]
        for i in range(self.num_hidden_layers):
            layer_inputs = self.apply_on_layer_begin(i, layer_inputs)
            hidden_states = self.encoderLayer[i](*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(i, layer_inputs)

            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        return [encoded_layers, conditional_emb]


class NEZHA(BERT):
    def __init__(self, *args, **kwargs):

        kwargs.update(
            {
                "p_bias": "typical_relative",
                "max_relative_position": kwargs.get("max_relative_position", 64),
            }
        )
        super(NEZHA, self).__init__(*args, **kwargs)


class RoFormer(BERT):
    def __init__(self, *args, **kwargs):
        kwargs.update({"p_bias": "rotary"})
        super(RoFormer, self).__init__(*args, **kwargs)

    def load_variable(self, state_dict, name, prefix="roformer"):
        return super().load_variable(state_dict, name, prefix)

    def variable_mapping(self, prefix="roformer"):
        mapping = super().variable_mapping(prefix)
        del mapping["embeddings.position_embeddings.weight"]
        return mapping


class RoFormerV2(RoFormer):
    @delete_arguments("with_pool", "with_nsp")
    def __init__(self, *args, **kwargs):
        kwargs.update(
            {"p_bias": "rotary", "weight": False, "bias": False, "norm_mode": "rmsnorm"}
        )
        super(RoFormerV2, self).__init__(*args, **kwargs)
        if self.with_mlm:
            del self.mlmLayerNorm
            del self.mlmBias
            del self.mlmDense
            self.mlmDecoder.register_parameter("bias", None)

    def variable_mapping(self, prefix="roformer"):
        mapping = super().variable_mapping(prefix)
        mapping_new = {}
        for k, v in mapping.items():
            if (not re.search("bias|layernorm", k.lower())) and (
                not re.search("bias|layernorm", v.lower())
            ):
                mapping_new[k] = v
        return mapping_new

    def apply_final_layers(self, inputs):
        encoded_layers, conditional_emb = inputs
        sequence_output = encoded_layers[-1]
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if self.with_mlm:
            mlm_scores = self.mlmDecoder(sequence_output)
        else:
            mlm_scores = None

        outputs = [value for value in [encoded_layers, mlm_scores] if value is not None]
        return outputs if len(outputs) > 1 else outputs[0]


class GAU_alpha(RoFormerV2):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            {
                "p_bias": "rotary",
                "weight": False,
                "bias": False,
                "norm_mode": "rmsnorm",
                "normalization": "softmax_plus",
            }
        )
        super().__init__(*args, **kwargs)

        layer = self.GAU_Layer(**kwargs)
        self.encoderLayer = nn.ModuleList(
            [
                copy.deepcopy(layer)
                if layer_id in self.keep_hidden_layers
                else Identity()
                for layer_id in range(self.num_hidden_layers)
            ]
        )

    def load_variable(self, state_dict, name, prefix=""):
        variable = state_dict[name]
        return (
            self.load_embeddings(variable)
            if name in {"embeddings.word_embeddings.weight", "mlmDecoder.weight"}
            else variable
        )

    def variable_mapping(self, prefix=""):
        return {k: k for k, _ in self.named_parameters()}

    class GAU_Layer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.gau = GatedAttentionUnit(**kwargs)
            self.dropout1 = nn.Dropout(kwargs.get("dropout_rate"))
            self.layerNorm1 = LayerNorm(**kwargs)

        def forward(
            self,
            hidden_states,
            attention_mask,
            conditional_emb=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
        ):
            gau_hidden_states = self.gau(hidden_states, attention_mask)
            hidden_states = hidden_states + self.dropout1(gau_hidden_states)
            hidden_states = self.layerNorm1((hidden_states, conditional_emb))
            return hidden_states


class ELECTRA(BERT):
    @insert_arguments(with_discriminator=False)
    @delete_arguments("with_pool", "with_mlm", "with_nsp")
    def __init__(self, max_position, **kwargs):
        super(ELECTRA, self).__init__(max_position, **kwargs)
        if self.with_discriminator:
            self.dense = nn.Linear(self.hidden_size, self.hidden_size)
            self.dense_act = get_activation(self.hidden_act)
            self.dense_prediction = nn.Linear(self.hidden_size, 1)
            self.dense_prediction_act = (
                get_activation("sigmoid")
                if self.with_discriminator is True
                else get_activation(self.with_discriminator)
            )

    def apply_final_layers(self, inputs):
        hidden_states = super().apply_final_layers(inputs)
        if self.with_discriminator:
            logit = self.dense_act(self.dense(hidden_states))
            return [
                hidden_states,
                self.dense_prediction_act(self.dense_prediction(logit)),
            ]
        else:
            return hidden_states

    def load_variable(self, state_dict, name):
        return super().load_variable(state_dict, name, prefix="electra")

    def variable_mapping(self):
        mapping = super(ELECTRA, self).variable_mapping(prefix="electra")
        mapping.update(
            {
                "dense.weight": "discriminator_predictions.dense.weight",
                "dense.bias": "discriminator_predictions.dense.bias",
                "dense_prediction.weight": "discriminator_predictions.dense_prediction.weight",
                "dense_prediction.bias": "discriminator_predictions.dense_prediction.bias",
            }
        )
        for del_key in [
            "pooler.weight",
            "pooler.bias",
            "nsp.weight",
            "nsp.bias",
            "mlmDense.weight",
            "mlmDense.bias",
            "mlmLayerNorm.weight",
            "mlmLayerNorm.bias",
            "mlmBias",
            "mlmDecoder.weight",
            "mlmDecoder.bias",
        ]:
            del mapping[del_key]

        return mapping


class Encoder(BERT):
    def __init__(self, *args, **kwargs):
        kwargs["vocab_size"] = kwargs.get("src_vocab_size", kwargs["vocab_size"])
        super().__init__(*args, **kwargs)
        self.encoder_attention_mask = None

    def forward(self, inputs):
        # Embedding
        outputs = self.apply_embeddings(inputs)
        encoder_attention_mask = [outputs[1]]
        # Main
        outputs = self.apply_main_layers(outputs)
        # Final
        outputs = self.apply_final_layers(outputs)
        return (
            [outputs] if isinstance(outputs, torch.Tensor) else outputs
        ) + encoder_attention_mask


class Decoder(LM_Mask, BERT):
    @delete_arguments("with_pool", "with_mlm", "with_nsp")
    def __init__(self, *args, with_lm=True, tie_emb_prj_weight=True, **kwargs):
        kwargs["vocab_size"] = kwargs.get("tgt_vocab_size", kwargs["vocab_size"])
        kwargs["is_decoder"] = True
        super().__init__(*args, **kwargs)
        self.decoderLayer = self.encoderLayer
        del self.encoderLayer
        self.with_lm = with_lm

        if self.with_lm:
            self.final_dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            if tie_emb_prj_weight:
                self.final_dense.weight = self.embeddings.word_embeddings.weight
                self.x_logit_scale = self.hidden_size**-0.5
            else:
                self.x_logit_scale = 1.0

    def apply_main_layers(self, inputs):
        (
            hidden_states,
            attention_mask,
            conditional_emb,
            encoder_hidden_state,
            encoder_attention_mask,
        ) = inputs[:5]
        decoded_layers = [hidden_states]
        layer_inputs = [
            hidden_states,
            attention_mask,
            conditional_emb,
            encoder_hidden_state,
            encoder_attention_mask,
        ]
        for i, layer_module in enumerate(self.decoderLayer):
            layer_inputs = self.apply_on_layer_begin(i, layer_inputs)
            hidden_states = layer_module(*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(i, layer_inputs)

            if self.output_all_encoded_layers:
                decoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            decoded_layers.append(hidden_states)
        return [decoded_layers, conditional_emb]

    def apply_final_layers(self, inputs):
        outputs = []
        hidden_states = super().apply_final_layers(
            inputs
        )
        outputs.append(hidden_states)
        if self.with_lm:
            logits = (
                self.final_dense(hidden_states) * self.x_logit_scale
            )
            activation = get_activation(
                "linear" if self.with_lm is True else self.with_lm
            )
            logits = activation(logits)
            outputs.append(logits)
        return outputs

    def variable_mapping(self, prefix="bert"):
        raw_mapping = super().variable_mapping(prefix)
        mapping = {}
        for k, v in raw_mapping.items():
            mapping[k.replace("encoderLayer", "decoderLayer")] = v
        return mapping


class Transformer(BERT_BASE):
    """encoder-decoder结构"""

    @delete_arguments("with_pool", "with_mlm", "with_nsp")
    def __init__(self, *args, tie_emb_src_tgt_weight=False, **kwargs):
        super(Transformer, self).__init__(*args, **kwargs)

        # encoder
        self.encoder = Encoder(*args, **kwargs)
        self.encoder.build(**kwargs)

        # decoder
        self.decoder = Decoder(*args, **kwargs)
        self.decoder.build(**kwargs)

        if tie_emb_src_tgt_weight:
            assert (
                self.encoder.vocab_size == self.decoder.vocab_size
            ), "To share word embedding, the vocab size of src/tgt shall be the same."
            self.encoder.embeddings.word_embeddings.weight = (
                self.decoder.embeddings.word_embeddings.weight
            )

    def forward(self, inputs):
        encoder_input, decoder_input = inputs[:2]

        # encoder
        # encoder_emb = self.encoder.apply_embeddings(encoder_input)
        # encode_outputs = self.encoder.apply_main_layers(encoder_emb)
        # encoder_hidden_state = self.encoder.apply_final_layers(encode_outputs)
        # encoder_attention_mask = encoder_emb[1]
        encoder_hidden_state, encoder_attention_mask = self.encoder(encoder_input)

        # decoder
        # decoder_emb = self.decoder.apply_embeddings(decoder_input)
        # decoder_outputs = self.decoder.apply_main_layers([*decoder_emb, encoder_hidden_state, encoder_attention_mask])
        # decoder_outputs = self.decoder.apply_final_layers(decoder_outputs) # [hidden_states, logits]
        decoder_outputs = self.decoder(
            decoder_input + [encoder_hidden_state, encoder_attention_mask]
        )
        return [
            encoder_hidden_state
        ] + decoder_outputs


class BART(Transformer):
    """encoder-decoder结构"""

    def __init__(self, *args, tie_emb_src_tgt_weight=True, **kwargs):
        super(BART, self).__init__(
            *args, tie_emb_src_tgt_weight=tie_emb_src_tgt_weight, **kwargs
        )
        self.tie_emb_src_tgt_weight = tie_emb_src_tgt_weight

    def load_variable(self, state_dict, name, prefix=""):
        variable = state_dict[name]
        if name in {
            "shared.weight",
            "encoder.embed_tokens.weight",
            "decoder.embed_tokens.weight",
        }:
            return self.load_embeddings(variable)
        elif name in {
            "encoder.embed_positions.weight",
            "decoder.embed_positions.weight",
        }:
            return self.load_pos_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=""):
        mapping = {
            "encoder.embeddings.word_embeddings.weight": "shared.weight"
            if self.tie_emb_src_tgt_weight
            else "encoder.embed_tokens.weight",
            "encoder.embeddings.position_embeddings.weight": "encoder.embed_positions.weight",
            "encoder.embeddings.layerNorm.weight": "encoder.layernorm_embedding.weight",
            "encoder.embeddings.layerNorm.bias": "encoder.layernorm_embedding.bias",
            "decoder.embeddings.word_embeddings.weight": "shared.weight"
            if self.tie_emb_src_tgt_weight
            else "decoder.embed_tokens.weight",
            "decoder.embeddings.position_embeddings.weight": "decoder.embed_positions.weight",
            "decoder.embeddings.layerNorm.weight": "decoder.layernorm_embedding.weight",
            "decoder.embeddings.layerNorm.bias": "decoder.layernorm_embedding.bias",
        }
        for i in range(self.num_hidden_layers):
            mapping.update(
                {
                    f"encoder.encoderLayer.{i}.multiHeadAttention.q.weight": f"encoder.layers.{i}.self_attn.q_proj.weight",
                    f"encoder.encoderLayer.{i}.multiHeadAttention.q.bias": f"encoder.layers.{i}.self_attn.q_proj.bias",
                    f"encoder.encoderLayer.{i}.multiHeadAttention.k.weight": f"encoder.layers.{i}.self_attn.k_proj.weight",
                    f"encoder.encoderLayer.{i}.multiHeadAttention.k.bias": f"encoder.layers.{i}.self_attn.k_proj.bias",
                    f"encoder.encoderLayer.{i}.multiHeadAttention.v.weight": f"encoder.layers.{i}.self_attn.v_proj.weight",
                    f"encoder.encoderLayer.{i}.multiHeadAttention.v.bias": f"encoder.layers.{i}.self_attn.v_proj.bias",
                    f"encoder.encoderLayer.{i}.multiHeadAttention.o.weight": f"encoder.layers.{i}.self_attn.out_proj.weight",
                    f"encoder.encoderLayer.{i}.multiHeadAttention.o.bias": f"encoder.layers.{i}.self_attn.out_proj.bias",
                    f"encoder.encoderLayer.{i}.layerNorm1.weight": f"encoder.layers.{i}.self_attn_layer_norm.weight",
                    f"encoder.encoderLayer.{i}.layerNorm1.bias": f"encoder.layers.{i}.self_attn_layer_norm.bias",
                    f"encoder.encoderLayer.{i}.feedForward.intermediateDense.weight": f"encoder.layers.{i}.fc1.weight",
                    f"encoder.encoderLayer.{i}.feedForward.intermediateDense.bias": f"encoder.layers.{i}.fc1.bias",
                    f"encoder.encoderLayer.{i}.feedForward.outputDense.weight": f"encoder.layers.{i}.fc2.weight",
                    f"encoder.encoderLayer.{i}.feedForward.outputDense.bias": f"encoder.layers.{i}.fc2.bias",
                    f"encoder.encoderLayer.{i}.layerNorm2.weight": f"encoder.layers.{i}.final_layer_norm.weight",
                    f"encoder.encoderLayer.{i}.layerNorm2.bias": f"encoder.layers.{i}.final_layer_norm.bias",
                    f"decoder.decoderLayer.{i}.multiHeadAttention.q.weight": f"decoder.layers.{i}.self_attn.q_proj.weight",
                    f"decoder.decoderLayer.{i}.multiHeadAttention.q.bias": f"decoder.layers.{i}.self_attn.q_proj.bias",
                    f"decoder.decoderLayer.{i}.multiHeadAttention.k.weight": f"decoder.layers.{i}.self_attn.k_proj.weight",
                    f"decoder.decoderLayer.{i}.multiHeadAttention.k.bias": f"decoder.layers.{i}.self_attn.k_proj.bias",
                    f"decoder.decoderLayer.{i}.multiHeadAttention.v.weight": f"decoder.layers.{i}.self_attn.v_proj.weight",
                    f"decoder.decoderLayer.{i}.multiHeadAttention.v.bias": f"decoder.layers.{i}.self_attn.v_proj.bias",
                    f"decoder.decoderLayer.{i}.multiHeadAttention.o.weight": f"decoder.layers.{i}.self_attn.out_proj.weight",
                    f"decoder.decoderLayer.{i}.multiHeadAttention.o.bias": f"decoder.layers.{i}.self_attn.out_proj.bias",
                    f"decoder.decoderLayer.{i}.layerNorm1.weight": f"decoder.layers.{i}.self_attn_layer_norm.weight",
                    f"decoder.decoderLayer.{i}.layerNorm1.bias": f"decoder.layers.{i}.self_attn_layer_norm.bias",
                    f"decoder.decoderLayer.{i}.crossAttention.q.weight": f"decoder.layers.{i}.encoder_attn.q_proj.weight",
                    f"decoder.decoderLayer.{i}.crossAttention.q.bias": f"decoder.layers.{i}.encoder_attn.q_proj.bias",
                    f"decoder.decoderLayer.{i}.crossAttention.k.weight": f"decoder.layers.{i}.encoder_attn.k_proj.weight",
                    f"decoder.decoderLayer.{i}.crossAttention.k.bias": f"decoder.layers.{i}.encoder_attn.k_proj.bias",
                    f"decoder.decoderLayer.{i}.crossAttention.v.weight": f"decoder.layers.{i}.encoder_attn.v_proj.weight",
                    f"decoder.decoderLayer.{i}.crossAttention.v.bias": f"decoder.layers.{i}.encoder_attn.v_proj.bias",
                    f"decoder.decoderLayer.{i}.crossAttention.o.weight": f"decoder.layers.{i}.encoder_attn.out_proj.weight",
                    f"decoder.decoderLayer.{i}.crossAttention.o.bias": f"decoder.layers.{i}.encoder_attn.out_proj.bias",
                    f"decoder.decoderLayer.{i}.layerNorm3.weight": f"decoder.layers.{i}.encoder_attn_layer_norm.weight",
                    f"decoder.decoderLayer.{i}.layerNorm3.bias": f"decoder.layers.{i}.encoder_attn_layer_norm.bias",
                    f"decoder.decoderLayer.{i}.feedForward.intermediateDense.weight": f"decoder.layers.{i}.fc1.weight",
                    f"decoder.decoderLayer.{i}.feedForward.intermediateDense.bias": f"decoder.layers.{i}.fc1.bias",
                    f"decoder.decoderLayer.{i}.feedForward.outputDense.weight": f"decoder.layers.{i}.fc2.weight",
                    f"decoder.decoderLayer.{i}.feedForward.outputDense.bias": f"decoder.layers.{i}.fc2.bias",
                    f"decoder.decoderLayer.{i}.layerNorm2.weight": f"decoder.layers.{i}.final_layer_norm.weight",
                    f"decoder.decoderLayer.{i}.layerNorm2.bias": f"decoder.layers.{i}.final_layer_norm.bias",
                }
            )

        return mapping


class T5_Encoder(Encoder):
    @insert_arguments(version="t5.1.0")
    def __init__(self, *args, **kwargs):
        kwargs.update(
            {
                "p_bias": "t5_relative",
                "relative_attention_num_buckets": kwargs.get(
                    "relative_attention_num_buckets"
                ),
                "version": self.version,
                "bias": False,
                "norm_mode": "rmsnorm",
            }
        )
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm

        layer = T5Layer(
            self.hidden_size,
            self.num_attention_heads,
            self.dropout_rate,
            self.attention_probs_dropout_prob,
            self.intermediate_size,
            self.hidden_act,
            is_dropout=self.is_dropout,
            conditional_size=self.conditional_size,
            **get_kw(BertLayer, kwargs),
        )
        self.encoderLayer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(self.num_hidden_layers)]
        )

        for i in range(1, self.num_hidden_layers):
            self.encoderLayer[
                i
            ].multiHeadAttention.relative_positions_encoding.weight = self.encoderLayer[
                0
            ].multiHeadAttention.relative_positions_encoding.weight
        self.final_layer_norm = LayerNorm(
            self.hidden_size,
            eps=1e-12,
            conditional_size=self.conditional_size,
            bias=False,
            mode="rmsnorm",
        )
        self.dropout = nn.Dropout(self.dropout_rate)

    def apply_final_layers(self, inputs):
        hidden_states = super().apply_final_layers(inputs)
        return self.dropout(self.final_layer_norm([hidden_states]))

    def load_variable(self, state_dict, name, prefix=""):
        variable = state_dict[name]
        if name in {"encoder.embed_tokens.weight", "shared.weight"}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=""):
        mapping = {
            f"{prefix}embeddings.word_embeddings.weight": "encoder.embed_tokens.weight",
            f"{prefix}encoderLayer.0.multiHeadAttention.relative_positions_encoding.weight": "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            f"{prefix}final_layer_norm.weight": "encoder.final_layer_norm.weight",
        }
        for i in range(self.num_hidden_layers):
            mapping.update(
                {
                    f"{prefix}encoderLayer.{i}.multiHeadAttention.q.weight": f"encoder.block.{i}.layer.0.SelfAttention.q.weight",
                    f"{prefix}encoderLayer.{i}.multiHeadAttention.k.weight": f"encoder.block.{i}.layer.0.SelfAttention.k.weight",
                    f"{prefix}encoderLayer.{i}.multiHeadAttention.v.weight": f"encoder.block.{i}.layer.0.SelfAttention.v.weight",
                    f"{prefix}encoderLayer.{i}.multiHeadAttention.o.weight": f"encoder.block.{i}.layer.0.SelfAttention.o.weight",
                    f"{prefix}encoderLayer.{i}.layerNorm1.weight": f"encoder.block.{i}.layer.0.layer_norm.weight",
                    f"{prefix}encoderLayer.{i}.feedForward.outputDense.weight": f"encoder.block.{i}.layer.1.DenseReluDense.wo.weight",
                    f"{prefix}encoderLayer.{i}.layerNorm2.weight": f"encoder.block.{i}.layer.1.layer_norm.weight",
                }
            )

            if self.version.endswith("t5.1.0"):
                mapping.update(
                    {
                        f"{prefix}encoderLayer.{i}.feedForward.intermediateDense.weight": f"encoder.block.{i}.layer.1.DenseReluDense.wi.weight"
                    }
                )
            elif self.version.endswith("t5.1.1"):
                mapping.update(
                    {
                        f"{prefix}encoderLayer.{i}.feedForward.intermediateDense.weight": f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight",
                        f"{prefix}encoderLayer.{i}.feedForward.intermediateDense1.weight": f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight",
                    }
                )
        return mapping


class T5_Decoder(Decoder):
    @insert_arguments(version="t5.1.0")
    def __init__(self, *args, **kwargs):
        kwargs.update(
            {
                "p_bias": "t5_relative",
                "relative_attention_num_buckets": kwargs.get(
                    "relative_attention_num_buckets"
                ),
                "version": self.version,
                "bias": False,
                "norm_mode": "rmsnorm",
            }
        )
        super().__init__(*args, **kwargs)
        del self.embeddings.layerNorm

        layer = T5Layer(
            self.hidden_size,
            self.num_attention_heads,
            self.dropout_rate,
            self.attention_probs_dropout_prob,
            self.intermediate_size,
            self.hidden_act,
            is_dropout=self.is_dropout,
            conditional_size=self.conditional_size,
            is_decoder=True,
            **get_kw(BertLayer, kwargs),
        )
        self.decoderLayer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(self.num_hidden_layers)]
        )

        for i in range(1, self.num_hidden_layers):
            self.decoderLayer[
                i
            ].multiHeadAttention.relative_positions_encoding.weight = self.decoderLayer[
                0
            ].multiHeadAttention.relative_positions_encoding.weight
        self.final_layer_norm = LayerNorm(
            self.hidden_size,
            eps=1e-12,
            conditional_size=self.conditional_size,
            bias=False,
            mode="rmsnorm",
        )
        self.dropout = nn.Dropout(self.dropout_rate)

    def apply_final_layers(self, inputs):
        inputs[0][1] = self.dropout(
            self.final_layer_norm([inputs[0][1]])
        )
        return super().apply_final_layers(inputs)

    def load_variable(self, state_dict, name, prefix=""):
        variable = state_dict[name]
        if name in {f"decoder.embed_tokens.weight", "lm_head.weight", "shared.weight"}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=""):
        mapping = {
            f"{prefix}embeddings.word_embeddings.weight": "decoder.embed_tokens.weight",
            f"{prefix}decoderLayer.0.multiHeadAttention.relative_positions_encoding.weight": "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            f"{prefix}final_layer_norm.weight": "decoder.final_layer_norm.weight",
            f"{prefix}final_dense.weight": "lm_head.weight",
        }

        for i in range(self.num_hidden_layers):
            mapping.update(
                {
                    f"{prefix}decoderLayer.{i}.multiHeadAttention.q.weight": f"decoder.block.{i}.layer.0.SelfAttention.q.weight",
                    f"{prefix}decoderLayer.{i}.multiHeadAttention.k.weight": f"decoder.block.{i}.layer.0.SelfAttention.k.weight",
                    f"{prefix}decoderLayer.{i}.multiHeadAttention.v.weight": f"decoder.block.{i}.layer.0.SelfAttention.v.weight",
                    f"{prefix}decoderLayer.{i}.multiHeadAttention.o.weight": f"decoder.block.{i}.layer.0.SelfAttention.o.weight",
                    f"{prefix}decoderLayer.{i}.layerNorm1.weight": f"decoder.block.{i}.layer.0.layer_norm.weight",
                    f"{prefix}decoderLayer.{i}.crossAttention.q.weight": f"decoder.block.{i}.layer.1.EncDecAttention.q.weight",
                    f"{prefix}decoderLayer.{i}.crossAttention.k.weight": f"decoder.block.{i}.layer.1.EncDecAttention.k.weight",
                    f"{prefix}decoderLayer.{i}.crossAttention.v.weight": f"decoder.block.{i}.layer.1.EncDecAttention.v.weight",
                    f"{prefix}decoderLayer.{i}.crossAttention.o.weight": f"decoder.block.{i}.layer.1.EncDecAttention.o.weight",
                    f"{prefix}decoderLayer.{i}.layerNorm3.weight": f"decoder.block.{i}.layer.1.layer_norm.weight",
                    f"{prefix}decoderLayer.{i}.feedForward.outputDense.weight": f"decoder.block.{i}.layer.2.DenseReluDense.wo.weight",
                    f"{prefix}decoderLayer.{i}.layerNorm2.weight": f"decoder.block.{i}.layer.2.layer_norm.weight",
                }
            )

            if self.version.endswith("t5.1.0"):
                mapping.update(
                    {
                        f"{prefix}decoderLayer.{i}.feedForward.intermediateDense.weight": f"decoder.block.{i}.layer.2.DenseReluDense.wi.weight"
                    }
                )
            elif self.version.endswith("t5.1.1"):
                mapping.update(
                    {
                        f"{prefix}decoderLayer.{i}.feedForward.intermediateDense.weight": f"decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight",
                        f"{prefix}decoderLayer.{i}.feedForward.intermediateDense1.weight": f"decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight",
                    }
                )
        return mapping


class T5(Transformer):
    @delete_arguments("with_pool", "with_mlm", "with_nsp")
    def __init__(self, *args, tie_emb_src_tgt_weight=True, **kwargs):
        super(T5, self).__init__(*args, **kwargs)
        self.tie_emb_src_tgt_weight = tie_emb_src_tgt_weight

        # encoder
        self.encoder = T5_Encoder(*args, **kwargs)
        self.encoder.build(**kwargs)

        # decoder
        self.decoder = T5_Decoder(*args, **kwargs)
        self.decoder.build(**kwargs)

    def load_variable(self, state_dict, name, prefix=""):
        variable = state_dict[name]
        if name in {
            "shared.weight",
            "encoder.embed_tokens.weight",
            "decoder.embed_tokens.weight",
            "lm_head.weight",
        }:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self, prefix=""):
        mapping = self.encoder.variable_mapping(prefix="encoder.")
        mapping.update(self.decoder.variable_mapping(prefix="decoder."))
        if self.tie_emb_src_tgt_weight:
            mapping.update(
                {
                    "encoder.embeddings.word_embeddings.weight": "shared.weight",
                    "decoder.embeddings.word_embeddings.weight": "shared.weight",
                }
            )
        return mapping


class GPT(LM_Mask, BERT):
    @insert_arguments(final_activation="softmax")
    @delete_arguments("with_pool", "with_mlm", "with_nsp")
    def __init__(self, max_position, **kwargs):
        super(GPT, self).__init__(max_position, **kwargs)
        del self.embeddings.layerNorm
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.dense.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(self.final_activation)

    def apply_final_layers(self, inputs):
        hidden_state = super().apply_final_layers(inputs)
        logit = self.dense(hidden_state)
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT, self).load_variable(state_dict, name, prefix="gpt")

    def variable_mapping(self):
        mapping = super(GPT, self).variable_mapping(prefix="gpt")
        return mapping


class GPT2(LM_Mask, BERT):

    @insert_arguments(final_activation="softmax")
    @delete_arguments("with_pool", "with_mlm", "with_nsp")
    def __init__(self, max_position, **kwargs):
        super(GPT2, self).__init__(max_position, **kwargs)
        del self.embeddings.layerNorm
        layer = self.Gpt2Layer(
            self.hidden_size,
            self.num_attention_heads,
            self.dropout_rate,
            self.attention_probs_dropout_prob,
            self.intermediate_size,
            self.hidden_act,
            is_dropout=self.is_dropout,
            conditional_size=self.conditional_size,
        )
        self.encoderLayer = nn.ModuleList(
            [
                copy.deepcopy(layer)
                if layer_id in self.keep_hidden_layers
                else Identity()
                for layer_id in range(self.num_hidden_layers)
            ]
        )
        self.LayerNormFinal = LayerNorm(
            self.hidden_size, eps=1e-12, conditional_size=self.conditional_size
        )
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.dense.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(self.final_activation)

    def apply_final_layers(self, inputs):
        hidden_state = super().apply_final_layers(inputs)
        logit = self.dense(self.LayerNormFinal([hidden_state]))
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT2, self).load_variable(state_dict, name, prefix="gpt2")

    def variable_mapping(self):
        mapping = super(GPT2, self).variable_mapping(prefix="gpt2")
        mapping.update(
            {
                "LayerNormFinal.weight": "gpt2.LayerNormFinal.weight",
                "LayerNormFinal.bias": "gpt2.LayerNormFinal.bias",
            }
        )
        return mapping

    class Gpt2Layer(BertLayer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(
            self,
            hidden_states,
            attention_mask,
            conditional_emb=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
        ):
            x = self.layerNorm1((hidden_states, conditional_emb))
            self_attn_output = self.multiHeadAttention(x, attention_mask)
            hidden_states = hidden_states + self.dropout1(self_attn_output)
            x = self.layerNorm2((hidden_states, conditional_emb))
            ffn_output = self.feedForward(x)
            hidden_states = hidden_states + self.dropout2(ffn_output)
            return hidden_states


class GPT2_ML(LM_Mask, BERT):
    @insert_arguments(final_activation="softmax")
    @delete_arguments("with_pool", "with_mlm", "with_nsp")
    def __init__(self, max_position, **kwargs):
        super().__init__(max_position, **kwargs)
        layer = self.Gpt2MlLayer(
            self.hidden_size,
            self.num_attention_heads,
            self.dropout_rate,
            self.attention_probs_dropout_prob,
            self.intermediate_size,
            self.hidden_act,
            is_dropout=self.is_dropout,
            conditional_size=self.conditional_size,
        )
        self.encoderLayer = nn.ModuleList(
            [
                copy.deepcopy(layer)
                if layer_id in self.keep_hidden_layers
                else Identity()
                for layer_id in range(self.num_hidden_layers)
            ]
        )
        self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.dense.weight = self.embeddings.word_embeddings.weight
        self.final_activation = get_activation(self.final_activation)

    def apply_final_layers(self, inputs):
        hidden_state = super().apply_final_layers(inputs)
        logit = self.dense(hidden_state)
        return self.final_activation(logit)

    def load_variable(self, state_dict, name):
        return super(GPT2_ML, self).load_variable(state_dict, name, prefix="gpt2_ml")

    def variable_mapping(self):
        mapping = super(GPT2_ML, self).variable_mapping(prefix="gpt2_ml")
        return mapping

    class Gpt2MlLayer(BertLayer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(
            self,
            hidden_states,
            attention_mask,
            conditional_emb=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
        ):
            self_attn_output = self.multiHeadAttention(hidden_states, attention_mask)
            hidden_states = hidden_states + self.dropout1(self_attn_output)
            x = self.layerNorm1((hidden_states, conditional_emb))
            ffn_output = self.feedForward(x)
            hidden_states = hidden_states + self.dropout2(ffn_output)
            hidden_states = self.layerNorm2((hidden_states, conditional_emb))
            return hidden_states


class Transformer_XL(BERT):
    @delete_arguments("with_pool", "with_nsp", "with_mlm")
    @insert_arguments(with_lm=False)
    def __init__(self, *args, mem_len=0, same_length=False, clamp_len=-1, **kwargs):
        # p_bias来控制embedding阶段无pos_embedding
        kwargs.update({"p_bias": "other_relative"})
        super().__init__(*args, **kwargs)
        self.mem_len, self.same_length, self.clamp_len = mem_len, same_length, clamp_len
        self.attn_type = kwargs.get("attn_type", 0)

        # embedding
        if kwargs.get("adaptive_embedding"):
            cutoffs, div_val, sample_softmax = (
                kwargs.get("cutoffs", []),
                kwargs.get("div_val", 1),
                kwargs.get("sample_softmax", False),
            )
            self.embeddings = AdaptiveEmbedding(
                self.vocab_size,
                self.embedding_size,
                self.hidden_size,
                cutoffs,
                div_val,
                sample_softmax,
                **get_kw(AdaptiveEmbedding, kwargs),
            )
        else:
            self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.pos_embeddings = XlnetPositionsEncoding(self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        if not kwargs.get("untie_r"):
            self.r_w_bias = nn.Parameter(
                torch.FloatTensor(self.num_attention_heads, self.attention_head_size)
            )
            self.r_r_bias = nn.Parameter(
                torch.FloatTensor(self.num_attention_heads, self.attention_head_size)
            )
            if self.segment_vocab_size > 0:
                self.r_s_bias = nn.Parameter(
                    torch.FloatTensor(
                        self.num_attention_heads, self.attention_head_size
                    )
                )
        else:
            self.r_w_bias, self.r_r_bias = None, None
            self.r_s_bias = None

        # transformer block
        layer = XlnetLayer(
            self.hidden_size,
            self.num_attention_heads,
            self.dropout_rate,
            self.attention_probs_dropout_prob,
            self.intermediate_size,
            self.hidden_act,
            is_dropout=self.is_dropout,
            conditional_size=self.conditional_size,
            r_w_bias=self.r_w_bias,
            r_r_bias=self.r_r_bias,
            r_s_bias=None,
            **get_kw(BertLayer, kwargs),
        )
        self.encoderLayer = nn.ModuleList(
            [
                copy.deepcopy(layer)
                if layer_id in self.keep_hidden_layers
                else Identity()
                for layer_id in range(self.num_hidden_layers)
            ]
        )

        # 映射
        if self.with_lm:
            self.dense = nn.Linear(self.hidden_size, self.vocab_size, bias=True)

    def init_mems(self, bsz):
        if isinstance(self.mem_len, (int, float)) and (self.mem_len > 0):
            mems = []
            param = next(self.parameters())
            for _ in range(self.num_hidden_layers + 1):
                empty = torch.zeros(
                    bsz,
                    self.mem_len,
                    self.hidden_size,
                    dtype=param.dtype,
                    device=param.device,
                )
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mlen, qlen):
        # does not deal with None
        if self.mems is None:
            return None
        # mems is not None
        assert len(hids) == len(self.mems), "len(hids) != len(mems)"
        # There are `mlen + qlen` steps that can be cached into mems
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([self.mems[i], hids[i]], dim=1)
                new_mems.append(cat[:, beg_idx:end_idx].detach())
        self.mems = new_mems

    def relative_positional_encoding(self, qlen, klen, device):
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=device, dtype=torch.long)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.dropout(self.pos_embeddings(pos_seq))
        return pos_emb

    def create_mask(self, word_emb, qlen, klen, mlen):

        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            mask_shift_len = qlen - mask_len if mask_len > 0 else qlen
            attention_mask = (
                1
                - (
                    torch.triu(all_ones, 1 + mlen)
                    + torch.tril(all_ones, -mask_shift_len)
                ).byte()
            )
        else:
            attention_mask = torch.tril(
                word_emb.new_ones(qlen, klen), diagonal=mlen
            ).byte()
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    def apply_embeddings(self, inputs):

        self.mems = self.init_mems(inputs[0].size(0))


        word_emb = self.dropout(self.embeddings(inputs[0]))
        index_ = 1
        btz, qlen = inputs[0].shape[:2]
        mlen = self.mems[0].size(1) if self.mems is not None else 0
        klen = mlen + qlen

        pos_emb = self.relative_positional_encoding(qlen, klen, word_emb.device)

        if self.segment_vocab_size > 0:
            segment_ids = inputs[index_]
            if mlen > 0:
                mem_pad = torch.zeros(
                    [btz, mlen], dtype=torch.long, device=word_emb.device
                )
                cat_ids = torch.cat([mem_pad, segment_ids], dim=1)
            else:
                cat_ids = segment_ids
            segment_ids = (segment_ids[:, :, None] != cat_ids[:, None]).long()
            index_ += 1
        else:
            segment_ids = None

        if self.attn_type in {"uni", 0}:
            attention_mask = self.create_mask(word_emb, qlen, klen, mlen)
        elif self.attn_type == "bi":
            attention_mask = (
                (inputs[0] != self.token_pad_ids).long().unsqueeze(1).unsqueeze(2)
            )
        non_tgt_mask = torch.eye(qlen).to(attention_mask)[None, None, :, :]
        non_tgt_mask = ((1 - attention_mask - non_tgt_mask) <= 0).long()

        return [word_emb, segment_ids, pos_emb, non_tgt_mask, None]

    def apply_main_layers(self, inputs):
        hidden_states, segment_ids, pos_emb, attention_mask, conditional_emb = inputs[
            :5
        ]
        encoded_layers = [hidden_states]

        layer_inputs = [
            hidden_states,
            segment_ids,
            pos_emb,
            attention_mask,
            None,
            conditional_emb,
        ]
        for i, layer_module in enumerate(self.encoderLayer):
            mems_i = None if self.mems is None else self.mems[i]
            layer_inputs[-2] = mems_i
            layer_inputs = self.apply_on_layer_begin(i, layer_inputs)
            hidden_states = layer_module(*layer_inputs)
            layer_inputs[0] = hidden_states
            layer_inputs = self.apply_on_layer_end(i, layer_inputs)
            encoded_layers.append(hidden_states)

        hidden_states = self.dropout(hidden_states)
        qlen = inputs[0].size(1)
        mlen = self.mems[0].size(0) if self.mems is not None else 0
        self._update_mems(encoded_layers, mlen, qlen)

        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[:1] + [hidden_states]
        return [encoded_layers, conditional_emb]

    def load_variable(self, state_dict, name, prefix=""):
        if (self.keep_tokens is not None) or (self.compound_tokens is not None):
            raise ValueError(
                "Custom keep_tokens and compound_tokens is not yet supported in Transformer_XL"
            )
        return state_dict[name]

    def variable_mapping(self, prefix=""):
        return {k: k for k, v in self.named_parameters()}


class XLNET(Transformer_XL):

    def __init__(self, *args, bi_data=False, **kwargs):
        self.attn_type = kwargs.get("attn_type", "bi")
        self.bi_data = bi_data
        kwargs["rel_shift_opt"] = "xlnet"
        super().__init__(*args, **kwargs)

    def relative_positional_encoding(self, qlen, klen, device):
        if self.attn_type == "bi":
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            beg, end = klen, -1
        else:
            raise ValueError(f"Unknown `attn_type` {self.attn_type}.")

        pos_seq = torch.arange(beg, end, -1.0, device=device, dtype=torch.long)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        fwd_pos_emb = self.pos_embeddings(pos_seq)

        if self.bi_data:
            pos_seq = torch.arange(-beg, -end, -1.0, device=device, dtype=torch.long)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            bwd_pos_emb = self.pos_embeddings(pos_seq)
            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=0)
        else:
            pos_emb = fwd_pos_emb

        pos_emb = self.dropout(pos_emb)
        return pos_emb

    def apply_final_layers(self, inputs):
        hidden_state = super().apply_final_layers(inputs)
        if self.with_lm:
            return [hidden_state, self.dense(hidden_state)]
        else:
            return hidden_state

    def load_variable(self, state_dict, name, prefix="transformer"):
        variable = state_dict[name]
        if name in {
            f"{prefix}.word_embedding.weight",
            "lm_loss.weight",
            "lm_loss.bias",
        }:
            return self.load_embeddings(variable)
        elif re.search("rel_attn\.(q|k|v|r)$", name):
            return variable.reshape(variable.shape[0], -1).T
        # elif re.search('rel_attn\.(o|seg_embed)$', name):
        elif re.search("rel_attn\.(o)$", name):
            return variable.reshape(variable.shape[0], -1)
        else:
            return variable

    def variable_mapping(self, prefix="transformer"):
        mapping = {
            "embeddings.weight": f"{prefix}.word_embedding.weight",
            "dense.weight": "lm_loss.weight",
            "dense.bias": "lm_loss.bias",
        }
        for i in range(self.num_hidden_layers):
            prefix_i = f"{prefix}.layer.%d." % i
            mapping.update(
                {
                    f"encoderLayer.{i}.multiHeadAttention.q.weight": prefix_i
                    + "rel_attn.q",
                    f"encoderLayer.{i}.multiHeadAttention.k.weight": prefix_i
                    + "rel_attn.k",
                    f"encoderLayer.{i}.multiHeadAttention.v.weight": prefix_i
                    + "rel_attn.v",
                    f"encoderLayer.{i}.multiHeadAttention.o.weight": prefix_i
                    + "rel_attn.o",
                    f"encoderLayer.{i}.multiHeadAttention.r.weight": prefix_i
                    + "rel_attn.r",
                    f"encoderLayer.{i}.multiHeadAttention.r_r_bias": prefix_i
                    + "rel_attn.r_r_bias",
                    f"encoderLayer.{i}.multiHeadAttention.r_s_bias": prefix_i
                    + "rel_attn.r_s_bias",
                    f"encoderLayer.{i}.multiHeadAttention.r_w_bias": prefix_i
                    + "rel_attn.r_w_bias",
                    # f'encoderLayer.{i}.multiHeadAttention.seg_embed.weight': prefix_i + 'rel_attn.seg_embed',
                    f"encoderLayer.{i}.multiHeadAttention.seg_embed": prefix_i
                    + "rel_attn.seg_embed",
                    f"encoderLayer.{i}.layerNorm1.weight": prefix_i
                    + "rel_attn.layer_norm.weight",
                    f"encoderLayer.{i}.layerNorm1.bias": prefix_i
                    + "rel_attn.layer_norm.bias",
                    f"encoderLayer.{i}.feedForward.intermediateDense.weight": prefix_i
                    + "ff.layer_1.weight",
                    f"encoderLayer.{i}.feedForward.intermediateDense.bias": prefix_i
                    + "ff.layer_1.bias",
                    f"encoderLayer.{i}.feedForward.outputDense.weight": prefix_i
                    + "ff.layer_2.weight",
                    f"encoderLayer.{i}.feedForward.outputDense.bias": prefix_i
                    + "ff.layer_2.bias",
                    f"encoderLayer.{i}.layerNorm2.weight": prefix_i
                    + "ff.layer_norm.weight",
                    f"encoderLayer.{i}.layerNorm2.bias": prefix_i
                    + "ff.layer_norm.bias",
                }
            )

        return mapping


def build_transformer_model(
    config_path=None,
    checkpoint_path=None,
    model="bert",
    application="encoder",
    **kwargs,
):

    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if "max_position" not in configs:
        configs["max_position"] = configs.get("max_position_embeddings", 512)
    if "dropout_rate" not in configs:
        configs["dropout_rate"] = configs.get("hidden_dropout_prob")
    if "segment_vocab_size" not in configs:
        configs["segment_vocab_size"] = configs.get("type_vocab_size", 2)

    models = {
        "bert": BERT,
        "roberta": BERT,
        "albert": ALBERT,
        "albert_unshared": ALBERT_Unshared,
        "nezha": NEZHA,
        "roformer": RoFormer,
        "roformer_v2": RoFormerV2,
        "gau_alpha": GAU_alpha,
        "electra": ELECTRA,
        "encoder": Encoder,
        "decoder": Decoder,
        "transformer": Transformer,
        "bart": BART,
        "gpt": GPT,
        "gpt2": GPT2,
        "gpt2_ml": GPT2_ML,
        "t5": T5,
        "t5_encoder": T5_Encoder,
        "t5_decoder": T5_Decoder,
        "t5.1.0": T5,
        "t5.1.0_encoder": T5_Encoder,
        "t5.1.0_decoder": T5_Decoder,
        "t5.1.1": T5,
        "t5.1.1_encoder": T5_Encoder,
        "t5.1.1_decoder": T5_Decoder,
        "mt5.1.1": T5,
        "mt5.1.1_encoder": T5_Encoder,
        "mt5.1.1_decoder": T5_Decoder,
        "transformer_xl": Transformer_XL,
        "xlnet": XLNET,
    }

    if isinstance(model, str):
        MODEL = models[model.lower()]
        if model.endswith("t5.1.1"):
            configs["version"] = model
    elif isinstance(model, type) and issubclass(
        model, BERT_BASE
    ):
        MODEL = model
    else:
        raise ValueError('"model" args type should be string or nn.Module')

    application = application.lower()
    if application in ["lm", "unilm"] and model in [
        "electra",
        "t5",
    ]:
        raise ValueError(
            f'"{model}" model can not be used as "{application}" application.\n'
        )

    if application == "lm":
        MODEL = extend_with_language_model(MODEL)
    elif application == "unilm":
        MODEL = extend_with_unified_language_model(MODEL)

    transformer = MODEL(**configs)
    transformer.build(**configs)
    transformer.apply(transformer.init_model_weights)

    if checkpoint_path is not None:
        transformer.load_weights_from_pytorch_checkpoint(checkpoint_path)
    transformer.configs = configs
    return transformer
