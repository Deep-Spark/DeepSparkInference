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

import torch
import torch.nn as nn
from quantization import disable_quant, ptq_mode, qat_mode
from training_ops.torch_transformer_layers import BertEmbeddingLayer


def get_hf_bert_enc_layer_params(layer):
    init_ws = []
    init_bs = []

    init_ws.append(layer.multiHeadAttention.q.weight.detach().clone())
    init_bs.append(layer.multiHeadAttention.q.bias.detach().clone())
    init_ws.append(layer.multiHeadAttention.k.weight.detach().clone())
    init_bs.append(layer.multiHeadAttention.k.bias.detach().clone())
    init_ws.append(layer.multiHeadAttention.v.weight.detach().clone())
    init_bs.append(layer.multiHeadAttention.v.bias.detach().clone())
    init_ws.append(layer.multiHeadAttention.o.weight.detach().clone())
    init_bs.append(layer.multiHeadAttention.o.bias.detach().clone())
    init_ws.append(layer.layerNorm1.weight.detach().clone())
    init_bs.append(layer.layerNorm1.bias.detach().clone())

    init_ws.append(layer.feedForward.intermediateDense.weight.detach().clone())
    init_bs.append(layer.feedForward.intermediateDense.bias.detach().clone())
    init_ws.append(layer.feedForward.outputDense.weight.detach().clone())
    init_bs.append(layer.feedForward.outputDense.bias.detach().clone())
    init_ws.append(layer.layerNorm2.weight.detach().clone())
    init_bs.append(layer.layerNorm2.bias.detach().clone())

    return init_ws, init_bs


def get_hf_bert_emb_layer_params(layer):
    init_ws = []

    init_ws.append(layer.word_embeddings.weight.detach().clone())
    init_ws.append(layer.position_embeddings.weight.detach().clone())
    max_seq_len, hidden_size = layer.position_embeddings.weight.shape
    init_ws.append(torch.zeros(2, hidden_size))
    init_ws.append(layer.layerNorm.weight.detach().clone())
    init_ws.append(layer.layerNorm.bias.detach().clone())

    return init_ws


def gen_bert_emb_config(training_args, config):
    bert_emb_config = BertEmbeddingLayer.get_config(
        vocab_size=config.vocab_size,
        embedding_dim=config.hidden_size,
        max_batch_tokens=4608,
        max_seq_len=config.max_position_embeddings,
        padding_idx=config.pad_token_id,
        dropout=config.hidden_dropout_prob,
        fp16=training_args.fp16,
        local_rank=training_args.local_rank,
    )
    bert_emb_config.type_vocab_size = config.type_vocab_size
    bert_emb_config.layer_norm_eps = config.layer_norm_eps
    return bert_emb_config


def inject_ls_layer(model, training_args, model_args, config):
    if model_args.module_type == 2:
        from training_ops.torch_transformer_layers import TransformerEncoderLayer
    else:
        print("use defualt hf model ...")
        return model
    if model_args.quant_mode == "ptq":
        quant_type = ptq_mode
    elif model_args.quant_mode == "qat":
        quant_type = qat_mode
    else:
        raise NotImplementedError(f"quant_type {model_args.quant_mode} NotImplemented!")

    if model_args.module_type == 2:
        bert_emb_config = gen_bert_emb_config(training_args, config)
        init_ws = get_hf_bert_emb_layer_params(model.bert.embeddings)
        model.bert.embeddings = BertEmbeddingLayer(bert_emb_config, init_ws)
        if model_args.enable_quant:
            model.bert.embeddings.apply(quant_type)
        else:
            model.bert.embeddings.apply(disable_quant)

    class LSHFTransformerEncoderLayer(TransformerEncoderLayer):
        def __init__(self, *args, **kwargs):
            super(LSHFTransformerEncoderLayer, self).__init__(*args, **kwargs)

        def forward(self, hidden_states, encoder_padding_mask, *args, **kwargs):
            # ls_encoder_padding_mask = encoder_padding_mask / -10000.0
            ls_encoder_padding_mask = (1.0 - encoder_padding_mask) * -3e38
            ls_encoder_padding_mask = ls_encoder_padding_mask.squeeze()
            output = super().forward(hidden_states, ls_encoder_padding_mask)
            return output

    def gen_bert_enc_config(training_args, config):
        bert_enc_config = TransformerEncoderLayer.get_config(
            max_batch_tokens=4608,
            max_seq_len=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            nhead=config.num_attention_heads,
            attn_prob_dropout_ratio=config.attention_probs_dropout_prob,
            activation_dropout_ratio=config.hidden_dropout_prob,
            hidden_dropout_ratio=config.hidden_dropout_prob,
            pre_layer_norm=False,
            fp16=training_args.fp16,
            local_rank=training_args.local_rank,
            activation_fn="gelu",
        )
        return bert_enc_config

    for i in range(config.num_hidden_layers):
        bert_enc_config = gen_bert_enc_config(training_args, config)
        init_ws, init_bs = get_hf_bert_enc_layer_params(model.bert.encoderLayer[i])
        model.bert.encoderLayer[i] = LSHFTransformerEncoderLayer(
            bert_enc_config, init_ws, init_bs
        ).cuda()
        if model_args.module_type == 2:
            if model_args.enable_quant:
                model.bert.encoderLayer[i].apply(quant_type)
            else:
                model.bert.encoderLayer[i].apply(disable_quant)
