# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

THIS_DIR=$(dirname $(readlink -f $0))

torchrun --nproc_per_node=8 --master_port 12333 \
  $THIS_DIR/run_qat.py \
  --model_name_or_path bert-large-uncased \
  --dataset_name squad_qat \
  --do_train \
  --do_eval \
  --max_seq_length 384 \
  --per_device_train_batch_size 4 \
  --doc_stride 128 \
  --learning_rate 8.7e-5 \
  --num_train_epochs 2 \
  --output_dir quant_bert_large \
  --overwrite_output_dir \
  --fp16 \
  --seed 1234 \
  --logging_steps 1000 \
  --module_type 2 \
  --enable_quant true
