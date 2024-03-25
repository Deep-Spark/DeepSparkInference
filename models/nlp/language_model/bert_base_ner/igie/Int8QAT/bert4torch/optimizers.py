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

from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def extend_with_exponential_moving_average(model, decay=0.999):
    class ExponentialMovingAverage:

        def __init__(self, model, decay):
            self.model = model
            self.decay = decay
            self.ema_weights = {}
            self.model_weights = {}

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.ema_weights[name] = param.data.clone()

        def step(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.ema_weights
                    new_average = (
                        1.0 - self.decay
                    ) * param.data + self.decay * self.ema_weights[name]
                    self.ema_weights[name] = new_average.clone()

        def apply_ema_weights(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.ema_weights
                    self.model_weights[name] = param.data
                    param.data = self.ema_weights[name]

        def restore_raw_weights(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.model_weights
                    param.data = self.model_weights[name]
            self.model_weights = {}

    return ExponentialMovingAverage(model, decay)
