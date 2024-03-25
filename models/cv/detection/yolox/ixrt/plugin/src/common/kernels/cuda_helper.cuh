/* Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
* All Rights Reserved.
*
*    Licensed under the Apache License, Version 2.0 (the "License"); you may
*    not use this file except in compliance with the License. You may obtain
*    a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0
*
*    Unless required by applicable law or agreed to in writing, software
*    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
*    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
*    License for the specific language governing permissions and limitations
*    under the License.
*/
#define DEVICE_FUNC __device__ __forceinline__
namespace nvinfer1::plugin {
constexpr float LOG2E = 1.442695040888963387;

DEVICE_FUNC float _exp(float x) { return __builtin_exp2f(LOG2E * x); }
DEVICE_FUNC float dequantize(int8_t x, float scale) { return scale * static_cast<float>(x); }
DEVICE_FUNC float sigmoid(float x) { return 1/((1.f + _exp(0.f - x))); }
}  // namespace nvinfer1::plugin