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
#pragma once

#include <cublasLt.h>

#include <cuda.h>
namespace nvinfer1::plugin {
#ifdef __ILUVATAR__
static const int kMaxThreadNbPerBlock = 1024;
static const int kMaxBlockNbPerSM = 8;
static const int kWarpSize = 64;
static const dim3 kMaxBlockDimension = {4096, 4096, 64};
static const dim3 kMaxGridDimension = {4294967295, 65536, 65536};
static const int kNbThreadsPerBlockGainBestPerformance = 1024;
static const int kMaxSharedMemSizePerBlock = (128 * 1024 * 4);
static const int kNbSmemLane = 64;
static const int kNbBytesPerSmemLane = 4;
#else
static const int kMaxThreadNbPerBlock = 1024;
static const int kMaxBlockNbPerSM = 8;
static const int kWarpSize = 32;
static const dim3 kMaxBlockDimension = {1024, 1024, 64};
static const dim3 kMaxGridDimension = {2147483647, 65535, 65535};
static const int kNbThreadsPerBlockGainBestPerformance = 256;
static const int kMaxSharedMemSizePerBlock = 48 * 1024 * 4;
static const int kNbSmemLane = 32;
static const int kNbBytesPerSmemLane = 4;
#endif

static const int kNbCe = 4;
static const int kNbCuPerCe = 4;
static const int kNbSppPerCu = 4;

static const float kLog2e = 1.442695040888963387;

#define DivUp(x, y) (((x) + (y)-1) / (y))

__device__ __forceinline__ float floatExp(float x) { return __builtin_exp2f(kLog2e * x); }

__device__ __forceinline__ float floatLog(float x) { return __logf(x); }

__forceinline__ int nearest_num(int x, int value) {
    if (x % value == 0) {
        return x;
    } else {
        int padding = value - x % value;
        return x + padding;
    }
}
}  // namespace nvinfer1::plugin
