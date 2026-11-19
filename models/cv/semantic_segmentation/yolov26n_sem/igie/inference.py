# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

import argparse
import os
from pathlib import Path

import numpy as np
import tvm
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG

from validator import IGIE_Validator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--engine",
                        type=str,
                        required=True,
                        help="igie engine path.")

    parser.add_argument("--batchsize",
                        type=int,
                        required=True,
                        help="inference batch size.")

    parser.add_argument("--datasets",
                        type=str,
                        required=True,
                        help="cityscapes dataset root or cityscapes.yaml path.")

    parser.add_argument("--input_name",
                        type=str,
                        required=True,
                        help="input name of the model.")

    parser.add_argument("--imgsz",
                        type=int,
                        nargs=2,
                        default=[1024, 2048],
                        help="input image size as H W.")

    parser.add_argument("--warmup",
                        type=int,
                        default=3,
                        help="number of warmup before test.")

    parser.add_argument("--acc_target",
                        type=float,
                        default=None,
                        help="Model inference Accuracy target.")

    parser.add_argument("--fps_target",
                        type=float,
                        default=None,
                        help="Model inference FPS target.")

    parser.add_argument("--perf_only",
                        type=bool,
                        default=False,
                        help="Run performance test only")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    batch_size = args.batchsize

    # create iluvatar target & device
    target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")
    device = tvm.device(target.kind.name, 0)

    # load engine
    lib = tvm.runtime.load_module(args.engine)

    # create runtime from engine
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](device))

    # just run perf test
    if args.perf_only:
        ftimer = module.module.time_evaluator("run", device, number=100, repeat=1)
        prof_res = np.array(ftimer().results) * 1000
        fps = batch_size * 1000 / np.mean(prof_res)
        print(f"\n* Mean inference time: {np.mean(prof_res):.3f} ms, Mean fps: {fps:.3f}")
    else:
        datasets = args.datasets
        if datasets.endswith(".yaml") or datasets.endswith(".yml"):
            data_yaml = datasets
        else:
            data_yaml = os.path.join(datasets, "cityscapes.yaml")

        overrides = {
            "mode": "val",
            "task": "semantic",
            # Val CFG requires an int; exact HxW is patched onto LetterBox later.
            "imgsz": max(args.imgsz),
            "rect": False,
        }
        cfg_args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        cfg_args.batch = batch_size
        cfg_args.plots = False
        cfg_args.save_json = False
        cfg_args.workers = 4

        data = check_det_dataset(data_yaml)
        cfg_args.data = data

        validator = IGIE_Validator(args=cfg_args, save_dir=Path("."))
        validator.stride = 32
        validator(module, device, imgsz=args.imgsz)

        miou = validator.metrics.miou if hasattr(validator.metrics, "miou") else validator.metrics.results_dict.get("metrics/mIoU", None)
        pixacc = getattr(validator.metrics, "pixel_accuracy", None)
        print(f"\n* mIoU: {float(miou):.4f}" if miou is not None else "")
        if pixacc is not None:
            print(f"* Pixel Accuracy: {float(pixacc):.4f}")


if __name__ == "__main__":
    main()
