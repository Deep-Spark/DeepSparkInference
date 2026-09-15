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

import numpy as np
import tvm
from sklearn.preprocessing import normalize
from tqdm import tqdm

from eval.verification import evaluate, load_bin, resolve_bin_path


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
                        help="datasets path.")

    parser.add_argument("--input_name",
                        type=str,
                        required=True,
                        help="input name of the model.")

    parser.add_argument("--imgsz",
                        type=int,
                        default=112)

    parser.add_argument("--warmup",
                        type=int,
                        default=3,
                        help="number of warmup before test.")

    parser.add_argument("--nfolds",
                        type=int,
                        default=10,
                        help="LFW K-fold count.")

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


def preprocess(images):
    images = images.astype(np.float32, copy=False)
    return ((images / 255.0) - 0.5) / 0.5


def extract_embeddings(module, input_name, device, data, batch_size):
    images = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
    embeddings = None
    ba = 0
    n_img = images.shape[0]
    pbar = tqdm(total=n_img, leave=False)
    while ba < n_img:
        bb = min(ba + batch_size, n_img)
        count = bb - ba
        batch = images[ba:bb]
        if count != batch_size:
            batch = np.resize(batch, (batch_size, *batch.shape[1:]))
        batch = preprocess(batch)
        module.set_input(input_name, tvm.nd.array(batch, device))
        module.run()
        out = module.get_output(0).asnumpy()
        if embeddings is None:
            embeddings = np.zeros((n_img, out.shape[1]), dtype=np.float32)
        embeddings[ba:bb, :] = out[:count]
        ba = bb
        pbar.update(count)
    pbar.close()
    return embeddings


def main():
    args = parse_args()
    batch_size = args.batchsize

    target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")
    device = tvm.device(target.kind.name, 0)

    lib = tvm.runtime.load_module(args.engine)
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](device))

    if args.perf_only:
        ftimer = module.module.time_evaluator("run", device, number=100, repeat=1)
        prof_res = np.array(ftimer().results) * 1000
        fps = batch_size * 1000 / np.mean(prof_res)
        print(f"\n* Mean inference time: {np.mean(prof_res):.3f} ms, Mean fps: {fps:.3f}")
        return

    for _ in range(args.warmup):
        module.run()

    bin_path = resolve_bin_path(args.datasets)
    print("loading..", bin_path)
    data_list, issame_list = load_bin(bin_path, [args.imgsz, args.imgsz])

    embeddings_list = []
    for data in data_list:
        embeddings_list.append(
            extract_embeddings(module, args.input_name, device, data, batch_size)
        )

    embeddings = normalize(embeddings_list[0] + embeddings_list[1])
    _, _, accuracy, val, val_std, far = evaluate(
        embeddings, issame_list, nrof_folds=args.nfolds
    )
    acc = np.mean(accuracy)
    std = np.std(accuracy)
    print("Accuracy-Flip: %1.5f+-%1.5f" % (acc, std))
    print("Validation rate: %2.5f+-%2.5f @ FAR=%2.5f" % (val, val_std, far))
    print("LFW ACC: {:.5f}".format(acc))


if __name__ == "__main__":
    main()
