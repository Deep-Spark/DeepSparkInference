import json
import os
import numpy as np
import argparse
import time

import tensorrt
from tensorrt import Dims
from common import create_engine_context, get_io_bindings, setup_io_bindings

import cuda.cudart as cudart


def engine_init(engine):
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    engine, context = create_engine_context(engine, logger)

    return engine, context


def tensorrt_infer(engine, context, features):
    input_names=["src_tokens"]
    output_names=["output"]
    input_idx = engine.get_binding_index(input_names[0])
    input_shape = features.shape
    context.set_binding_shape(input_idx, Dims(input_shape))

    inputs, outputs, allocations = setup_io_bindings(engine, context)
    pred_output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
    (err,) = cudart.cudaMemcpy(
        inputs[0]["allocation"],
        features,
        inputs[0]["nbytes"],
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    )
    assert err == cudart.cudaError_t.cudaSuccess
    context.execute_v2(allocations)
    (err,) = cudart.cudaMemcpy(
        pred_output,
        outputs[0]["allocation"],
        outputs[0]["nbytes"],
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    assert err == cudart.cudaError_t.cudaSuccess
    return pred_output


def generate_batch(features):
    all_inputs = []
    tmp = []
    for data in features:
        if len(tmp) == args.max_batch_size:
            batch_max_len = max([len(i) for i in tmp])
            new_tmp = []
            for i in tmp:
                i = i[:args.max_seq_len]
                i = [pad_id]*(batch_max_len-len(i)) + i
                new_tmp.append(i)
            all_inputs.append(np.array(new_tmp).astype(np.int32))
            tmp = []
        tmp.append(data)
    if len(tmp):
        batch_max_len = max([len(i) for i in tmp])
        new_tmp = []
        for i in tmp:
            i = i[:args.max_seq_len]
            i = [pad_id]*(batch_max_len-len(i)) + i
            new_tmp.append(i)
        all_inputs.append(np.array(new_tmp).astype(np.int32))
    return all_inputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="build ixrt graph and convert weights", usage=""
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        required=True,
        help="max batch size for inference",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=102,
        help="max sequence length for inference",
    )
    parser.add_argument(
        "--data_dir",
        type=str
    )
    parser.add_argument(
        "--model_dir",
        type=str
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert args.max_seq_len <= 102
    pad_id = 1
    feature_file = os.path.join(args.data_dir,'features.json')

    with open(feature_file,'r') as f:
        features = json.loads(f.read())

    all_inputs = generate_batch(features)
    print(f"max_batch_size: {args.max_batch_size}, max_seq_len: {args.max_seq_len}")

    print("1. build engine")
    engine_path = os.path.join(args.model_dir,'transformer.engine')
    print(f"load engine from {engine_path}")

    engine, context = engine_init(engine_path)

    print("2. warmup")
    for i in range(5):
        batch = np.random.randint(10, 20, [args.max_batch_size, args.max_seq_len]).astype(
            np.int32
        )
        tensorrt_infer(engine, context, batch)

    print("3. inference")
    start_time = time.time()
    num_sentences = 0
    for i,batch in enumerate(all_inputs):
        num_sentences += batch.shape[0]
        res = tensorrt_infer(engine, context, batch)

    end_time = time.time()
    QPS = num_sentences/(end_time-start_time)
    print(f"Translated {num_sentences} sentences, {QPS} sentences/s")
    target_qps = float(os.environ['Accuracy'])

    # Release the resouce of context and engine
    del context
    del engine

    print("QPS: = ", QPS, "target QPS: ", target_qps)
    if QPS >= target_qps:
        print("pass!")
        exit()
    else:
        print("failed!")
        exit(10)
