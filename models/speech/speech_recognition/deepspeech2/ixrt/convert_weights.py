import os
import onnx
import argparse
import numpy as np
from onnx import TensorProto, numpy_helper, helper

def parse_args():
    parser = argparse.ArgumentParser(description="Convert the  weight of lstm in model.")
    parser.add_argument("--input_onnx", type=str, default="/home/yanlong.hao/DeepSpeech2/ixrt-modelzoo/data/checkpoints/deepspeech2/deepspeech2_part.onnx")
    parser.add_argument("--output_onnx", type=str, default="/home/yanlong.hao/DeepSpeech2/ixrt-modelzoo/data/checkpoints/deepspeech2/deepspeech2.onnx")

    args = parser.parse_args()
    return args


def convert_weights(args):
    onnx_model = onnx.load(args.input_onnx)
    graph = onnx_model.graph
    node  = graph.node
    initializer = graph.initializer

    for i in range(len(node)):
        if node[i].op_type == "LSTM":
            count = 0
            for t in node[i].input:
                if not t:
                    count += 1
            print("count: ", count)
            for _ in range(count):
                node[i].input.remove("")

            hidden_size = 0
            for j in range(len(node[i].attribute)):
                if node[i].attribute[j].name == "hidden_size":
                    hidden_size = node[i].attribute[j].i

            w_name = node[i].input[1]
            r_name = node[i].input[2]
            b_name = node[i].input[3]

            w_data = None
            r_data = None
            b_data = None

            for data in initializer:
                if data.name ==  node[i].input[1]:
                    dims = list(data.dims).copy()
                    dims_A = dims.copy()
                    w_origin_data = np.frombuffer(data.raw_data, dtype=np.float32)
                    W_save = np.transpose(w_origin_data.reshape(dims), [0, 2, 1])
                    w1 = W_save[0, :, :hidden_size].reshape(-1)
                    w2 = W_save[0, :, hidden_size : hidden_size * 2].reshape(-1)
                    w3 = W_save[0, :, hidden_size * 2 : hidden_size * 3].reshape(-1)
                    w4 = W_save[0, :, hidden_size * 3 : hidden_size * 4].reshape(-1)

                    w_r1 = W_save[1, :, :hidden_size].reshape(-1)
                    w_r2 = W_save[1, :, hidden_size : hidden_size * 2].reshape(-1)
                    w_r3 = W_save[1, :, hidden_size * 2 : hidden_size * 3].reshape(-1)
                    w_r4 = W_save[1, :, hidden_size * 3 : hidden_size * 4].reshape(-1)

                    w_data = np.concatenate([w1, w2, w3, w4, w_r1, w_r2, w_r3, w_r4])
                    print("w_data shape: ", w_data.shape)

                if data.name ==  node[i].input[2]:
                    dims = list(data.dims).copy()
                    dims_B = dims.copy()
                    r_origin_data = np.frombuffer(data.raw_data, dtype=np.float32)
                    R_save = np.transpose(r_origin_data.reshape(dims), [0, 2, 1])
                    r1 = R_save[0, :, :hidden_size].reshape(-1)
                    r2 = R_save[0, :, hidden_size : hidden_size * 2].reshape(-1)
                    r3 = R_save[0, :, hidden_size * 2 : hidden_size * 3].reshape(-1)
                    r4 = R_save[0, :, hidden_size * 3 : hidden_size * 4].reshape(-1)

                    r_r1 = R_save[1, :, :hidden_size].reshape(-1)
                    r_r2 = R_save[1, :, hidden_size : hidden_size * 2].reshape(-1)
                    r_r3 = R_save[1, :, hidden_size * 2 : hidden_size * 3].reshape(-1)
                    r_r4 = R_save[1, :, hidden_size * 3 : hidden_size * 4].reshape(-1)

                    r_data = np.concatenate([r1, r2, r3, r4, r_r1, r_r2, r_r3, r_r4])
                    print("r_data shape: ", r_data.shape)

                if data.name ==  node[i].input[3]:
                    dims = data.dims
                    b_origin_data = np.frombuffer(data.raw_data, dtype=np.float32)
                    B_save = b_origin_data.reshape(dims)
                    bias_ih = B_save[0, : hidden_size * 4]
                    bias_hh = B_save[0, hidden_size * 4 : hidden_size * 8]
                    bias_f = bias_ih + bias_hh  # bias add merge
                    bias_r_ih = B_save[1, : hidden_size * 4]
                    bias_r_hh = B_save[1, hidden_size * 4 : hidden_size * 8]
                    bias_r = bias_r_ih + bias_r_hh  # bias add merge
                    b_data = np.concatenate([bias_f, bias_r])
                    print("b_data shape: ", b_data.shape)

            for save_data in initializer:
                if w_name == save_data.name:
                    save_data.raw_data=w_data.astype(np.float32).tobytes()

                elif r_name == save_data.name:
                    save_data.raw_data=r_data.astype(np.float32).tobytes()

                elif b_name == save_data.name:
                    save_data.raw_data=b_data.astype(np.float32).tobytes()
                    save_data.dims[1] = int(save_data.dims[1] / 2)


    for data in initializer:

        if data.name == "p2o.helper.constant.2":
            raw_data = np.frombuffer(data.raw_data, dtype=np.int64)
            tmp_data = raw_data.copy()
            tmp_data[0] = 1
            # tmp_data[0] = 16
            tmp_data[1] = -1
            tmp_data[2] = 1248
            data.raw_data = tmp_data.tobytes()

    lstm_reshape_name = "p2o.helper.constant.4"
    # batch size: 1
    lstm_reshape_params = helper.make_tensor(lstm_reshape_name, onnx.TensorProto.INT64, [3], [-1,1,2048])
    # batch size: 16
    # lstm_reshape_params = helper.make_tensor(lstm_reshape_name, onnx.TensorProto.INT64, [3], [-1,16,2048])
    initializer.append(lstm_reshape_params)

    first_reshape_node = True
    for i in range(len(node)):
        if node[i].op_type == "Reshape":
            if first_reshape_node:
                first_reshape_node = False
                continue
            else:
                node[i].input[1] = lstm_reshape_name

    onnx.save(onnx_model, args.output_onnx)



if __name__ == "__main__":
    args = parse_args()
    convert_weights(args)
    print("Save Down!")
