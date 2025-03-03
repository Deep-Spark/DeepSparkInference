#!/usr/bin/env python3
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
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import time
import json
import ctypes
import argparse
import collections
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import helpers.tokenization as tokenization
import helpers.data_processing as dp
from tqdm import tqdm
import math

from load_ixrt_plugin import load_ixrt_plugin
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--engine',
            help='Path to BERT TensorRT engine')
    parser.add_argument("-b", "--batch-size", default=1, help="Batch size for inference.", type=int)
    parser.add_argument('-p', '--passage', nargs='*',
            help='Text for paragraph/passage for BERT QA',
            default='')
    parser.add_argument('-pf', '--passage-file',
            help='File containing input passage',
            default='')
    parser.add_argument('-q', '--question', nargs='*',
            help='Text for query/question for BERT QA',
            default='')
    parser.add_argument('-qf', '--question-file',
            help='File containing input question',
            default='')
    parser.add_argument('-sq', '--squad-json',
            help='SQuAD json file',
            default='')
    parser.add_argument('-o', '--output-prediction-file',
            help='Output prediction file for SQuAD evaluation',
            default='./predictions.json')
    parser.add_argument('-v', '--vocab-file',
            help='Path to file containing entire understandable vocab')
    parser.add_argument('-s', '--sequence-length',
            help='The sequence length to use. Defaults to 128',
            default=128, type=int)
    parser.add_argument('--max-query-length',
            help='The maximum length of a query in number of tokens. Queries longer than this will be truncated',
            default=64, type=int)
    parser.add_argument('--max-answer-length',
            help='The maximum length of an answer that can be generated',
            default=30, type=int)
    parser.add_argument('--n-best-size',
            help='Total number of n-best predictions to generate in the nbest_predictions.json output file',
            default=20, type=int)
    parser.add_argument('--doc-stride',
            help='When splitting up a long document into chunks, what stride to take between chunks',
            default=128, type=int)
    parser.add_argument('--target_qps',
            help="target qps metric", required=False, type=int)
    parser.add_argument("-i", "--int8", action="store_true", help="Indicates that inference should be run in INT8 precision", required=False)
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    paragraph_text = None
    squad_examples = None
    output_prediction_file = None

    if not args.passage == '':
        paragraph_text = ' '.join(args.passage)
    elif not args.passage_file == '':
        f = open(args.passage_file, 'r')
        paragraph_text = f.read()
    elif not args.squad_json == '':
        squad_examples = dp.read_squad_json(args.squad_json)
        output_prediction_file = args.output_prediction_file
    else:
        paragraph_text = input("Paragraph: ")

    question_text = None
    if not args.question == '':
        question_text = ' '.join(args.question)
    elif not args.question_file == '':
        f = open(args.question_file, 'r')
        question_text = f.read()

    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    # When splitting up a long document into chunks, how much stride to take between chunks.
    doc_stride = args.doc_stride
    # The maximum total input sequence length after WordPiece tokenization.
    # Sequences longer than this will be truncated, and sequences shorter
    max_seq_length = args.sequence_length

    def question_features(tokens, question):
        # Extract features from the paragraph and question
        return dp.convert_example_to_features(tokens, question, tokenizer, max_seq_length, doc_stride, args.max_query_length)

    load_ixrt_plugin(TRT_LOGGER)

    # The first context created will use the 0th profile. A new context must be created
    # for each additional profile needed. Here, we only use batch size 1, thus we only need the first profile.
    with open(args.engine, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        # select engine profile
        selected_profile = -1
        num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles
        for idx in range(engine.num_optimization_profiles):
            profile_shape = engine.get_profile_shape(profile_index = idx, binding = idx * num_binding_per_profile)
            if profile_shape[0][0] <= args.batch_size and profile_shape[2][0] >= args.batch_size and profile_shape[0][1] <= max_seq_length and profile_shape[2][1] >= max_seq_length:
                selected_profile = idx
                break
        if selected_profile == -1:
            raise RuntimeError("Could not find any profile that can run batch size {}.".format(args.batch_size))

        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()
        
        # if args.use_trt:
        #     context.active_optimization_profile = selected_profile
        # else:
        context.set_optimization_profile_async(selected_profile, stream.handle)
        binding_idx_offset = selected_profile * num_binding_per_profile

        input_shape = (args.batch_size, max_seq_length)
        input_nbytes = trt.volume(input_shape) * 4
        for binding in range(3):
            context.set_binding_shape(binding, input_shape)
        assert context.all_binding_shapes_specified

        # Allocate device memory for inputs.
        d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(binding_idx_offset + 3)), dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)

        def inference(features, tokens):
            global h_output

            _NetworkOutput = collections.namedtuple(  # pylint: disable=invalid-name
                    "NetworkOutput",
                    ["start_logits", "end_logits", "feature_index"])
            networkOutputs = []

            eval_time_elapsed = 0
            for feature_index, feature in enumerate(features):
                # Copy inputs
                input_ids_batch = np.repeat(np.expand_dims(feature.input_ids, 0), args.batch_size, axis=0)
                segment_ids_batch = np.repeat(np.expand_dims(feature.segment_ids, 0), args.batch_size, axis=0)
                input_mask_batch = np.repeat(np.expand_dims(feature.input_mask, 0), args.batch_size, axis=0)

                input_ids = cuda.register_host_memory(np.ascontiguousarray(input_ids_batch.ravel()))
                segment_ids = cuda.register_host_memory(np.ascontiguousarray(segment_ids_batch.ravel()))
                input_mask = cuda.register_host_memory(np.ascontiguousarray(input_mask_batch.ravel()))

                eval_start_time = time.time()
                cuda.memcpy_htod_async(d_inputs[0], input_ids, stream)
                cuda.memcpy_htod_async(d_inputs[1], segment_ids, stream)
                cuda.memcpy_htod_async(d_inputs[2], input_mask, stream)

                # Run inference
                context.execute_async_v2(bindings=[0 for i in range(binding_idx_offset)] +[int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
                # Synchronize the stream
                stream.synchronize()
                eval_time_elapsed += (time.time() - eval_start_time)

                # Transfer predictions back from GPU
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
                # for x in h_output[0].reshape(-1,2):
                #     print(x)
                # Only retrieve and post-process the first batch
                batch = h_output[0]
                
                networkOutputs.append(_NetworkOutput(
                    start_logits = np.array(batch.squeeze()[:, 0]),
                    end_logits = np.array(batch.squeeze()[:, 1]),
                    feature_index = feature_index
                    ))
            
            eval_time_elapsed /= len(features)

            # Total number of n-best predictions to generate in the nbest_predictions.json output file
            n_best_size = 20

            # The maximum length of an answer that can be generated. This is needed
            # because the start and end predictions are not conditioned on one another
            max_answer_length = 30

            prediction, nbest_json, scores_diff_json = dp.get_predictions(tokens, features,
                    networkOutputs, args.n_best_size, args.max_answer_length)

            return eval_time_elapsed, prediction, nbest_json

        def print_single_query(eval_time_elapsed, prediction, nbest_json):
            print("------------------------")
            print("Running inference in {:.3f} Sentences/Sec".format(args.batch_size/eval_time_elapsed))
            print("------------------------")

            print("Answer: '{}'".format(prediction))
            print("With probability: {:.3f}".format(nbest_json[0]['probability'] * 100.0))

        def inference_all_dynamic(features_list, squad_examples, sort_index, all_precision):
            # h_output = torch.tensor((args.batch_size, max_seq_length, 2))
            global h_output
            _NetworkOutput = collections.namedtuple(  # pylint: disable=invalid-name
                    "NetworkOutput",
                    ["start_logits", "end_logits", "feature_index"])
            networkOutputs = []    
        
            batch_input_ids = []
            batch_segment_ids = []
            all_token_ids = []
            batch_example_list = []
            batch_feature_list = []
            batch_feature = []
            batch_example = []
            max_batch_length = 0
            seq_length_list = []
            for index in sort_index:
                batch_feature.append(features_list[index])
                batch_example.append(squad_examples[index])
                max_batch_length = max(max_batch_length, len(features_list[index].input_ids))
                if args.int8:
                    max_batch_length = math.ceil(max_batch_length / 2) * 2
                else:
                    # workround to solve bs=1 10% slow
                    if args.batch_size == 1:
                        max_batch_length = math.ceil(max_batch_length / 64) * 64
                seq_length_list.append(len(features_list[index].input_ids))
                if len(batch_feature) == args.batch_size:
                    batch_input_ids = [
                        np.pad(bf.input_ids, (0, max_batch_length - bf.input_ids.shape[0]), 'constant',constant_values = (0)).reshape(1, -1)
                            for bf in batch_feature
                    ]
                    batch_input_ids = np.concatenate(batch_input_ids, axis=0)
                    batch_segment_ids = [
                        np.pad(bf.segment_ids, (0, max_batch_length - bf.segment_ids.shape[0]), 'constant',constant_values = (0)).reshape(1, -1)
                            for bf in batch_feature
                    ]
                    batch_segment_ids = np.concatenate(batch_segment_ids, axis=0)
                    all_token_ids.append(
                        [
                            batch_input_ids.astype(np.int32),
                            batch_segment_ids.astype(np.int32)
                        ]
                    )
                    batch_example_list.append(batch_example)
                    batch_feature_list.append(batch_feature)
                    batch_input_ids = []
                    batch_segment_ids = []
                    batch_feature = []
                    batch_example = []
                    max_batch_length = 0

            if len(batch_feature):
                batch_input_ids = [
                    np.pad(bf.input_ids, (0, max_batch_length - bf.input_ids.shape[0]), 'constant',constant_values = (0)).reshape(1, -1)
                        for bf in batch_feature
                ]
                batch_input_ids = np.concatenate(batch_input_ids, axis=0)
                batch_segment_ids = [
                    np.pad(bf.segment_ids, (0, max_batch_length - bf.segment_ids.shape[0]), 'constant',constant_values = (0)).reshape(1, -1)
                        for bf in batch_feature
                ]
                batch_segment_ids = np.concatenate(batch_segment_ids, axis=0)
                all_token_ids.append(
                    [
                        batch_input_ids.astype(np.int32),
                        batch_segment_ids.astype(np.int32)
                    ]
                )
                batch_input_ids = []
                batch_segment_ids = []
                batch_example_list.append(batch_example)
                batch_feature_list.append(batch_feature)

            # warm up
            for i in range(20):
                for binding in range(3):
                    context.set_binding_shape(binding, (args.batch_size, max_seq_length))
                assert context.all_binding_shapes_specified
                cuda.memcpy_htod_async(d_inputs[0], np.zeros((args.batch_size, max_seq_length), dtype=np.int32).ravel(), stream)
                cuda.memcpy_htod_async(d_inputs[1], np.zeros((args.batch_size, max_seq_length), dtype=np.int32).ravel(), stream)
                context.execute_async_v2(bindings=[0 for i in range(binding_idx_offset)] +[int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
            stream.synchronize()
            
            infer_toal_time = 0
            output_index = 0
            for input_ids, segment_ids in tqdm(all_token_ids):
                for binding in range(3):
                    context.set_binding_shape(binding, input_ids.shape)
                assert context.all_binding_shapes_specified

                cuda.memcpy_htod_async(d_inputs[0], input_ids.ravel(), stream)
                cuda.memcpy_htod_async(d_inputs[1], segment_ids.ravel(), stream)
                stream.synchronize()

                infer_start_time = time.time()
                context.execute_async_v2(bindings=[0 for i in range(binding_idx_offset)] +[int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
                stream.synchronize()
                infer_end_time = time.time()
                infer_time = infer_end_time - infer_start_time
                infer_toal_time += infer_time
                
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
    
                new_h_output = np.array(h_output.reshape(-1)[:input_ids.shape[0]*input_ids.shape[1]*2]).reshape(input_ids.shape[0], input_ids.shape[1], 2)
                for index in range(input_ids.shape[0]):
                    networkOutputs.append(_NetworkOutput(
                        start_logits = new_h_output[index, :seq_length_list[output_index], 0],
                        end_logits = new_h_output[index, :seq_length_list[output_index], 1],
                        feature_index = index
                    ))
                    output_index += 1

            output_index = 0
            for (be, bf) in zip(batch_example_list, batch_feature_list):
                for index in range(len(bf)):
                    prediction, nbest_json, scores_diff_json = dp.get_predictions(be[index].doc_tokens, bf,
                        [networkOutputs[output_index]], args.n_best_size, args.max_answer_length)
                    output_index += 1
                    all_precision[be[index].id] = prediction
            return infer_toal_time, all_precision

        status = 0
        if squad_examples:
            all_predictions = collections.OrderedDict()

            features_list = []
            lengths = []

            for example_index, example in enumerate(squad_examples):
                features = question_features(example.doc_tokens, example.question_text)
                features_list.append(features[0])
                lengths.append(len(features[0].input_ids))

            sort_index = np.argsort(lengths)
            infer_time, all_predictions = inference_all_dynamic(features_list, squad_examples, sort_index, all_predictions)
            print(F"E2E time : {infer_time:.3f} seconds")
            
            qps = math.ceil(len(squad_examples)/args.batch_size)*args.batch_size/infer_time
            print(f"Latency QPS: {qps} sentences/s")

            metricResult = {"metricResult": {}}
            metricResult["metricResult"]["E2E time"] = round(infer_time, 3)
            metricResult["metricResult"]["Latency QPS"] = round(qps, 3)
            print(metricResult)

            with open(output_prediction_file, "w") as f:
                f.write(json.dumps(all_predictions, indent=4))
                print("\nOutput dump to {}".format(output_prediction_file))

            if args.target_qps:
                if qps >= args.target_qps:
                    print(f"target qps: {args.target_qps}, qps: {qps}, pass.")
                else:
                    print(f"target qps: {args.target_qps}, qps: {qps}, failed.")
                    status = 1
        else:
            # Extract tokecs from the paragraph
            doc_tokens = dp.convert_doc_tokens(paragraph_text)

            if question_text:
                print("\nPassage: {}".format(paragraph_text))
                print("\nQuestion: {}".format(question_text))

                features = question_features(doc_tokens, question_text)
                eval_time_elapsed, prediction, nbest_json = inference(features, doc_tokens)
                print_single_query(eval_time_elapsed, prediction, nbest_json)
            else:
                # If no question text is provided, loop until the question is 'exit'
                EXIT_CMDS = ["exit", "quit"]
                question_text = input("Question (to exit, type one of {:}): ".format(EXIT_CMDS))

                while question_text.strip() not in EXIT_CMDS:
                    features = question_features(doc_tokens, question_text)
                    eval_time_elapsed, prediction, nbest_json = inference(features, doc_tokens)
                    # print_single_query(eval_time_elapsed, prediction, nbest_json)
                    # question_text = input("Question (to exit, type one of {:}): ".format(EXIT_CMDS))
        del context
        del engine
        sys.exit(status)