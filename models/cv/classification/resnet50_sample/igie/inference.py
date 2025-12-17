import os
import torch
import numpy as np
from tqdm import tqdm
from pprint import pprint

import tvm
from tvm import relay
from tvm.contrib import graph_executor

import logging
logging.basicConfig(level=logging.INFO, format = '[%(asctime)s %(filename)s line:%(lineno)d] %(levelname)s: %(message)s')
logging.getLogger('autotvm').setLevel(logging.INFO)
logging.getLogger('te_compiler').setLevel(logging.ERROR)

from utils.common import get_args_parser, get_input_shape, get_target, get_file_path, first_layer_process

from utils.datasets import get_dataloader
# from ixpylogger.inference_logger import get_inferlogger
# inferlogger = get_inferlogger()

def main():
    args = get_args_parser().parse_args()
    try:
        from dltest import show_infer_arguments
        show_infer_arguments(args)
    except:
        pass

    ### 设置模型初始信息
    input_name = args.input_name
    input_shape = get_input_shape(args.input_shape)
    args.batch_size = input_shape[0]
    
    ### 创建Target设备.
    target, device = get_target(args)

    precision = args.precision
    
    ### 设置engine文件path
    engine_dir = os.getcwd() + "/engine"
    export_engine_path = get_file_path(args, engine_dir)
    
    lib = tvm.runtime.load_module(export_engine_path) 
    
    ### 将engine导入GPU Device设备
    module = graph_executor.GraphModule(lib["default"](device))
    
    ### 7. warmup预热模型推理
    np.random.seed(12345)
    image = np.random.uniform(size=input_shape).astype("float32")
    if precision == "fp16":
        image = np.pad(np.transpose(image, (0, 2, 3, 1)), ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=0).astype("float16")
    elif precision == "int8":
        image = first_layer_process(image, layout=args.model_layout)
    image = tvm.nd.array(image, device)
    input_data = tvm.nd.array(image, device)
    module.set_input(input_name, input_data)
    print("Start Warm-up...")
    for _ in range(args.warmup):
        module.run()
        device.sync()
    
    ### 8. 开始推理模型
    print("Start Model Inference...")
    total_num = 0.0
    total_count = 0
    top1_acc = 0.0
    top5_acc = 0.0
    infernec_throughput = 0.0
    total_infernec_time = 0.0

    ### 使用cudaEvcent计时 or CPU计时
    # cpu_dev = tvm.device("cpu")
    timer = tvm.get_global_func("profiling.get_timer")(device)
    start = tvm.get_global_func("profiling.start")
    stop = tvm.get_global_func("profiling.stop")
    elapse_time = tvm.get_global_func("profiling.elapse_time")

    ### 校验测试结果
    result_stat  = {}
    
    ### 加载数据集dataloader (ImageNet or Cifar or COCO)
    dataloader = get_dataloader(args)

    if args.acc1_target != 0:
        for image, label in tqdm(dataloader):
            if precision == "fp16":
                image = np.pad(np.transpose(image, (0, 2, 3, 1)), ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=0).astype("float16")
            elif precision == "int8":
                image = first_layer_process(image, layout=args.model_layout)
            image = tvm.nd.array(image, device)
            module.set_input(input_name, image)
            device.sync()
            
            ### 执行推理
            start(timer)
            module.run()
            stop(timer)
            
            infernec_time = (elapse_time(timer) / 1e6)  ## ns / 1e6 -> ms
            infernec_throughput = (1000 / infernec_time * args.batch_size)
            total_infernec_time += infernec_time

            ### 获取模型推理的输出结果
            preds = module.get_output(0).asnumpy()
            total_num += image.shape[0]
            total_count += 1
            preds = torch.from_numpy(preds)
            for idx in range(len(label)):
                label_value = label[idx]
                if label_value == torch.topk(preds[idx].float(), 1).indices.data:
                    top1_acc += 1
                    top5_acc += 1

                elif label_value in torch.topk(preds[idx].float(), 5).indices.data:
                    top5_acc += 1

            ### 设置打印输出间隔
            if total_num % 512 == 0:
                logging.info('* Inference Latency:  {:.3f} ms, Inference fps:  {:.3f} fps'.format(infernec_time, infernec_throughput))
                logging.info('* Acc @1 {:.5f} Acc @5 {:.5f}'.format(top1_acc/total_num, top5_acc/total_num))
            
        result_stat["acc@1"] = top1_acc/total_num
        result_stat["acc@5"] = top5_acc/total_num
    
        # inferlogger.report_model_name("resnet50")
        # inferlogger.report_accuracy(running_value=result_stat["acc@1"], target_value=args.acc1_target, metric_name="Top-1")
        print(F"Acc@1 : {top1_acc/total_num}")
        print(F"Acc@5 : {top5_acc/total_num}")

    if args.fps_target != 0 and args.test_count > 0:
        test_count = args.test_count
        fps_target_value = args.fps_target

        for n in range(test_count):
            module.set_input(input_name, input_data)
            device.sync()
            
            ### 执行推理
            start(timer)
            module.run()
            stop(timer)
            infernec_time = (elapse_time(timer) / 1e6)  ## ns / 1e6 -> ms
            infernec_throughput = (1000 / infernec_time * args.batch_size)
            total_infernec_time += infernec_time

            if (total_count - 1) % args.print_freq == 0:
                logging.info('* Inference Latency:  {:.3f} ms, Inference fps:  {:.3f} fps'.format(infernec_time, infernec_throughput))

        result_stat["fps"] = test_count * args.batch_size/total_infernec_time*1000
        result_stat["latency"] = 1.0 / result_stat["fps"]

        # inferlogger.report_model_name("resnet50")
        # inferlogger.report_samples_per_second(running_value=round(result_stat["fps"], 2), target_value=fps_target_value)
        # inferlogger.report_latency(running_value=round(result_stat["latency"], 2), unit='us')
        print("FPS : ", result_stat["fps"])
        print("Latency : ", result_stat["latency"])

    # inferlogger.show_results()
    # infer_status = inferlogger.check_results()
    # exit(int(not infer_status))
        
if __name__ == '__main__':
    main()
