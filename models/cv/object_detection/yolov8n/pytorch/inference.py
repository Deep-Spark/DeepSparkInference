import os
import argparse
from ultralytics import YOLO
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weight", 
                    type=str, 
                    required=True, 
                    help="pytorch model weight.")
    
    parser.add_argument("--datasets", 
                        type=str, 
                        required=True, 
                        help="datasets path.")
    
    parser.add_argument("--half", 
                        action="store_true",
                        help="use fp16 inference.")
    
    parser.add_argument("--bsize", 
                        type=int,
                        default=32,
                        help="inference bsize .")
    
    parser.add_argument("--imgsz", 
                        type=int,
                        default=640,
                        help="inference image size.")  
    
    parser.add_argument("--perf_only",
                        action="store_true",
                        help="Run performance test only")
    
    args = parser.parse_args()

    return args
    

def main():
    args = parse_args()
    eps=1e-3
    model = YOLO(args.weight)
    results = model.val(data=args.datasets, 
                    device="cuda:0",
                    batch=args.bsize,
                    imgsz=args.imgsz, 
                    half=args.half,
                    verbose=not args.perf_only,
                    conf=0.001)
    if args.perf_only:
        dtype_str = "fp16" if args.half else "fp32"
        speed = results.speed["inference"]
        fps = round(1000 / (speed + eps), 2)
        print(
            f'Benchmark {dtype_str} fps: {fps} inference time : {round(speed, 2)} (ms/frame) '
        )
    
if __name__ == "__main__":
    main()