import numpy as np 
import cv2
import glob
import os 
import math
from tqdm import tqdm
import time
from .db_postprocess import DBPostProcess
from .eval_det_iou import DetectionIoUEvaluator
from .common import  get_io_bindings
import torch
from cuda import cuda, cudart


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}icdar_2015_images{os.sep}', f'{os.sep}icdar_2015_labels{os.sep}gt_'  # /images/, /labels/ substrings
    
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def rotate(angle, x, y):
    """
    基于原点的弧度旋转

    :param angle:   弧度
    :param x:       x
    :param y:       y
    :return:
    """
    rotatex = math.cos(angle) * x - math.sin(angle) * y
    rotatey = math.cos(angle) * y + math.sin(angle) * x
    return rotatex, rotatey

def xy_rorate(theta, x, y, centerx, centery):
    """
    针对中心点进行旋转

    :param theta:
    :param x:
    :param y:
    :param centerx:
    :param centery:
    :return:
    """
    r_x, r_y = rotate(theta, x - centerx, y - centery)
    return centerx+r_x, centery+r_y

def rec_rotate(x, y, width, height, theta):
    """
    传入矩形的x,y和宽度高度,弧度,转成QUAD格式
    :param x:
    :param y:
    :param width:
    :param height:
    :param theta:
    :return:
    """
    centerx = x + width / 2
    centery = y + height / 2

    x1, y1 = xy_rorate(theta, x, y, centerx, centery)
    x2, y2 = xy_rorate(theta, x+width, y, centerx, centery)
    x3, y3 = xy_rorate(theta, x, y+height, centerx, centery)
    x4, y4 = xy_rorate(theta, x+width, y+height, centerx, centery)

    return [(int(x1), int(y1)), (int(x2), int(y2)),  (int(x4), int(y4)), (int(x3), int(y3))]  #clock wise


def letterbox(im, new_shape=(736, 1280), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints

    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
        
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
    
    # Compute padding
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape != new_unpad:  # resize
        im = cv2.resize(im, new_unpad[::-1], interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im1 = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im1, r, dw, dh


def draw_det_res(img,dt_boxes):
    if len(dt_boxes) > 0:
        src_im = img
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 114, 255), thickness=2)
        cv2.imwrite("det3.jpg", src_im)


class TextDetector(object):
    def __init__(self,engine,context, configs):   
        self.engine= engine
        self.context = context
        self.configs = configs
        self.postprocess = DBPostProcess()
    def batch_forward(self,inputs,outputs,allocations,batch_data,shape_list):
        
        output = np.zeros(outputs[0]["shape"], outputs[0]["dtype"])
        input = np.zeros(inputs[0]["shape"], inputs[0]["dtype"])
        real_batch = batch_data.shape[0]
        batch_data= np.transpose(batch_data,[0,3,1,2])
        batch_data = batch_data.astype(inputs[0]["dtype"])
        batch_data = np.ascontiguousarray(batch_data)
        input[:real_batch, :, :, :] = batch_data
        
        err, = cuda.cuMemcpyHtoD(inputs[0]["allocation"], batch_data, batch_data.nbytes)
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        self.context.execute_v2(allocations)
        err, = cuda.cuMemcpyDtoH(output, outputs[0]["allocation"], outputs[0]["nbytes"])
        assert(err == cuda.CUresult.CUDA_SUCCESS)
        outs_dict={"maps":output}
        post_result = self.postprocess(outs_dict,shape_list)
        return post_result
    
    def get_dataloader(self,datasets_dir, bsz):
        image_files= glob.glob(str(datasets_dir+"/*"))
        batch_img, shape_list, batch_label_files = [],[],[] 
        label_files = img2label_paths(image_files)
        for image_file,label_file in zip(image_files,label_files):
            img = cv2.imread(image_file)    
            letter_img, img, org_img, scale, pad_w, pad_h = self.pre_process(image_file)
            shape_list.append([img.shape[0],img.shape[1],pad_h,pad_w,scale]) 
            batch_img.append(np.expand_dims(img, 0))
            batch_label_files.append(label_file)
            if len(batch_img) == bsz:
                yield np.concatenate(batch_img, 0), np.array(shape_list).astype(np.int32),batch_label_files
                batch_img, shape_list, batch_label_files = [],[],[]
    
        if len(batch_img) > 0:
            yield np.concatenate(batch_img, 0), np.array(shape_list), batch_label_files
    
    def eval_icdar_2015(self,img_dir,batch_size):
        dataloader = self.get_dataloader(img_dir,batch_size)
        label_files =[]
        evaluator = DetectionIoUEvaluator()
        
        inputs, outputs, allocations = get_io_bindings(self.engine)        
        gts =[]
        preds=[]
        all_boxes= []
        for i, data in enumerate(tqdm(dataloader,disable=False)):        
            batch_data, shape_list,batch_label = data
            label_files.extend(batch_label)
            post_result= self.batch_forward(inputs,outputs,allocations,batch_data,shape_list)
            all_boxes.extend(post_result)
        print("============start evel=========================")    
        for i, per_image_boxes in  enumerate(all_boxes):
            one_pred=[]
            dt_boxes = per_image_boxes["points"]
            for bbox in dt_boxes:
                one_pred_res={}
                one_pred_res["points"]=[tuple(x) for x in bbox.tolist()] 
                one_pred_res["text"]="text"
                one_pred_res["ignore"] =False
                one_pred.append(one_pred_res)
            preds.append(one_pred)
            label_file= label_files[i] 
            one_gt=[]  
            with open(label_file) as f:
                lines = f.readlines()
                for line in lines:
                    one_gt_res={}                  
                    line_label=line.strip().split(",")[:9]
                    x1,y1,x2,y2,x3,y3,x4,y4,label =line_label
                    gt_bbox=  [(int(x1), int(y1)), (int(x2), int(y2)),  (int(x3), int(y3)), (int(x4), int(y4))]
                    one_gt_res["points"]=gt_bbox
                    one_gt_res["text"]=label
                    if label=="###":
                        one_gt_res["ignore"] =True
                    else:
                        one_gt_res["ignore"] =False
                    one_gt.append(one_gt_res)
                gts.append(one_gt)
                
                 
        results = []
        for gt, pred in zip(gts, preds):
            results.append(evaluator.evaluate_image(gt, pred))
            metrics = evaluator.combine_results(results)     
        return metrics
        
    def perf(self,warm_up,loop_count,batch_size):
        inputs, outputs, allocations = get_io_bindings(self.engine)        
        if warm_up > 0:
            print("\nWarm Start.")
            for i in range(warm_up):
                self.context.execute_v2(allocations)
            print("Warm Done.")
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(loop_count):
            self.context.execute_v2(allocations)
        torch.cuda.synchronize()
        end_time = time.time()
        forward_time = end_time - start_time
        fps = loop_count * batch_size / forward_time
        fps = round(fps,2)
        return fps
        
        
    def pre_process(self,img_file):
        org_img = cv2.imread(img_file)
        image = org_img.copy()
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        letter_img, r, dw, dh= letterbox(image,self.configs["image_shape"])        
        in_img = letter_img.copy()
        #image = cv2.resize(image, (1280, 736))
        in_img = in_img.astype(np.float32)
        in_img /= 255 
        in_img =(in_img-0.456)/0.224    
        return letter_img,in_img,org_img, r, dw, dh   
     
         
        
        
        
    
