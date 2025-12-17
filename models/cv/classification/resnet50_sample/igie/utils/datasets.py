import os
import cv2
import numpy as np
import tensorflow as tf
try:
    tf = tf.compat.v1
except ImportError:
    tf = tf
tf.enable_eager_execution()
import glob
from PIL import Image
import random
import logging
import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset

from pycocotools.coco import COCO
from utils.yolo_utils.misc import letterbox, xywhn2xyxy, xyxy2xywhn, coco91_to_coco80_dict
from .common import get_input_shape

### Tensorflow image pre-process function
def _mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel."""
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)

def _central_crop(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

def _aspect_preserving_resize(image, resize_min):
    """Resize images preserving the original aspect ratio.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, resize_min)
    return _resize_image(image, new_height, new_width)

def _smallest_size_at_least(height, width, resize_min):
    resize_min = tf.cast(resize_min, tf.float32)
    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim
    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)
    return new_height, new_width

def _resize_image(image, height, width):
    """Simple wrapper around tf.resize_images."""
    return tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False)



### Pytorch image pre-process function
def _torch_imagenet_preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    # preprocess image to nomalized tensor for pytorch
    _PYTORCH_IMAGENET_PREPROCESS = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    img = _PYTORCH_IMAGENET_PREPROCESS(img)
    return img


### ImageNet datasets
class ImageNetDataset(Dataset):
    def __init__(self, image_dir_path, label_dir_path="", layout="NHWC", image_size=(224, 224)):
        super(Dataset, self).__init__()
        self.image_dir_path = image_dir_path
        self.label_dir_path = label_dir_path
        self.layout = layout
        
        if len(image_size) == 1:
            self.image_height = self.image_width = image_size
        if len(image_size) == 2:
            self.image_height = image_size[0]
            self.image_width = image_size[1]
        assert self.layout in ["NHWC", "NCHW"], f"layout should be NHWC or NCHW, got {self.layout} "
        self.img_list = os.listdir(self.image_dir_path)
        self.label_dict = self.get_label_dict()
        
        self.images = []
        self.length = 0

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        
        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

        for image_dir in self.img_list:
            image_path = os.path.join(self.image_dir_path, image_dir)
            if os.path.isdir(image_path):
                for image in os.listdir(image_path):
                    self.images.append(os.path.join(image_path, image))
                    self.length += 1
            
            elif os.path.isfile(image_path):
                if is_image(image_path):
                    self.length += 1
                    self.images.append(image_path)
    
    def __getitem__(self, index):
        ## NHWC pre-process for tensorflow
        if self.layout == "NHWC":
            image = cv2.imread(self.images[index])
            image = cv2.cvtColor(image, 4)
            resize_image = _aspect_preserving_resize(image, 256)
            crop_image = _central_crop(resize_image, self.image_height, self.image_width)  
            crop_image.set_shape([self.image_height, self.image_width, 3])
            crop_image = tf.to_float(crop_image)
            processed_image = _mean_image_subtraction(crop_image, [123.68, 116.78, 103.94]).numpy()
        
        ## NCHW pre-process for Pytorch
        elif self.layout == "NCHW":
            processed_image = _torch_imagenet_preprocess(self.images[index])
        else:
            raise ValueError("Unsupported data layout")

        image_name = self.images[index].split('/')[-1].strip()
        label = self.label_dict[image_name]

        return processed_image, label

    def __len__(self):
        return self.length

    def get_label_dict(self):
        image_label = {}
        label_path = os.path.join(self.image_dir_path, 'val_map.txt')
        if self.label_dir_path != "":
            label_path = self.label_dir_path
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                lines = file.readlines()
        
            for line in lines:
                image = line.split('\t')[0].strip()
                label = line.split('\t')[1].strip()
                image_label[image] = int(label)
        
        return image_label
    

# ImageNet100 datasets
class UserCustomDataset(Dataset):
    def __init__(self, image_dir_path, label_dir_path="",  preprocess_func=_torch_imagenet_preprocess, shuffle_files=False):
        super(Dataset, self).__init__()
        self.image_dir_path = image_dir_path
        self.label_dir_path = label_dir_path
        
        self.preprocess_function = preprocess_func
        self.images = []
        self.length = 0

        # Find images in the given input path
        print(self.image_dir_path)
        input = os.path.realpath(self.image_dir_path)
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        
        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

        if os.path.isdir(input):
            self.images = glob.glob(str(Path(input) / '**' / '*.*'), recursive=True)
            self.images.sort()
            if shuffle_files:
                random.seed(47)
                random.shuffle(self.images)
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        
        self.length = len(self.images)
        if self.length < 1:
            raise RuntimeError("No valid {} images found in {}".format("/".join(extensions), input))
        
        self.label_dict = self.get_label_dict()
            

    def __getitem__(self, index):
        # TODO: add image pre-process
        image = self.preprocess_function(self.images[index])
        label = int(self.images[index].split('/')[-2].split('Class')[-1]) - 1
        return image, label

    def __len__(self):
        return self.length

    def get_label_dict(self):
        # TODO: add label process
        image_label = {}
        for image in self.images:
            image_name = image.split('/')[-1].strip()
            image_label[image_name] = int(1)
        
        return image_label

    @staticmethod
    def collate_fn(batch):
        im, label = zip(*batch)
        # print(np.array(im).shape)
        # print("label: ", label)
        return np.array([i[0] for i in im]), np.array(label)

   

### cifar Dataset
import pickle
mean = np.array([129.304, 124.07, 112.434]).reshape(1, 3, 1, 1).astype("float32")
std = np.array([68.17, 65.392, 70.418]).reshape(1, 3, 1, 1).astype("float32")

def normlize(data):
    data = data.astype("float32")
    return (data - mean) / std

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

class CifarDataset(Dataset):
    def __init__(self, images, labels):
        self.index = 0
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return normlize(img.reshape(1, 3, 32, 32)), target

    def __len__(self):
        return len(self.images)
   
def load_cifar_dataset(data_path):
    data = unpickle(data_path)
    images = data['data']
    labels = data['fine_labels']
    cifar_dataset = CifarDataset(images, labels)
    return cifar_dataset


### COCO128 Dataset
class DetectionCOCO128Dataset(Dataset):

    def __init__(
        self,
        image_dir_path,
        label_dir_path,
        layout="NCHW",
        image_size=640,
        stride=32,
        pad_color=114,
    ):

        self.image_dir_path = image_dir_path
        self.label_dir_path = label_dir_path
        self.image_size = image_size
        self.stride = stride
        self.pad_color = pad_color
        self.layout = layout

        self.img_files = sorted(
            glob.glob(os.path.join(self.image_dir_path, "*.*")))
        self.label_files = sorted(
            glob.glob(os.path.join(self.label_dir_path, "*.*")))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # load image
        img, (h0, w0), (h, w) = self._load_image(index)

        # letterbox
        img, ratio, pad = letterbox(img,
                                    self.image_size,
                                    color=(self.pad_color, self.pad_color,
                                           self.pad_color))
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # load label
        raw_label = self._load_txt_label(index)
        nl = len(raw_label)  # number of labels
        if nl:
            # normalized xywh to pixel xyxy format
            raw_label[:, 1:] = xywhn2xyxy(raw_label[:, 1:],
                                          ratio[0] * w,
                                          ratio[1] * h,
                                          padw=pad[0],
                                          padh=pad[1])

            raw_label[:, 1:] = xyxy2xywhn(raw_label[:, 1:],
                                          w=img.shape[1],
                                          h=img.shape[0],
                                          clip=True,
                                          eps=1E-3)

        labels_out = np.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = raw_label

        # Convert
        img = img.transpose(2, 0, 1)[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.0  # 0~1 np array
        
        if self.layout == "NHWC":
            img = img.transpose(1, 2, 0)

        return img, labels_out, self.img_files[index], shapes

    def _load_image(self, i):
        im = cv2.imread(self.img_files[i])  # BGR
        h0, w0 = im.shape[:2]  # orig hw
        r = self.image_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_LINEAR)
        return im.astype("float32"), (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

    def _load_txt_label(self, i):
        label_path = self.img_files[i].replace("images", "labels").replace(
            ".jpg", ".txt")

        labels = []
        if label_path in self.label_files:
            with open(label_path) as f:
                data = f.read().splitlines()

            for i in data:
                labels.append([float(j) for j in i.split()])

        return np.array(labels)

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return np.concatenate([i[None] for i in im], axis=0), np.concatenate(label, 0), path, shapes
        
        
### COCO2017 Dataset
class DetectionCOCO2017Dataset(Dataset):

    def __init__(self,
                 image_dir_path,
                 annotation_json_path,
                 layout="NCHW", 
                 image_size=640,
                 stride=32,
                 val_mode=True,
                 pad_color=114):

        self.image_dir_path = image_dir_path
        self.annotation_json_path = annotation_json_path
        self.image_size = image_size
        self.stride = stride
        self.val_mode = val_mode
        self.pad_color = pad_color
        self.layout = layout

        self.coco = COCO(annotation_file=self.annotation_json_path)
        if self.val_mode:
            self.img_ids = list(sorted(self.coco.imgs.keys()))  # 5000
        else:  # train mode need images with labels
            self.img_ids = sorted(list(self.coco.imgToAnns.keys()))  # 4952

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        # load image
        img_path = self._get_image_path(index)
        img, (h0, w0), (h, w) = self._load_image(index)

        # letterbox
        img, ratio, pad = letterbox(img,
                                    self.image_size,
                                    color=(self.pad_color, self.pad_color,
                                           self.pad_color))
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # load label
        raw_label = self._load_json_label(index)
        # normalized xywh to pixel xyxy format
        raw_label[:, 1:] = xywhn2xyxy(raw_label[:, 1:],
                                      ratio[0] * w,
                                      ratio[1] * h,
                                      padw=pad[0],
                                      padh=pad[1])

        raw_label[:, 1:] = xyxy2xywhn(raw_label[:, 1:],
                                      w=img.shape[1],
                                      h=img.shape[0],
                                      clip=True,
                                      eps=1E-3)

        nl = len(raw_label)  # number of labels
        labels_out = np.zeros((nl, 6))
        labels_out[:, 1:] = raw_label

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.0  # 0~1 np array

        if self.layout == "NHWC":
            img = img.transpose(1, 2, 0)
        
        return img, labels_out, img_path, shapes

    def _get_image_path(self, index):
        idx = self.img_ids[index]
        path = self.coco.loadImgs(idx)[0]["file_name"]
        img_path = os.path.join(self.image_dir_path, path)
        return img_path

    def _load_image(self, index):
        img_path = self._get_image_path(index)

        im = cv2.imread(img_path)  # BGR
        h0, w0 = im.shape[:2]  # orig hw
        r = self.image_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_LINEAR)
        return im.astype("float32"), (
            h0, w0), im.shape[:2]  # im, hw_original, hw_resized

    def _load_json_label(self, index):
        _, (h0, w0), _ = self._load_image(index)

        idx = self.img_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=idx)
        targets = self.coco.loadAnns(ids=ann_ids)

        labels = []
        for target in targets:
            cat = target["category_id"]
            coco80_cat = coco91_to_coco80_dict[cat]
            cat = np.array([[coco80_cat]])

            x, y, w, h = target["bbox"]
            x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
            xyxy = np.array([[x1, y1, x2, y2]])
            xywhn = xyxy2xywhn(xyxy, w0, h0)
            labels.append(np.hstack((cat, xywhn)))

        if labels:
            labels = np.vstack(labels)
        else:
            if self.val_mode:
                # for some image without label
                labels = np.zeros((1, 5))
            else:
                raise ValueError(
                    f"set val_mode = False to use images with labels")

        return labels

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return np.concatenate([i[None] for i in im], axis=0), np.concatenate(label, 0), path, shapes
    
    
### Fake datasets
def make_fake_dataset(args, n=128):
    input_shape = get_input_shape(args.input_shape)
    print(f"input_shape: {input_shape}")
    input_shape[0] *= n
    valid_x = torch.randn(*input_shape)
    valid_y = torch.randint(args.num_classes, (input_shape[0], 1))
    dataset = TensorDataset(valid_x, valid_y)
    
    batch = input_shape[0]
    def collate_fn(data):
        im, label = data[0][0], data[0][1]
        imgs = []
        labels = []
        for b in range(batch):
            imgs.append(np.array(im))
            labels.append(np.array(label))
        return np.array(imgs), np.array(labels)
    dataset.collate_fn = collate_fn
    return dataset

### dataLoader
def get_dataloader(args): 
    batch_size = args.batch_size
    model_layout = args.model_layout
    
    if model_layout == "NHWC":
        image_size = [args.input_shape[1], args.input_shape[2]]
    elif  model_layout == "NCHW":
        image_size = [args.input_shape[2], args.input_shape[3]]
    
    ### 如果没有传入 --data-path 参数，则使用随机Fake数据集
    if args.data_path is None:
        logging.warning('args.data_path is {} , use random tensor as input datasets'.format(args.data_path))
        if args.num_classes is None:
            args.num_classes = 1000
        datasets = make_fake_dataset(args)
        dataLoader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, num_workers=args.workers, pin_memory=True)
    
    ### 根据--data-path 参数解析内置的ImageNet、COCO、Cifar datasets
    ### UserCustomDataset是用户自定义的数据集，需要传入数据预处理函数。
    elif os.path.exists(str(args.data_path)):
        
        if "imagenet" in args.data_path:
            datasets = ImageNetDataset(args.data_path, layout=model_layout, image_size=image_size)
            dataLoader = torch.utils.data.DataLoader(datasets, batch_size, num_workers=args.workers, drop_last=True)
        
        elif "cifar" in args.data_path:
            test_path = args.data_path + os.sep + "test"
            datasets = load_cifar_dataset(test_path)
            dataLoader = torch.utils.data.DataLoader(datasets, batch_size, drop_last=True)
        
        elif "coco" in args.data_path and "128" in args.data_path:
            datasets = DetectionCOCO128Dataset(image_dir_path=args.data_path,
                                        label_dir_path=args.label_path,
                                        layout=model_layout,
                                        image_size=image_size[0])
            dataLoader = torch.utils.data.DataLoader(datasets, batch_size, drop_last=True, collate_fn=datasets.collate_fn)
        
        elif "coco" in args.data_path and "2017" in args.data_path:
            datasets = DetectionCOCO2017Dataset(image_dir_path=args.data_path,
                                        annotation_json_path=args.label_path,
                                        layout=model_layout,
                                        image_size=image_size[0])
            dataLoader = torch.utils.data.DataLoader(datasets, batch_size, drop_last=True, collate_fn=datasets.collate_fn)
        
        else:
            logging.warning(" Using user custom dataset, please check image pre-process.\n")
            datasets = UserCustomDataset(args.data_path, label_dir_path=args.label_path, preprocess_func=_torch_imagenet_preprocess, image_size=image_size)
            dataLoader = torch.utils.data.DataLoader(datasets, batch_size, drop_last=True, collate_fn=datasets.collate_fn)
            
    else:
        raise ValueError("--data-path args {} is not exists, Please set correct data-path.")
    return dataLoader
