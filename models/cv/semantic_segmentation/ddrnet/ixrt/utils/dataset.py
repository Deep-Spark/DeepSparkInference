import os
import cv2
import numpy as np
from math import ceil
from tqdm import tqdm


class Dataset:
    def __init__(self, 
                 root, 
                 list_path, 
                 batch_size=4,
                 num_classes=19,
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        self.root = root
        self.list_path = list_path 
        self.batch_size = batch_size
        self.num_classes = num_classes 
        self.mean = mean 
        self.std = std
        self.downsample_rate = downsample_rate

        self.img_list = [line.strip().split() for line in open(list_path)]
        self.files = self.read_files()
        self.num_batches = ceil(len(self.files) / self.batch_size)

        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}
        
        self.batch_images, self.batch_labels, self.batch_sizes, self.batch_names = self.batching()

    def read_files(self):
        files = []
        for i, item in enumerate(self.img_list):
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name,
                "weight": 1
            })
            # if i == 4:
            #    break
        return files
        
    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        temp = label.copy()
        for k, v in self.label_mapping.items():
            label[temp == k] = v
        return np.array(label).astype('int32')
    
    def gen_sample(self, image, label):

        image = self.input_transform(image)
        label = self.label_transform(label)

        if self.downsample_rate != 1:
            label = cv2.resize(
                label,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )
        return image, label

    def _preprocess(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,'cityscapes',item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape
        label = cv2.imread(os.path.join(self.root,'cityscapes',item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        image, label = self.gen_sample(image, label)
        return image.copy(), label.copy(), np.array(size), name 

    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, index):
        return (self.batch_images[index], self.batch_labels[index], self.batch_sizes[index], self.batch_names[index])
    
    def batching(self):
        all_images = []
        all_labels = []
        all_sizes = []
        all_names = []
        
        num_batches = self.num_batches
        batch_size = self.batch_size
        for i in tqdm(range(len(self.files)), desc="Loading Cityscapes Dataset"):
            image, label, size, name = self._preprocess(i)
            all_images.append(image)
            all_labels.append(label)
            all_sizes.append(size)
            all_names.append(name)
        
        batch_images = []
        batch_labels = []
        batch_sizes = []
        batch_names = []
        
        for j in range(num_batches):
            start = j * batch_size 
            if j == num_batches - 1:
                end = None
            else:
                end = (j + 1) * batch_size
            batch_images.append(np.stack(all_images[start:end]))                
            batch_labels.append(np.stack(all_labels[start:end]))                
            batch_sizes.append(np.stack(all_sizes[start:end]))                
            batch_names.append(all_names[start:end])
        return (batch_images, batch_labels, batch_sizes, batch_names)
