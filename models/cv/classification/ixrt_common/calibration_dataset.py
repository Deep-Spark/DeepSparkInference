import os
import numpy as np
from PIL import Image

import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T


class CalibrationImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(CalibrationImageNet, self).__init__(*args, **kwargs)
        img2label_path = os.path.join(self.root, "val_map.txt")
        if not os.path.exists(img2label_path):
            raise FileNotFoundError(f"Not found label file `{img2label_path}`.")

        self.img2label_map = self.make_img2label_map(img2label_path)

    def make_img2label_map(self, path):
        with open(path) as f:
            lines = f.readlines()

        img2lable_map = dict()
        for line in lines:
            line = line.lstrip().rstrip().split("\t")
            if len(line) != 2:
                continue
            img_name, label = line
            img_name = img_name.strip()
            if img_name in [None, ""]:
                continue
            label = int(label.strip())
            img2lable_map[img_name] = label
        return img2lable_map

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        img_name = os.path.basename(path)
        target = self.img2label_map[img_name]

        return sample, target
    
class CalibrationRGBImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(CalibrationRGBImageNet, self).__init__(*args, **kwargs)
        img2label_path = os.path.join(self.root, "val_map.txt")
        if not os.path.exists(img2label_path):
            raise FileNotFoundError(f"Not found label file `{img2label_path}`.")

        self.img2label_map = self.make_img2label_map(img2label_path)

    def make_img2label_map(self, path):
        with open(path) as f:
            lines = f.readlines()

        img2lable_map = dict()
        for line in lines:
            line = line.lstrip().rstrip().split("\t")
            if len(line) != 2:
                continue
            img_name, label = line
            img_name = img_name.strip()
            if img_name in [None, ""]:
                continue
            label = int(label.strip())
            img2lable_map[img_name] = label
        return img2lable_map

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = sample.convert("RGB")
        sample = np.array(sample)[:, :, ::-1]
        sample = Image.fromarray(np.uint8(sample))
        if self.transform is not None:
            sample = self.transform(sample)
        img_name = os.path.basename(path)
        target = self.img2label_map[img_name]

        return sample, target


def create_mobilenetv1_dataloaders(data_path, num_samples=1024, img_sz=224, batch_size=2, workers=0):
    dataset = CalibrationRGBImageNet(
        data_path,
        transform=T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(img_sz),
                T.ToTensor(),
                T.Normalize(mean=[103.940002441/255, 116.779998779/255, 123.680000305/255], std=[1.0/(255*0.0170000009239), 1.0/(255*0.0170000009239), 1.0/(255*0.0170000009239)]),
            ]
        ),
    )

    calibration_dataset = dataset
    if num_samples is not None:
        calibration_dataset = torch.utils.data.Subset(
            dataset, indices=range(num_samples)
        )

    calibration_dataloader = DataLoader(
        calibration_dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
    )

    verify_dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
    )

    return calibration_dataloader, verify_dataloader


def getmobilenetv1dataloader(dataset_dir, step=20, batch_size=32, workers=2, img_sz=224, total_sample=50000):
    num_samples = min(total_sample, step * batch_size)
    if step < 0:
        num_samples = None
    calibration_dataloader, _ = create_mobilenetv1_dataloaders(
        dataset_dir,
        img_sz=img_sz,
        batch_size=batch_size,
        workers=workers,
        num_samples=num_samples,
    )
    return calibration_dataloader


def create_dataloaders(data_path, num_samples=1024, img_sz=224, batch_size=2, workers=0):
    dataset = CalibrationImageNet(
        data_path,
        transform=T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(img_sz),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    calibration_dataset = dataset
    if num_samples is not None:
        calibration_dataset = torch.utils.data.Subset(
            dataset, indices=range(num_samples)
        )

    calibration_dataloader = DataLoader(
        calibration_dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
    )

    verify_dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
    )

    return calibration_dataloader, verify_dataloader


def getdataloader(dataset_dir, step=20, batch_size=32, workers=2, img_sz=224, total_sample=50000):
    num_samples = min(total_sample, step * batch_size)
    if step < 0:
        num_samples = None
    calibration_dataloader, _ = create_dataloaders(
        dataset_dir,
        img_sz=img_sz,
        batch_size=batch_size,
        workers=workers,
        num_samples=num_samples,
    )
    return calibration_dataloader