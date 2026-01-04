import os
import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from datasets.coco import CocoDetection

def create_dataloaders(data_path, annFile, img_sz=640, batch_size=32, step=32, workers=1, data_process_type="yolov5"):
    dataset = CocoDetection(
        root=data_path,
        annFile=annFile,
        img_size=img_sz,
        data_process_type=data_process_type
    )
    calibration_dataset = dataset
    num_samples = min(5000, batch_size * step)
    if num_samples > 0:
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
    return calibration_dataloader