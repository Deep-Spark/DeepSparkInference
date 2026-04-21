#!/usr/bin/env python
# coding=utf-8

import numpy as np
from .dataset import Dataset
from .metrics import get_confusion_matrix


def input_transform(image, mean, std):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image
