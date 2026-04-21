#!/usr/bin/env python
# coding=utf-8

"""
Define function to build confusion_matrix.
"""

import numpy as np


def get_confusion_matrix(label, pred, size, num_class=19, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    
    seg_gt = np.asarray(label[:, :size[-3], :size[-2]], dtype=np.int32)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def get_confusion_matrix_batch(label, pred, size, num_class=19, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred in one batch.
    Arguments:
        label: (batch_size, h, w)
        pred:  (batch_size, h, w, c)
        size:  (batch_size, 2)
    """
    batch_size, h, w, c = pred.shape   
    confusion_matrix = np.zeros((num_class, num_class))
    for i in range(batch_size):
        confusion_matrix += get_confusion_matrix(
            label[i], 
            pred[i:i+1], 
            size[i], 
            19, 
            255
            )
    return confusion_matrix
