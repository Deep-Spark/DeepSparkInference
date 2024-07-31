import os
import torch
import numpy as np

from .embedding import RelPositionalEncoding


rel_positional_encoding = RelPositionalEncoding(256, 0.1)


def make_pad_mask(lengths: np.ndarray, max_len: int = 0) -> np.ndarray :
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (numpy.ndarray): Batch of lengths (B,).
    Returns:
        numpy.ndarray: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """

    batch_size = lengths.shape[0]
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = np.arange(0, max_len, dtype=np.int64)
    seq_range_expand = np.tile(seq_range, batch_size).reshape(batch_size, max_len)
    seq_length_expand = lengths[..., None]
    mask = seq_range_expand >= seq_length_expand
    mask = np.expand_dims(mask, axis=1)
    mask = ~mask
    mask = mask[:, :, 2::2][:, :, 2::2]
    mask = mask.astype(np.int32)
    return mask
