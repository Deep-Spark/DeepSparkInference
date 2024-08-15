import math
from collections import defaultdict
from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence


def remove_duplicates_and_blank(hyp: List[int],
                                blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    mask = mask[:, 2::2][:, 2::2]
    return mask


class DecodeResult:

    def __init__(self,
                 tokens: List[int],
                 score: float = 0.0,
                 confidence: float = 0.0,
                 tokens_confidence: List[float] = None,
                 times: List[int] = None,
                 nbest: List[List[int]] = None,
                 nbest_scores: List[float] = None,
                 nbest_times: List[List[int]] = None):
        """
        Args:
            tokens: decode token list
            score: the total decode score of this result
            confidence: the total confidence of this result, it's in 0~1
            tokens_confidence: confidence of each token
            times: timestamp of each token, list of (start, end)
            nbest: nbest result
            nbest_scores: score of each nbest
            nbest_times:
        """
        self.tokens = tokens
        self.score = score
        self.confidence = confidence
        self.tokens_confidence = tokens_confidence
        self.times = times
        self.nbest = nbest
        self.nbest_scores = nbest_scores
        self.nbest_times = nbest_times


def ctc_greedy_search(ctc_probs: torch.Tensor,
                      ctc_lens: torch.Tensor,
                      blank_id: int = 0) -> List[DecodeResult]:

    batch_size = ctc_probs.shape[0]
    maxlen = ctc_probs.size(1)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)

    mask_ctc_lens = ctc_lens[0].item()
    mask = make_pad_mask(ctc_lens, mask_ctc_lens)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(mask, blank_id)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    scores = topk_prob.max(1)
    results = []
    for hyp in hyps:
        results.append(remove_duplicates_and_blank(hyp, blank_id))
    return results

