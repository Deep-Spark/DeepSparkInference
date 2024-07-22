# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
import torch
from ctc import CTCPrefixScorer
import time

def forward(self, enc_states, wav_len):  # noqa: C901
    """Applies beamsearch and returns the predicted tokens."""
    enc_lens = torch.round(enc_states.shape[1] * wav_len).int()
    device = enc_states.device
    batch_size = enc_states.shape[0]

    memory = self.reset_mem(batch_size * self.beam_size, device=device)

    if self.lm_weight > 0:
        lm_memory = self.reset_lm_mem(batch_size * self.beam_size, device)

    if self.ctc_weight > 0:
        # (batch_size * beam_size, L, vocab_size)
        ctc_outputs = self.ctc_forward_step(enc_states)
        ctc_scorer = CTCPrefixScorer(
            ctc_outputs,
            enc_lens,
            batch_size,
            self.beam_size,
            self.blank_index,
            self.eos_index,
            self.ctc_window_size,
        )
        ctc_memory = None

    # Inflate the enc_states and enc_len by beam_size times
    enc_states = inflate_tensor(enc_states, times=self.beam_size, dim=0)
    enc_lens = inflate_tensor(enc_lens, times=self.beam_size, dim=0)

    # Using bos as the first input
    inp_tokens = (
        torch.zeros(batch_size * self.beam_size, device=device)
        .fill_(self.bos_index)
        .long()
    )

    # The first index of each sentence.
    self.beam_offset = (
        torch.arange(batch_size, device=device) * self.beam_size
    )

    # initialize sequence scores variables.
    sequence_scores = torch.empty(
        batch_size * self.beam_size, device=device
    )
    sequence_scores.fill_(float("-inf"))

    # keep only the first to make sure no redundancy.
    sequence_scores.index_fill_(0, self.beam_offset, 0.0)

    # keep the hypothesis that reaches eos and their corresponding score and log_probs.
    hyps_and_scores = [[] for _ in range(batch_size)]

    # keep the sequences that still not reaches eos.
    alived_seq = torch.empty(
        batch_size * self.beam_size, 0, device=device
    ).long()

    # Keep the log-probabilities of alived sequences.
    alived_log_probs = torch.empty(
        batch_size * self.beam_size, 0, device=device
    )

    min_decode_steps = int(enc_states.shape[1] * self.min_decode_ratio)
    max_decode_steps = int(enc_states.shape[1] * self.max_decode_ratio)

    # Initialize the previous attention peak to zero
    # This variable will be used when using_max_attn_shift=True
    prev_attn_peak = torch.zeros(batch_size * self.beam_size, device=device)

    for t in range(max_decode_steps):
        # terminate condition
        if self._check_full_beams(hyps_and_scores, self.beam_size):
            break
        
        log_probs, memory, attn = self.forward_step(
            inp_tokens, memory, enc_states, enc_lens
        )
        log_probs = self.att_weight * log_probs
        
        # Keep the original value
        log_probs_clone = log_probs.clone().reshape(batch_size, -1)
        vocab_size = log_probs.shape[-1]

        if self.using_max_attn_shift:
            # Block the candidates that exceed the max shift
            cond, attn_peak = self._check_attn_shift(attn, prev_attn_peak)
            log_probs = mask_by_condition(
                log_probs, cond, fill_value=self.minus_inf
            )
            prev_attn_peak = attn_peak

        # Set eos to minus_inf when less than minimum steps.
        if t < min_decode_steps:
            log_probs[:, self.eos_index] = self.minus_inf

        # Set the eos prob to minus_inf when it doesn't exceed threshold.
        if self.using_eos_threshold:
            cond = self._check_eos_threshold(log_probs)
            log_probs[:, self.eos_index] = mask_by_condition(
                log_probs[:, self.eos_index],
                cond,
                fill_value=self.minus_inf,
            )

        # adding LM scores to log_prob if lm_weight > 0
        if self.lm_weight > 0:
            lm_log_probs, lm_memory = self.lm_forward_step(
                inp_tokens, lm_memory
            )
            log_probs = log_probs + self.lm_weight * lm_log_probs

        # adding CTC scores to log_prob if ctc_weight > 0
        if self.ctc_weight > 0:
            g = alived_seq
            # block blank token
            log_probs[:, self.blank_index] = self.minus_inf
            if self.ctc_weight != 1.0 and self.ctc_score_mode == "partial":
                # pruning vocab for ctc_scorer
                _, ctc_candidates = log_probs.topk(
                    self.beam_size * 2, dim=-1
                )
            else:
                ctc_candidates = None

            ctc_log_probs, ctc_memory = ctc_scorer.forward_step(
                g, ctc_memory, ctc_candidates, attn
            )
            log_probs = log_probs + self.ctc_weight * ctc_log_probs
    
        scores = sequence_scores.unsqueeze(1).expand(-1, vocab_size)
        scores = scores + log_probs

        # length normalization
        if self.length_normalization:
            scores = scores / (t + 1)

        # keep topk beams
        scores, candidates = scores.view(batch_size, -1).topk(
            self.beam_size, dim=-1
        )

        # The input for the next step, also the output of current step.
        inp_tokens = (candidates % vocab_size).view(
            batch_size * self.beam_size
        )

        scores = scores.view(batch_size * self.beam_size)
        sequence_scores = scores

        # recover the length normalization
        if self.length_normalization:
            sequence_scores = sequence_scores * (t + 1)

        # The index of which beam the current top-K output came from in (t-1) timesteps.
        predecessors = (
            torch.div(candidates, vocab_size, rounding_mode="floor")
            + self.beam_offset.unsqueeze(1).expand_as(candidates)
        ).view(batch_size * self.beam_size)

        # Permute the memory to synchoronize with the output.
        memory = self.permute_mem(memory, index=predecessors)
        if self.lm_weight > 0:
            lm_memory = self.permute_lm_mem(lm_memory, index=predecessors)

        if self.ctc_weight > 0:
            ctc_memory = ctc_scorer.permute_mem(ctc_memory, candidates)

        # If using_max_attn_shift, then the previous attn peak has to be permuted too.
        if self.using_max_attn_shift:
            prev_attn_peak = torch.index_select(
                prev_attn_peak, dim=0, index=predecessors
            )

        # Add coverage penalty
        if self.coverage_penalty > 0:
            cur_attn = torch.index_select(attn, dim=0, index=predecessors)

            # coverage: cumulative attention probability vector
            if t == 0:
                # Init coverage
                self.coverage = cur_attn

            # the attn of transformer is [batch_size*beam_size, current_step, source_len]
            if len(cur_attn.size()) > 2:
                self.converage = torch.sum(cur_attn, dim=1)
            else:
                # Update coverage
                self.coverage = torch.index_select(
                    self.coverage, dim=0, index=predecessors
                )
                self.coverage = self.coverage + cur_attn

            # Compute coverage penalty and add it to scores
            penalty = torch.max(
                self.coverage, self.coverage.clone().fill_(0.5)
            ).sum(-1)
            penalty = penalty - self.coverage.size(-1) * 0.5
            penalty = penalty.view(batch_size * self.beam_size)
            penalty = (
                penalty / (t + 1) if self.length_normalization else penalty
            )
            scores = scores - penalty * self.coverage_penalty

        # Update alived_seq
        alived_seq = torch.cat(
            [
                torch.index_select(alived_seq, dim=0, index=predecessors),
                inp_tokens.unsqueeze(1),
            ],
            dim=-1,
        )

        # Takes the log-probabilities
        beam_log_probs = log_probs_clone[
            torch.arange(batch_size).unsqueeze(1), candidates
        ].reshape(batch_size * self.beam_size)
        alived_log_probs = torch.cat(
            [
                torch.index_select(
                    alived_log_probs, dim=0, index=predecessors
                ),
                beam_log_probs.unsqueeze(1),
            ],
            dim=-1,
        )

        is_eos = self._update_hyp_and_scores(
            inp_tokens,
            alived_seq,
            alived_log_probs,
            hyps_and_scores,
            scores,
            timesteps=t,
        )

        # Block the paths that have reached eos.
        sequence_scores.masked_fill_(is_eos, float("-inf"))

    if not self._check_full_beams(hyps_and_scores, self.beam_size):
        # Using all eos to fill-up the hyps.
        eos = (
            torch.zeros(batch_size * self.beam_size, device=device)
            .fill_(self.eos_index)
            .long()
        )
        _ = self._update_hyp_and_scores(
            eos,
            alived_seq,
            alived_log_probs,
            hyps_and_scores,
            scores,
            timesteps=max_decode_steps,
        )

    (
        topk_hyps,
        topk_scores,
        topk_lengths,
        log_probs,
    ) = self._get_top_score_prediction(hyps_and_scores, topk=self.topk,)
    # pick the best hyp
    predictions = topk_hyps[:, 0, :]
    predictions = batch_filter_seq2seq_output(
        predictions, eos_id=self.eos_index
    )

    if self.return_log_probs:
        return predictions, topk_scores, log_probs
    else:
        return predictions, topk_scores


def inflate_tensor(tensor, times, dim):
    """This function inflates the tensor for times along dim.

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor to be inflated.
    times : int
        The tensor will inflate for this number of times.
    dim : int
        The dim to be inflated.

    Returns
    -------
    torch.Tensor
        The inflated tensor.

    Example
    -------
    >>> tensor = torch.Tensor([[1,2,3], [4,5,6]])
    >>> new_tensor = inflate_tensor(tensor, 2, dim=0)
    >>> new_tensor
    tensor([[1., 2., 3.],
            [1., 2., 3.],
            [4., 5., 6.],
            [4., 5., 6.]])
    """
    return torch.repeat_interleave(tensor, times, dim=dim)

def batch_filter_seq2seq_output(prediction, eos_id=-1):
    """Calling batch_size times of filter_seq2seq_output.

    Arguments
    ---------
    prediction : list of torch.Tensor
        A list containing the output ints predicted by the seq2seq system.
    eos_id : int, string
        The id of the eos.

    Returns
    ------
    list
        The output predicted by seq2seq model.

    Example
    -------
    >>> predictions = [torch.IntTensor([1,2,3,4]), torch.IntTensor([2,3,4,5,6])]
    >>> predictions = batch_filter_seq2seq_output(predictions, eos_id=4)
    >>> predictions
    [[1, 2, 3], [2, 3]]
    """
    outputs = []
    for p in prediction:
        res = filter_seq2seq_output(p.tolist(), eos_id=eos_id)
        outputs.append(res)
    return outputs

def filter_seq2seq_output(string_pred, eos_id=-1):
    """Filter the output until the first eos occurs (exclusive).

    Arguments
    ---------
    string_pred : list
        A list containing the output strings/ints predicted by the seq2seq system.
    eos_id : int, string
        The id of the eos.

    Returns
    ------
    list
        The output predicted by seq2seq model.

    Example
    -------
    >>> string_pred = ['a','b','c','d','eos','e']
    >>> string_out = filter_seq2seq_output(string_pred, eos_id='eos')
    >>> string_out
    ['a', 'b', 'c', 'd']
    """
    if isinstance(string_pred, list):
        try:
            eos_index = next(
                i for i, v in enumerate(string_pred) if v == eos_id
            )
        except StopIteration:
            eos_index = len(string_pred)
        string_out = string_pred[:eos_index]
    else:
        raise ValueError("The input must be a list.")
    return string_out