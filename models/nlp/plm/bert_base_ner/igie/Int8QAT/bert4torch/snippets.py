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

import collections
import gc
import inspect
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
import warnings

import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset

is_py2 = six.PY2

if not is_py2:
    basestring = str


def take_along_dim(input_tensor, indices, dim=None):
    if torch.__version__ >= "1.9.0":
        return torch.take_along_dim(input_tensor, indices, dim)
    else:
        if dim is None:
            res = input_tensor.flatten()[indices]
        else:
            res = np.take_along_axis(
                input_tensor.cpu().numpy(), indices.cpu().numpy(), axis=dim
            )
            res = torch.from_numpy(res).to(input_tensor.device)
        return res


def is_string(s):
    return isinstance(s, basestring)


def truncate_sequences(maxlen, indices, *sequences):
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


def text_segmentate(text, maxlen, seps="\n", strips=None, truncate=True):
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = "", []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips, truncate))
                text = ""
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips, truncate))
        return texts
    elif truncate and (not seps) and (len(text) > maxlen):
        return [
            text[i * maxlen : (i + 1) * maxlen]
            for i in range(0, int(np.ceil(len(text) / maxlen)))
        ]
    else:
        return [text]


def merge_segmentate(sequences, maxlen, sep=""):
    sequences_new = []
    text = ""
    for t in sequences:
        if text and len(text + sep + t) <= maxlen:
            text = text + sep + t
        elif text:
            sequences_new.append(text)
            text = t
        elif len(t) < maxlen:
            text = t
        else:
            sequences_new.append(t)
            text = ""
    if text:
        sequences_new.append(text)
    return sequences_new


def text_augmentation(
    texts,
    noise_dict=None,
    noise_len=0,
    noise_p=0.0,
    skip_words=None,
    strategy="random",
    allow_dup=True,
):
    def insert(text, insert_idx, noise_dict):
        text = list(text)
        for i in insert_idx:
            text[i] = text[i] + random.choice(noise_dict)
        return "".join(text)

    def delete(text, delete_idx):
        text = list(text)
        for i in delete_idx:
            text[i] = ""
        return "".join(text)

    def replace(text, replace_idx, noise_dict):
        text = list(text)
        for i in replace_idx:
            text[i] = random.choice(noise_dict)
        return "".join(text)

    def search(pattern, sequence, keep_last=True):
        n = len(pattern)
        pattern_idx_set = set()
        for i in range(len(sequence)):
            if sequence[i : i + n] == pattern:
                pattern_idx_set = (
                    pattern_idx_set.union(set(range(i, i + n)))
                    if keep_last
                    else pattern_idx_set.union(set(range(i, i + n - 1)))
                )
        return pattern_idx_set

    if (noise_len == 0) and (noise_p == 0):
        return texts

    assert strategy in {
        "insert",
        "delete",
        "replace",
        "random",
    }, "EDA strategy only support insert, delete, replace, random"

    if isinstance(texts, str):
        texts = [texts]

    if skip_words is None:
        skip_words = []
    elif isinstance(skip_words, str):
        skip_words = [skip_words]

    for id, text in enumerate(texts):
        sel_len = noise_len if noise_len > 0 else int(len(text) * noise_p)
        skip_idx = set()
        for item in skip_words:
            skip_idx = skip_idx.union(search(item, text, strategy != "insert"))

        sel_idxs = [i for i in range(len(text)) if i not in skip_idx]
        sel_len = (
            sel_len if allow_dup else min(sel_len, len(sel_idxs))
        )
        if (sel_len == 0) or (len(sel_idxs) == 0):
            continue
        sel_idx = np.random.choice(sel_idxs, sel_len, replace=allow_dup)
        if strategy == "insert":
            texts[id] = insert(text, sel_idx, noise_dict)
        elif strategy == "delete":
            texts[id] = delete(text, sel_idx)
        elif strategy == "replace":
            texts[id] = replace(text, sel_idx, noise_dict)
        elif strategy == "random":
            if random.random() < 0.333:
                skip_idx = set()
                for item in skip_words:
                    skip_idx = skip_idx.union(search(item, text, keep_last=False))
                texts[id] = insert(text, sel_idx, noise_dict)
            elif random.random() < 0.667:
                texts[id] = delete(text, sel_idx)
            else:
                texts[id] = replace(text, sel_idx, noise_dict)
    return texts if len(texts) > 1 else texts[0]


def lowercase_and_normalize(text, never_split=()):
    if is_py2:
        text = unicode(text)

    # convert non-special tokens to lowercase
    escaped_special_toks = [re.escape(s_tok) for s_tok in never_split]
    pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
    text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

    text = unicodedata.normalize("NFD", text)
    text = "".join([ch for ch in text if unicodedata.category(ch) != "Mn"])
    return text


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode="post"):
    if isinstance(inputs[0], (np.ndarray, list)):
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, "__getitem__"):
            length = [length]

        slices = [np.s_[: length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == "post":
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == "pre":
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, "constant", constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    elif isinstance(inputs[0], torch.Tensor):
        assert (
            mode == "post"
        ), '"mode" argument must be "post" when element is torch.Tensor'
        if length is not None:
            inputs = [i[:length] for i in inputs]
        return pad_sequence(inputs, padding_value=value, batch_first=True)
    else:
        raise ValueError('"input" argument must be tensor/list/ndarray.')


def insert_arguments(**arguments):
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


def delete_arguments(*arguments):
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k in arguments:
                if k in kwargs:
                    raise TypeError(
                        "%s got an unexpected keyword argument '%s'"
                        % (self.__class__.__name__, k)
                    )
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


class Progbar(object):
    """Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(
        self, target, width=30, verbose=1, interval=0.05, stateful_metrics=None
    ):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = (
            hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        ) or "ipykernel" in sys.modules
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [
                        v * (current - self._seen_so_far),
                        current - self._seen_so_far,
                    ]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += current - self._seen_so_far
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = " - %.0fs" % (now - self._start)
        if self.verbose == 1:
            if (
                now - self._last_update < self.interval
                and self.target is not None
                and current < self.target
            ):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write("\b" * prev_total_width)
                sys.stdout.write("\r")
            else:
                sys.stdout.write("\n")

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = "%%%dd/%d [" % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += "=" * (prog_width - 1)
                    if current < self.target:
                        bar += ">"
                    else:
                        bar += "="
                bar += "." * (self.width - prog_width)
                bar += "]"
            else:
                bar = "%7d/Unknown" % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = "%d:%02d:%02d" % (
                        eta // 3600,
                        (eta % 3600) // 60,
                        eta % 60,
                    )
                elif eta > 60:
                    eta_format = "%d:%02d" % (eta // 60, eta % 60)
                else:
                    eta_format = "%ds" % eta

                info = " - ETA: %s" % eta_format
            else:
                if time_per_unit >= 1:
                    info += " %.0fs/step" % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += " %.0fms/step" % (time_per_unit * 1e3)
                else:
                    info += " %.0fus/step" % (time_per_unit * 1e6)

            for k in self._values:
                info += " - %s:" % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += " %.4f" % avg
                    else:
                        info += " %.4e" % avg
                else:
                    info += " %s" % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += " " * (prev_total_width - self._total_width)

            if self.target is not None and current >= self.target:
                info += "\n"

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += " - %s:" % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += " %.4f" % avg
                    else:
                        info += " %.4e" % avg
                info += "\n"

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


class Callback(object):
    """Callback基类"""

    def __init__(self):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, global_step, epoch, logs=None):
        pass

    def on_epoch_end(self, global_step, epoch, logs=None):
        pass

    def on_batch_begin(self, global_step, batch, logs=None):
        pass

    def on_batch_end(self, global_step, batch, logs=None):
        pass

    def on_dataloader_end(self, logs=None):
        pass


class ProgbarLogger(Callback):
    """Callback that prints metrics to stdout.

    # Arguments
        count_mode: One of "steps" or "samples".
            Whether the progress bar should
            count samples seen or steps (batches) seen.
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is.
            All others will be averaged over time (e.g. loss, etc).

    # Raises
        ValueError: In case of invalid `count_mode`.
    """

    def __init__(self, epochs, steps, metrics, stateful_metrics=None, verbose=1):
        super(ProgbarLogger, self).__init__()
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()
        self.params = {
            "epochs": epochs,
            "steps": steps,
            "verbose": verbose,
            "metrics": metrics,
        }
        self.verbose = verbose
        self.epochs = epochs

    def add_metrics(self, metrics, add_position=None):
        if add_position is None:
            add_position = len(self.params["metrics"])
        if isinstance(metrics, str):
            metrics = [metrics]

        add_metrics = []
        for metric in metrics:
            if metric not in self.params["metrics"]:
                add_metrics.append(metric)
        self.params["metrics"] = (
            self.params["metrics"][:add_position]
            + add_metrics
            + self.params["metrics"][add_position:]
        )

    def on_train_begin(self, logs=None):
        if self.verbose:
            print("Start Training".center(40, "="))

    def on_epoch_begin(self, global_step=None, epoch=None, logs=None):
        if self.verbose:
            print("Epoch %d/%d" % (epoch + 1, self.epochs))
            self.target = self.params["steps"]
            self.progbar = Progbar(
                target=self.target,
                verbose=self.verbose,
                stateful_metrics=self.stateful_metrics,
            )
        self.seen = 0

    def on_batch_begin(self, global_step=None, batch=None, logs=None):
        if self.seen < self.target:
            self.log_values = []

    def on_batch_end(self, global_step=None, batch=None, logs=None):
        logs = logs or {}
        self.seen += 1
        for k in self.params["metrics"]:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and self.seen < self.target:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, global_step=None, epoch=None, logs=None):
        logs = logs or {}
        for k in self.params["metrics"]:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values)

    def on_train_end(self, logs=None):
        if self.verbose:
            print("Finish Training".center(40, "="))


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor="loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
    ):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                "EarlyStopping mode %s is unknown, fallback to auto mode." % mode,
                RuntimeWarning,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.greater if "acc" in self.monitor else np.less
        self.min_delta = (
            self.min_delta if self.monitor_op == np.greater else -self.min_delta
        )

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, steps, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Epoch {self.stopped_epoch+1}: early stopping\n")

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s"
                % (self.monitor, ",".join(list(logs.keys()))),
                RuntimeWarning,
            )
        return monitor_value


def metric_mapping(metric, y_pred, y_true):
    if metric == "accuracy":
        if isinstance(y_pred, (list, tuple)):
            y_pred = y_pred[0]
        y_pred = torch.argmax(y_pred, dim=-1)
        acc = torch.sum(y_pred.eq(y_true)).item() / y_true.size(0)
        return acc
    return None


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)


class AutoRegressiveDecoder(object):
    def __init__(self, start_id, end_id, maxlen, minlen=1, device="cpu"):
        self.start_id = start_id
        self.end_id = end_id
        self.maxlen = maxlen
        self.minlen = minlen
        self.models = {}
        self.device = device
        if start_id is None:
            self.first_output_ids = torch.empty((1, 0), dtype=int, device=device)
        else:
            self.first_output_ids = torch.tensor([[self.start_id]], device=device)

    @staticmethod
    def wraps(default_rtype="probas", use_states=False):
        def actual_decorator(predict):
            def new_predict(
                self, inputs, output_ids, states, temperature=1, rtype=default_rtype
            ):
                assert rtype in ["probas", "logits"]
                prediction = predict(self, inputs, output_ids, states)

                if not use_states:
                    prediction = (prediction, None)

                if default_rtype == "logits":
                    prediction = (
                        nn.Softmax(dim=-1)(prediction[0] / temperature),
                        prediction[1],
                    )
                elif temperature != 1:
                    probas = torch.power(prediction[0], 1.0 / temperature)
                    probas = probas / probas.sum(axis=-1, keepdims=True)
                    prediction = (probas, prediction[1])

                if rtype == "probas":
                    return prediction
                else:
                    return torch.log(prediction[0] + 1e-12), prediction[1]

            return new_predict

        return actual_decorator

    def predict(self, inputs, output_ids, states=None):
        raise NotImplementedError

    def beam_search(
        self, inputs_raw, topk, states=None, temperature=1, min_ends=1, add_btz_dim=True
    ):
        inputs = []
        for i in inputs_raw:
            if isinstance(i, torch.torch.Tensor):
                pass
            elif isinstance(i, (list, tuple, np.ndarray)) and add_btz_dim:
                i = torch.tensor([i], device=self.device)
            elif isinstance(i, (list, tuple, np.ndarray)) and not add_btz_dim:
                i = torch.tensor(i, device=self.device)
            else:
                raise ValueError(
                    "Beam search inputs ele only support tensor、array、list、tuple"
                )
            inputs.append(i)

        output_ids, output_scores = self.first_output_ids, torch.zeros(
            1, device=self.device
        )
        for step in range(self.maxlen):
            scores, states = self.predict(
                inputs, output_ids, states, temperature, "logits"
            )
            if step == 0:
                inputs = [i.repeat([topk] + [1] * (len(i.shape) - 1)) for i in inputs]
            scores = output_scores.reshape((-1, 1)) + scores
            indices = scores.flatten().argsort(dim=-1, descending=True)[:topk] 
            indices_1 = torch.div(
                indices, scores.shape[1], rounding_mode="trunc"
            )
            indices_2 = (indices % scores.shape[1]).reshape((-1, 1))
            output_ids = torch.cat([output_ids[indices_1], indices_2], 1)
            output_scores = take_along_dim(scores, indices, dim=None)
            is_end = output_ids[:, -1] == self.end_id
            end_counts = (output_ids == self.end_id).sum(1)
            if output_ids.shape[1] >= self.minlen:
                best = output_scores.argmax()
                if is_end[best] and end_counts[best] >= min_ends:
                    return output_ids[best]
                else:
                    flag = ~is_end | (end_counts < min_ends)
                    if not flag.all():
                        inputs = [i[flag] for i in inputs]
                        output_ids = output_ids[flag]
                        output_scores = output_scores[flag]
                        end_counts = end_counts[flag]
                        topk = flag.sum()
        return output_ids[output_scores.argmax()]

    def random_sample(
        self, inputs, n, topk=None, topp=None, states=None, temperature=1, min_ends=1
    ):
        inputs = [torch.tensor([i], device=self.device) for i in inputs]
        output_ids = self.first_output_ids
        results = []
        for step in range(self.maxlen):
            probas, states = self.predict(
                inputs, output_ids, states, temperature, "probas"
            )
            probas /= probas.sum(dim=-1, keepdims=True)
            if step == 0:
                probas = probas.repeat([n] + [1] * (len(probas.shape) - 1))
                inputs = [i.repeat([n] + [1] * (len(i.shape) - 1)) for i in inputs]
                output_ids = output_ids.repeat([n] + [1] * (len(output_ids.shape) - 1))
            if topk is not None:
                k_indices = probas.argsort(dim=-1, descending=True)[:, :topk]
                probas = take_along_dim(probas, k_indices, dim=1)
                probas /= probas.sum(dim=1, keepdims=True)
            if topp is not None:
                p_indices = probas.argsort(dim=-1, descending=True)
                probas = take_along_dim(probas, p_indices, dim=-1)
                cumsum_probas = torch.cumsum(probas, dim=-1)
                flag = torch.roll(cumsum_probas >= topp, 1, dims=1)
                flag[:, 0] = False
                probas[flag] = 0
                probas /= probas.sum(dim=1, keepdims=True)

            sample_func = lambda p: torch.multinomial(p, 1)
            sample_ids = torch.stack([sample_func(p) for p in probas])
            sample_ids = sample_ids.reshape((-1, 1))
            if topp is not None:
                sample_ids = take_along_dim(p_indices, sample_ids, dim=1)
            if topk is not None:
                sample_ids = take_along_dim(k_indices, sample_ids, dim=1)
            output_ids = torch.cat([output_ids, sample_ids], 1)
            is_end = output_ids[:, -1] == self.end_id
            end_counts = (output_ids == self.end_id).sum(1)
            if output_ids.shape[1] >= self.minlen:
                flag = is_end & (end_counts >= min_ends)
                if flag.any():
                    for ids in output_ids[flag]:
                        results.append(ids)
                    flag = flag == False
                    inputs = [i[flag] for i in inputs]
                    output_ids = output_ids[flag]
                    end_counts = end_counts[flag]
                    if len(output_ids) == 0:
                        break
        for ids in output_ids:
            results.append(ids)
        return results


def search_layer(model, layer_name, retrun_first=True):
    return_list = []
    for name, param in model.named_parameters():
        if param.requires_grad and layer_name in name:
            return_list.append(param)
    if len(return_list) == 0:
        return None
    if retrun_first:
        return return_list[0]
    else:
        return return_list


class ListDataset(Dataset):
    def __init__(self, file_path=None, data=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError(
                "The input args shall be str format file_path / list format dataset"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load_data(file_path):
        return file_path


class IterDataset(IterableDataset):
    """流式读取文件"""

    def __init__(self, file_path=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.file_path = file_path
        else:
            raise ValueError(
                "The input args shall be str format file_path / list format dataset"
            )

    def __iter__(self):
        return self.load_data(self.file_path)

    @staticmethod
    def load_data(file_path):
        return file_path


# sinusoid编码
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """Returns: [seq_len, d_hid]"""
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid)
    )
    embeddings_table = torch.zeros(n_position, d_hid)
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table

def cal_ts_num(tensor_shape):
    cal_num = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(
                obj
            ):  # or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor = obj
            else:
                continue
            if tensor.is_cuda and tensor.size() == tensor_shape:
                print(tensor.shape)
                cal_num += 1
        except Exception as e:
            print("A trivial exception occured: {}".format(e))
    print(cal_num)


def get_kw(cls, kwargs):
    kwargs_new = {}
    for k in kwargs:
        if k not in set(inspect.getargspec(cls)[0]):
            kwargs_new[k] = kwargs[k]
    return kwargs_new


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings", **kwargs):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="emb", **kwargs):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD:
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(
        self,
        epsilon=1.0,
        alpha=0.3,
        emb_name="word_embeddings",
        is_first_attack=False,
        **kwargs,
    ):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name="emb", **kwargs):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (param.grad is not None):
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (param.grad is not None):
                param.grad = self.grad_backup[name]


class VAT:
    def __init__(
        self,
        model,
        emb_name="word_embeddings",
        noise_var=1e-5,
        noise_gamma=1e-6,
        adv_step_size=1e-3,
        adv_alpha=1,
        norm_type="l2",
        **kwargs,
    ):
        self.model = model
        self.noise_var = noise_var
        self.noise_gamma = noise_gamma
        self.adv_step_size = adv_step_size
        self.adv_alpha = adv_alpha
        self.norm_type = norm_type
        self.embed = None
        for (name, module) in self.model.named_modules():
            if emb_name in name:
                module.register_forward_hook(hook=self.hook)

    def hook(self, module, fea_in, fea_out):
        self.embed = fea_out
        return None

    def forward_(self, train_X, new_embed):
        if isinstance(train_X, (tuple, list)):
            new_train_X = [new_embed] + train_X[1:]
            adv_output = (
                self.model.forward(*new_train_X)
                if self.model.forward.__code__.co_argcount >= 3
                else self.model.forward(new_train_X)
            )
        elif isinstance(train_X, torch.Tensor):
            adv_output = self.model.forward(new_embed)
        return adv_output

    def virtual_adversarial_training(self, train_X, logits):
        noise = self.embed.data.new(self.embed.size()).normal_(0, 1) * self.noise_var
        noise.requires_grad_()
        # x + r
        new_embed = self.embed.data.detach() + noise
        adv_output = self.forward_(train_X, new_embed)  # forward第一次
        adv_logits = (
            adv_output[0] if isinstance(adv_output, (list, tuple)) else adv_output
        )
        adv_loss = self.kl(adv_logits, logits.detach(), reduction="batchmean")
        (delta_grad,) = torch.autograd.grad(adv_loss, noise, only_inputs=True)
        norm = delta_grad.norm()
        if torch.isnan(norm) or torch.isinf(norm):
            return None
        # inner sum
        noise = noise + delta_grad * self.adv_step_size
        # projection
        noise = self.adv_project(noise, norm_type=self.norm_type, eps=self.noise_gamma)
        new_embed = self.embed.data.detach() + noise
        new_embed = new_embed.detach()
        adv_output = self.forward_(train_X, new_embed)
        adv_logits = (
            adv_output[0] if isinstance(adv_output, (list, tuple)) else adv_output
        )
        adv_loss_f = self.kl(adv_logits, logits.detach())
        adv_loss_b = self.kl(logits, adv_logits.detach())
        adv_loss = (adv_loss_f + adv_loss_b) * self.adv_alpha
        return adv_loss

    @staticmethod
    def kl(inputs, targets, reduction="sum"):
        loss = F.kl_div(
            F.log_softmax(inputs, dim=-1),
            F.softmax(targets, dim=-1),
            reduction=reduction,
        )
        return loss

    @staticmethod
    def adv_project(grad, norm_type="inf", eps=1e-6):
        if norm_type == "l2":
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == "l1":
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction


class WebServing(object):
    def __init__(self, host="0.0.0.0", port=8000, server="paste"):

        import bottle

        self.host = host
        self.port = port
        self.server = server
        self.bottle = bottle

    def wraps(self, func, arguments, method="GET"):
        def new_func():
            outputs = {"code": 0, "desc": "succeeded", "data": {}}
            kwargs = {}
            for key, value in arguments.items():
                if method == "GET":
                    result = self.bottle.request.GET.getunicode(key)
                else:
                    result = self.bottle.request.POST.getunicode(key)
                if result is None:
                    if value[1]:
                        outputs["code"] = 1
                        outputs["desc"] = 'lack of "%s" argument' % key
                        return json.dumps(outputs, ensure_ascii=False)
                else:
                    if value[0] is not None:
                        result = value[0](result)
                    kwargs[key] = result
            try:
                outputs["data"] = func(**kwargs)
            except Exception as e:
                outputs["code"] = 2
                outputs["desc"] = str(e)
            return json.dumps(outputs, ensure_ascii=False)

        return new_func

    def route(self, path, func, arguments, method="GET"):
        func = self.wraps(func, arguments, method)
        self.bottle.route(path, method=method)(func)

    def start(self):
        self.bottle.run(host=self.host, port=self.port, server=self.server)


def get_pool_emb(
    hidden_state=None,
    pooler=None,
    attention_mask=None,
    pool_strategy="cls",
    custom_layer=None,
):
    if pool_strategy == "pooler":
        return pooler
    elif pool_strategy == "cls":
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(
            hidden_state, torch.Tensor
        ), f"{pool_strategy} strategy request tensor hidden_state"
        return hidden_state[:, 0]
    elif pool_strategy in {"last-avg", "mean"}:
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(
            hidden_state, torch.Tensor
        ), f"{pool_strategy} pooling strategy request tensor hidden_state"
        hid = torch.sum(hidden_state * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / attention_mask
    elif pool_strategy in {"last-max", "max"}:
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(
            hidden_state, torch.Tensor
        ), f"{pool_strategy} pooling strategy request tensor hidden_state"
        hid = hidden_state * attention_mask[:, :, None]
        return torch.max(hid, dim=1)
    elif pool_strategy == "first-last-avg":
        assert isinstance(
            hidden_state, list
        ), f"{pool_strategy} pooling strategy request list hidden_state"
        hid = torch.sum(hidden_state[1] * attention_mask[:, :, None], dim=1)
        hid += torch.sum(hidden_state[-1] * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / (2 * attention_mask)
    elif pool_strategy == "custom":
        assert isinstance(
            hidden_state, list
        ), f"{pool_strategy} pooling strategy request list hidden_state"
        assert isinstance(
            custom_layer, (int, list, tuple)
        ), f"{pool_strategy} pooling strategy request int/list/tuple custom_layer"
        custom_layer = [custom_layer] if isinstance(custom_layer, int) else custom_layer
        hid = 0
        for i, layer in enumerate(custom_layer, start=1):
            hid += torch.sum(hidden_state[layer] * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / (i * attention_mask)
    else:
        raise ValueError("pool_strategy illegal")


def seed_everything(seed=None):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if (seed is None) or not (min_seed_value <= seed <= max_seed_value):
        random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
