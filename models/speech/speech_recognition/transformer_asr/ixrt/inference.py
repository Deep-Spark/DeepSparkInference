#!/usr/bin/env/python3
"""

AISHELL-1 transformer model recipe. (Adapted from the LibriSpeech recipe.)

"""
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
import sys
import time
import torch
import logging
import speechbrain as sb
from speechbrain import Stage
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.checkpoints import Checkpointer
import numpy as np
from speechbrain.utils import data_utils
import tensorrt
from torch.utils.data import DataLoader
from tqdm import tqdm
import convert
import beam_search
from load_ixrt_plugin import load_ixrt_plugin
from tensorrt import Dims
from speechbrain.lobes.models.transformer import Transformer
Transformer.get_lookahead_mask = convert.patch_get_lookahead_mask
load_ixrt_plugin()
logger = logging.getLogger(__name__)


def volume(shape):
    result = 1
    for i in shape:
        result *= i
    return result


class ASR(sb.core.Brain):
    def __init__(self, engine_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self.forward_time = 0
        # ixrt
        self.logger = tensorrt.Logger(tensorrt.Logger.ERROR)
        with open(engine_path, "rb") as f, tensorrt.Runtime(self.logger) as self.runtime:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            assert self.engine
            self.context = self.engine.create_execution_context()
            assert self.context
        self.encoder_ln_out = torch.zeros((64,2048,256), dtype=torch.float16).cuda()
        self.infer_time = 0
        self.hparams.valid_search.return_log_probs = True
        self.modules.CNN = self.modules.CNN.half()
        self.hparams.valid_search = self.hparams.valid_search.half()
        self.hparams.valid_search.model = self.hparams.valid_search.model.half()
        self.hparams.valid_search.fc = self.hparams.valid_search.fc.half()
        self.hparams.valid_search.ctc_fc = self.hparams.valid_search.ctc_fc.half()
        self.hparams.valid_search.minus_inf = -10000
        self.hparams.valid_search.softmax = self.hparams.valid_search.softmax.half()
        self.hparams.valid_search.model.decoder = convert.convert_decoder_model(self.hparams.valid_search.model.decoder)
    # Given all input/output bindings, run in a dynamic shape way
    def ixrt_infer(self, engine, context, bindings):
        assert engine.num_bindings == len(bindings)
        io_buffers = [0] * engine.num_bindings
        for name, arr in bindings.items():
            idx = engine.get_binding_index(name)
            io_buffers[idx] = arr.data_ptr()
            # dynamic input
            if engine.binding_is_input(idx):
                context.set_binding_shape(idx, Dims(arr.shape))

        forward_start_time = time.time()
        assert context.execute_v2(io_buffers)

        torch.cuda.synchronize()
        self.forward_time += time.time() - forward_start_time
        outputs = {}
        for name, arr in bindings.items():
            idx = engine.get_binding_index(name)
            if not engine.binding_is_input(idx):
                # dynamic output
                shape = context.get_binding_shape(idx)
                outputs[name] = arr.view(-1)[:volume(shape)].view(*shape)
        return outputs

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

        torch.cuda.synchronize()
        start_time = time.time()

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        # forward modules
        src = self.modules.CNN(feats.half())

        # Orignal PyTorch implementation, comment this to compare
        # enc_out, _ = self.modules.Transformer(
        #     src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        # )
        # logits = self.modules.ctc_lin(enc_out)
        # p_ctc = self.hparams.log_softmax(logits)
        # hyps, _  = self.hparams.test_search(
        #         enc_out.detach(), wav_lens
        # )
        # return p_ctc, wav_lens, hyps

        # transformer
        if src.ndim == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        # ixrt inference
        t1 = time.time()
        bindings = {"input": src.half(), "length_radio": wav_lens.half(),
                    "encoder_ln_out": self.encoder_ln_out}

        infer_result = self.ixrt_infer(self.engine, self.context, bindings)
        encoder_ln_out = infer_result["encoder_ln_out"]
        t2 = time.time()

        hyps, _, p_ctc = beam_search.forward(self.hparams.valid_search, encoder_ln_out.half(), wav_lens.half())
        torch.cuda.synchronize()
        infer_time = time.time() - start_time

        self.infer_time += infer_time

        return p_ctc, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        # (
        #     p_ctc,
        #     p_seq,
        #     wav_lens,
        #     hyps,
        # ) = predictions

        # 去除 seq2seq log-probabilities
        (
            p_ctc,
            wav_lens,
            hyps,
        ) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0)
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        # 去除 seq2seq 部分 loss
        # loss = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            if current_epoch % valid_search_interval == 0 or (stage == sb.Stage.TEST):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                if self.hparams.remove_spaces:
                    predicted_words = ["".join(p) for p in predicted_words]
                    target_words = ["".join(t) for t in target_words]
                    self.cer_metric.append(ids, predicted_words, target_words)

            # 不计算 acc 部分
            # # compute the accuracy of the one-step-forward prediction
            # self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return -torch.ones([1])

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        self.check_and_reset_optimizer()

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # anneal lr every update
            self.hparams.noam_annealing(self.optimizer)

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            # self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.cer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"forward time": self.forward_time}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            # stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or stage == sb.Stage.TEST:
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.stage_one_epochs:
                lr = self.hparams.noam_annealing.current_lr
                steps = self.hparams.noam_annealing.n_steps
                optimizer = self.optimizer.__class__.__name__
            else:
                lr = self.hparams.lr_sgd
                steps = -1
                optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=10,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.cer_file, "w") as w:
                self.cer_metric.write_stats(w)

    def check_and_reset_optimizer(self):
        """reset the optimizer if training enters stage 2"""
        current_epoch = self.hparams.epoch_counter.current
        if not hasattr(self, "switched"):
            self.switched = False
            if isinstance(self.optimizer, torch.optim.SGD):
                self.switched = True

        if self.switched is True:
            return

        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            self.switched = True

    def on_fit_start(self):
        """Initialize the right optimizer on the training start"""
        super().on_fit_start()

        # if the model is resumed from stage two, reinitialize the optimizer
        current_epoch = self.hparams.epoch_counter.current
        current_optimizer = self.optimizer
        if current_epoch > self.hparams.stage_one_epochs:
            del self.optimizer
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            # Load latest checkpoint to resume training if interrupted
            if self.checkpointer is not None:

                # do not reload the weights if training is interrupted right before stage 2
                group = current_optimizer.param_groups[0]
                if "momentum" not in group:
                    return

                self.checkpointer.recover_if_possible(
                    device=torch.device(self.device))

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key)
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

    def evaluate(
            self,
            test_set,
            max_key=None,
            min_key=None,
            progressbar=None,
            test_loader_kwargs={},
    ):
        self.debug = False
        self.debug_batches = 1
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
                isinstance(test_set, DataLoader)
                or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        self.step = 0
        with torch.no_grad():
            for batch in tqdm(
                    test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            run_on_main(
                self.on_stage_end, args=[Stage.TEST, avg_test_loss, None]
            )
        self.step = 0
        return avg_test_loss


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True)
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # Defining tokenizer and loading it
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

    return (
        train_data,
        valid_data,
        test_data,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # 1.  # Dataset prep (parsing Librispeech)
    from aishell_prepare import prepare_aishell  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_aishell,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_data,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    hparams["pretrainer"].collect_files(default_source=hparams['ckpt_path'])
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        engine_path=hparams['engine_path']
    )

    asr_brain.tokenizer = tokenizer

    # Changing the samplers if dynamic batching is activated
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }
    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # evaluation
    print("*** start evaluation ***")
    start_time = time.time()
    asr_brain.evaluate(
        test_data, test_loader_kwargs=hparams["test_dataloader_opts"])
    eval_time = asr_brain.infer_time

    ## 统计数据总音频时长
    duration = 0.0
    for value in test_data.data.values():
        duration = duration + value['duration']
    num_samples = len(test_data)
    print(f"samples: {num_samples}, QPS: {num_samples / eval_time} ")
    print(f"infer time :{eval_time},RTF: {eval_time / duration} ")
