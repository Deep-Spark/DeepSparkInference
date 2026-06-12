#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthesize CV3-Eval local manifests with CosyVoice2 for objective scoring in cv3-eval.

Expected layout under --output-dir (same as decode_dir in run_infer_cv3_eval.sh):
  <subset>/wavs/<utt_id>.wav

Example (from CosyVoice repo root, after conda/pip deps are installed):
  python examples/cv3_eval/infer_cv3_eval.py \\
    --cv3-eval-dir /path/to/CV3-Eval \\
    --model-dir pretrained_models/CosyVoice2-0.5B \\
    --task zero_shot \\
    --subsets zh en \\
    --output-dir /path/to/decode/zero_shot

Then in cv3-eval:
  bash run_infer_cv3_eval.sh --decode_dir /path/to/decode/zero_shot
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "third_party" / "Matcha-TTS") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402


def load_kv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, rest = line.split(maxsplit=1)
            out[utt] = rest
    return out


def resolve_audio_path(cv3_eval_dir: Path, wav_field: str) -> str:
    p = Path(wav_field)
    if p.is_file():
        return str(p.resolve())
    cand = (cv3_eval_dir / wav_field).resolve()
    if cand.is_file():
        return str(cand)
    raise FileNotFoundError(f"prompt wav not found: {wav_field} (tried {cand})")


def concat_infer_outputs(gen) -> torch.Tensor | None:
    parts: list[torch.Tensor] = []
    for out in gen:
        parts.append(out["tts_speech"])
    if not parts:
        return None
    return torch.cat(parts, dim=1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cv3-eval-dir", type=Path, required=True, help="Root of CV3-Eval repo (contains data/)")
    p.add_argument("--model-dir", type=str, required=True, help="CosyVoice2 model dir (cosyvoice2.yaml inside)")
    p.add_argument(
        "--task",
        type=str,
        default="zero_shot",
        choices=("zero_shot", "cross_lingual_zeroshot", "emotion_zeroshot"),
        help="CV3-Eval task name; must match data/<task>/ in the benchmark tree",
    )
    p.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["zh"],
        help='Subset folder names under data/<task>/, e.g. "zh en" or "to_zh to_en"',
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="decode_dir root passed to cv3-eval (will write <output-dir>/<subset>/wavs/*.wav)",
    )
    p.add_argument("--load-jit", action="store_true", help="Load CosyVoice2 TorchScript encoders")
    p.add_argument("--load-trt", action="store_true", help="Load TensorRT flow decoder (requires built engines)")
    p.add_argument("--load-igie", action="store_true", help="Load IGIE flow decoder (requires built engines)")
    p.add_argument("--fp16", action="store_true", help="FP16 inference")
    p.add_argument("--resume", action="store_true", help="Skip utterances whose wav already exists")
    p.add_argument("--max-utts", type=int, default=0, help="If >0, only process first N lines of each subset (debug)")
    return p.parse_args()


def main():
    args = parse_args()
    cv3 = args.cv3_eval_dir.resolve()
    data_task = cv3 / "data" / args.task
    if not data_task.is_dir():
        raise FileNotFoundError(f"missing benchmark dir: {data_task}")

    model = AutoModel(
        model_dir=args.model_dir,
        load_jit=args.load_jit,
        load_trt=args.load_trt,
        load_igie=args.load_igie,
        fp16=args.fp16,
    )
    sr = model.sample_rate

    for subset in args.subsets:
        subdir = data_task / subset
        text_p = subdir / "text"
        prompt_text_p = subdir / "prompt_text"
        prompt_wav_p = subdir / "prompt_wav.scp"
        for req in (text_p, prompt_text_p, prompt_wav_p):
            if not req.is_file():
                raise FileNotFoundError(f"subset {subset}: missing {req}")

        prompt_texts = load_kv_file(prompt_text_p)
        prompt_wavs = load_kv_file(prompt_wav_p)
        out_wav_dir = args.output_dir / subset / "wavs"
        out_wav_dir.mkdir(parents=True, exist_ok=True)

        n_done = 0
        with text_p.open("r", encoding="utf-8") as tf:
            for line_i, line in enumerate(tf):
                if args.max_utts and line_i >= args.max_utts:
                    break
                line = line.strip()
                if not line:
                    continue
                utt, target_text = line.split(maxsplit=1)
                out_path = out_wav_dir / f"{utt}.wav"
                if args.resume and out_path.is_file():
                    continue
                if utt not in prompt_texts or utt not in prompt_wavs:
                    raise KeyError(f"{subset}: utt {utt} missing in prompt_text or prompt_wav.scp")
                prompt_text = prompt_texts[utt]
                prompt_wav = resolve_audio_path(cv3, prompt_wavs[utt])

                if args.task == "cross_lingual_zeroshot":
                    audio = concat_infer_outputs(
                        model.inference_cross_lingual(target_text, prompt_wav, stream=False)
                    )
                else:
                    audio = concat_infer_outputs(
                        model.inference_zero_shot(target_text, prompt_text, prompt_wav, stream=False)
                    )
                if audio is None:
                    print(f"skip {utt}: empty synthesis", flush=True)
                    continue
                torchaudio.save(str(out_path), audio.cpu(), sr)
                n_done += 1
                if n_done % 20 == 0:
                    print(f"{subset}: wrote {n_done} wavs ...", flush=True)
        print(f"{subset}: finished, total new wavs {n_done}", flush=True)


if __name__ == "__main__":
    main()
