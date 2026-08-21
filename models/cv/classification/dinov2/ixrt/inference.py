#!/usr/bin/env python3
"""DINOv2-base on ixRT：性能、精度（对齐 onnxruntime CPU FP32）与下游 kNN 验证。

三种模式：
  perf  纯性能：吞吐 FPS + 单次时延 avg/p50/p99
  acc   数值精度：同一批输入上 ixRT 输出 vs onnxruntime CPU FP32 输出的余弦相似度/误差
  knn   下游可用性：对 ImageFolder 目录提特征，做 leave-one-out 1-NN 分类 top-1
"""

import argparse
import glob
import os
import random
import time

import numpy as np

from ixrt_helper import EngineRunner

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG")


def preprocess(path, imgsz=224, shortest_edge=256):
    """复刻 BitImageProcessor：bicubic 缩放短边 -> center crop -> /255 -> 标准化。"""
    from PIL import Image

    if shortest_edge < imgsz:
        shortest_edge = imgsz
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = shortest_edge / min(width, height)
    image = image.resize((max(imgsz, round(width * scale)), max(imgsz, round(height * scale))), Image.BICUBIC)
    width, height = image.size
    left, top = (width - imgsz) // 2, (height - imgsz) // 2
    image = image.crop((left, top, left + imgsz, top + imgsz))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def list_images(root, limit=0, per_class=0, seed=0):
    """返回 [(path, label)]；root 下有子目录则按 ImageFolder 处理，否则 label 全为 ''。"""
    subdirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    items = []
    if subdirs:
        for subdir in subdirs:
            label = os.path.basename(subdir)
            files = sorted(
                p for p in glob.glob(os.path.join(subdir, "**", "*"), recursive=True) if p.endswith(IMAGE_EXTS)
            )
            if per_class:
                files = files[:per_class]
            items += [(p, label) for p in files]
    else:
        files = sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True) if p.endswith(IMAGE_EXTS))
        items = [(p, "") for p in files]
    if limit and len(items) > limit:
        random.Random(seed).shuffle(items)
        items = items[:limit]
    if not items:
        raise SystemExit(f"{root} 下没找到图片（支持 {IMAGE_EXTS}）")
    return items


def make_batches(items, batch_size, imgsz):
    """按 engine 的固定 batch 切分，最后一批用重复样本补齐并记录有效长度。"""
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        valid = len(chunk)
        data = np.stack([preprocess(path, imgsz) for path, _ in chunk])
        if valid < batch_size:
            pad = np.repeat(data[-1:], batch_size - valid, axis=0)
            data = np.concatenate([data, pad], axis=0)
        yield data, [label for _, label in chunk], valid


def to_embedding(last_hidden_state):
    """DINOv2 常用表征：CLS token 拼 patch token 均值 -> 1536 维。"""
    hidden = last_hidden_state.astype(np.float32)
    cls_token = hidden[:, 0, :]
    patch_mean = hidden[:, 1:, :].mean(axis=1)
    return np.concatenate([cls_token, patch_mean], axis=1)


def l2_normalize(matrix):
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def cosine_rows(a, b):
    a, b = a.reshape(a.shape[0], -1).astype(np.float64), b.reshape(b.shape[0], -1).astype(np.float64)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-12)


class OrtRunner:
    def __init__(self, onnx_path, threads=0):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.batch_size = shape[0] if isinstance(shape[0], int) else 1

    def infer(self, data):
        return self.session.run(None, {self.input_name: data.astype(np.float32)})


def run_perf(args):
    runner = EngineRunner(args.engine, device=args.device)
    batch_size = runner.batch_size
    rng = np.random.default_rng(args.seed)
    data = rng.standard_normal((batch_size, 3, args.imgsz, args.imgsz), dtype=np.float32)
    runner.set_input(data)

    print(f"\n[perf] batch_size={batch_size} warmup={args.warmup} iters={args.iters}")
    for _ in range(args.warmup):
        runner.run()
    runner.sync()

    latencies = []
    start = time.perf_counter()
    for _ in range(args.iters):
        iter_start = time.perf_counter()
        runner.run()
        runner.sync()
        latencies.append((time.perf_counter() - iter_start) * 1000)
    elapsed = time.perf_counter() - start

    latencies = np.array(latencies)
    fps = batch_size * args.iters / elapsed
    print(f"FPS               : {fps:.2f} images/s")
    print(f"Latency avg       : {latencies.mean():.3f} ms / batch({batch_size})")
    print(f"Latency p50 / p99 : {np.percentile(latencies, 50):.3f} / {np.percentile(latencies, 99):.3f} ms")
    print(f"Per-image latency : {latencies.mean() / batch_size:.3f} ms")
    runner.close()
    if args.fps_target > 0:
        ok = fps >= args.fps_target
        print(f"Performance Check : {fps:.2f} >= target {args.fps_target} -> {'pass!' if ok else 'failed!'}")
        return 0 if ok else 1
    return 0


def run_acc(args):
    if not args.onnx:
        raise SystemExit("acc 模式需要 --onnx 指向 build_engine.py 产出的静态 ONNX（FP32 参考）")
    runner = EngineRunner(args.engine, device=args.device)
    batch_size = runner.batch_size
    reference = OrtRunner(args.onnx, threads=args.threads)

    if args.images:
        items = list_images(args.images, limit=args.limit or batch_size * 2, per_class=args.per_class, seed=args.seed)
        batches = list(make_batches(items, batch_size, args.imgsz))
        source = f"真实图片 {args.images}（{len(items)} 张）"
    else:
        rng = np.random.default_rng(args.seed)
        batches = [
            (rng.standard_normal((batch_size, 3, args.imgsz, args.imgsz), dtype=np.float32), [], batch_size)
            for _ in range(2)
        ]
        source = "随机输入（未提供 --images）"

    print(f"\n[acc] 参考 = onnxruntime CPU FP32，被测 = ixRT engine；输入来源：{source}")
    hidden_cos, embed_cos, abs_errs, rel_errs = [], [], [], []
    for data, _, valid in batches:
        ixrt_out = runner.infer(data)[0][:valid]
        ort_out = reference.infer(data)[0][:valid]
        hidden_cos += cosine_rows(ixrt_out, ort_out).tolist()
        embed_cos += cosine_rows(to_embedding(ixrt_out), to_embedding(ort_out)).tolist()
        diff = np.abs(ixrt_out.astype(np.float64) - ort_out.astype(np.float64))
        abs_errs.append(diff.max())
        rel_errs.append((diff / np.maximum(np.abs(ort_out.astype(np.float64)), 1e-6)).mean())

    hidden_cos, embed_cos = np.array(hidden_cos), np.array(embed_cos)
    print(f"样本数                        : {len(hidden_cos)}")
    print(f"last_hidden_state 余弦相似度  : min={hidden_cos.min():.6f} mean={hidden_cos.mean():.6f}")
    print(f"1536 维检索特征余弦相似度     : min={embed_cos.min():.6f} mean={embed_cos.mean():.6f}")
    print(f"最大绝对误差                  : {max(abs_errs):.5f}")
    print(f"平均相对误差                  : {np.mean(rel_errs):.5f}")
    ok = embed_cos.min() >= args.cos_target
    print(f"Accuracy Check : min cosine {embed_cos.min():.6f} >= target {args.cos_target} -> {'pass!' if ok else 'failed!'}")
    runner.close()
    return 0 if ok else 1


def leave_one_out_top1(features, labels):
    features = l2_normalize(features)
    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)
    predictions = np.array(labels)[similarity.argmax(axis=1)]
    return float((predictions == np.array(labels)).mean())


def run_knn(args):
    if not args.images:
        raise SystemExit("knn 模式需要 --images 指向 ImageFolder 结构目录（每个子目录一个类别）")
    items = list_images(args.images, limit=args.limit, per_class=args.per_class, seed=args.seed)
    labels = [label for _, label in items]
    if len(set(labels)) < 2:
        raise SystemExit("kNN 需要至少 2 个类别子目录")

    runner = EngineRunner(args.engine, device=args.device)
    batch_size = runner.batch_size
    print(f"\n[knn] {len(items)} 张图 / {len(set(labels))} 类，batch_size={batch_size}")

    batches = list(make_batches(items, batch_size, args.imgsz))
    ixrt_feats = []
    start = time.perf_counter()
    for data, _, valid in batches:
        ixrt_feats.append(to_embedding(runner.infer(data)[0])[:valid])
    ixrt_elapsed = time.perf_counter() - start
    ixrt_feats = np.concatenate(ixrt_feats)

    ixrt_top1 = leave_one_out_top1(ixrt_feats, labels)
    print(f"ixRT   leave-one-out 1-NN top-1 : {ixrt_top1:.4f}  （提特征耗时 {ixrt_elapsed:.1f}s）")

    if args.onnx:
        reference = OrtRunner(args.onnx, threads=args.threads)
        ort_feats = []
        start = time.perf_counter()
        for data, _, valid in batches:
            ort_feats.append(to_embedding(reference.infer(data)[0])[:valid])
        ort_elapsed = time.perf_counter() - start
        ort_feats = np.concatenate(ort_feats)
        ort_top1 = leave_one_out_top1(ort_feats, labels)
        print(f"ORT    leave-one-out 1-NN top-1 : {ort_top1:.4f}  （提特征耗时 {ort_elapsed:.1f}s）")
        print(f"top-1 差值                      : {ixrt_top1 - ort_top1:+.4f}")
        print(f"加速比（CPU FP32 / ixRT）        : {ort_elapsed / max(ixrt_elapsed, 1e-9):.1f}x")

    if args.save_features:
        np.savez(args.save_features, features=ixrt_feats, labels=np.array(labels))
        print(f"特征已保存: {args.save_features}")
    runner.close()
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv2-base ixRT 测试")
    parser.add_argument("--mode", choices=["perf", "acc", "knn"], default="perf")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--onnx", default="", help="静态 FP32 ONNX，用作 CPU 参考")
    parser.add_argument("--images", default="", help="图片目录（ImageFolder 结构可做 kNN）")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="最多取多少张图，0 = 不限")
    parser.add_argument("--per-class", type=int, default=0, help="每类最多取多少张")
    parser.add_argument("--threads", type=int, default=0, help="onnxruntime CPU 线程数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps-target", type=float, default=-1.0)
    parser.add_argument("--cos-target", type=float, default=0.999)
    parser.add_argument("--save-features", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    return {"perf": run_perf, "acc": run_acc, "knn": run_knn}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
