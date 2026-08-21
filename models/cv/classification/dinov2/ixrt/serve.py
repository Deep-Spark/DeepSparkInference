#!/usr/bin/env python3
"""把 DINOv2 ixRT engine 包成一个图像特征提取 HTTP 服务。

    POST /embed       multipart 上传若干图片，返回 1536 维（CLS + patch mean）特征
    POST /similarity  上传两张图片，返回余弦相似度
    GET  /health      返回 engine 元信息

engine 是固定 batch 的，请求内会自动切批并补齐；GPU 推理用锁串行化，
所以只用单 worker 启动（uvicorn 默认即单进程）。
"""

import argparse
import io
import threading

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from inference import IMAGENET_MEAN, IMAGENET_STD, l2_normalize, to_embedding
from ixrt_helper import EngineRunner

app = FastAPI(title="DINOv2-base ixRT feature extractor")
STATE = {}
LOCK = threading.Lock()


def preprocess_bytes(raw, imgsz):
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    shortest_edge = max(imgsz, round(imgsz * 256 / 224))
    width, height = image.size
    scale = shortest_edge / min(width, height)
    image = image.resize((max(imgsz, round(width * scale)), max(imgsz, round(height * scale))), Image.BICUBIC)
    width, height = image.size
    left, top = (width - imgsz) // 2, (height - imgsz) // 2
    image = image.crop((left, top, left + imgsz, top + imgsz))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def embed_arrays(arrays):
    runner, batch_size = STATE["runner"], STATE["batch_size"]
    features = []
    with LOCK:
        for start in range(0, len(arrays), batch_size):
            chunk = arrays[start : start + batch_size]
            valid = len(chunk)
            data = np.stack(chunk)
            if valid < batch_size:
                data = np.concatenate([data, np.repeat(data[-1:], batch_size - valid, axis=0)], axis=0)
            features.append(to_embedding(runner.infer(data)[0])[:valid])
    return np.concatenate(features)


async def read_images(files, imgsz):
    arrays = []
    for upload in files:
        raw = await upload.read()
        try:
            arrays.append(preprocess_bytes(raw, imgsz))
        except Exception as exc:  # noqa: BLE001 - 用户上传的任意文件
            raise HTTPException(status_code=400, detail=f"{upload.filename} 解码失败: {exc}") from exc
    return arrays


@app.get("/health")
def health():
    return {"status": "ok", "engine": STATE["engine_path"], "batch_size": STATE["batch_size"], "imgsz": STATE["imgsz"]}


@app.post("/embed")
async def embed(files: list[UploadFile] = File(...), normalize: bool = True):
    arrays = await read_images(files, STATE["imgsz"])
    features = embed_arrays(arrays)
    if normalize:
        features = l2_normalize(features)
    return {
        "dim": int(features.shape[1]),
        "count": int(features.shape[0]),
        "filenames": [f.filename for f in files],
        "embeddings": features.tolist(),
    }


@app.post("/similarity")
async def similarity(files: list[UploadFile] = File(...)):
    if len(files) != 2:
        raise HTTPException(status_code=400, detail="需要正好 2 张图片")
    features = l2_normalize(embed_arrays(await read_images(files, STATE["imgsz"])))
    return {
        "filenames": [f.filename for f in files],
        "cosine_similarity": float(features[0] @ features[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    runner = EngineRunner(args.engine, device=args.device)
    STATE.update(
        runner=runner,
        batch_size=runner.batch_size,
        engine_path=args.engine,
        imgsz=args.imgsz,
    )
    print(f"[serve] engine batch_size={runner.batch_size} imgsz={args.imgsz} -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
