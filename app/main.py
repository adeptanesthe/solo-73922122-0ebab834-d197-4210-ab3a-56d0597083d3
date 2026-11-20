"""FastAPI 入口：上传参考/待检 WAV，返回冲次核对结果；送料啸声追踪入口。"""
from __future__ import annotations

import math
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .audio import AudioValidationError, MAX_DURATION_SEC, SAMPLE_RATE
from .service import MAX_UPLOAD_BYTES, Store
from .tracking import TrackValidationError

DATA_DIR = Path(os.environ.get("APP_DATA_DIR", "/workspace/data"))
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="冲压机冲次声音核对", version="1.1.0")
store = Store(DATA_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/data/uploads/{name}")
def get_upload(name: str) -> FileResponse:
    # 动态读取 store 的上传目录，更换 DATA_DIR 或测试夹具后仍可回传录音
    if "/" in name or ".." in name or not name.endswith(".wav"):
        raise HTTPException(status_code=400, detail="非法文件名")
    path = store.uploads_dir / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="录音不存在")
    return FileResponse(path, media_type="audio/wav")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def _read_wav(file: UploadFile, label: str) -> bytes:
    if not (file.filename or "").lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail=f"{label}必须是 .wav 文件")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{label}文件为空")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label}文件过大；仅支持 {SAMPLE_RATE}Hz/16bit/单声道、"
                f"不超过 {MAX_DURATION_SEC:.0f} 秒的 WAV"
            ),
        )
    return raw


def _parse_number(text: str, label: str) -> float:
    try:
        value = float(text.strip())
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400, detail=f"{label}必须是数字，当前填写：{text!r}"
        ) from None
    if math.isnan(value) or math.isinf(value):
        raise HTTPException(status_code=400, detail=f"{label}必须是有限数值")
    return value


@app.get("/api/result/last")
def last_result() -> dict:
    result = store.get_last()
    if result is None:
        raise HTTPException(status_code=404, detail="暂无有效检查结果")
    return result


@app.delete("/api/result/last")
def clear_result() -> dict[str, str]:
    store.clear()
    return {"status": "cleared"}


@app.post("/api/inspect")
async def inspect(
    reference: UploadFile = File(..., description="正常循环参考 WAV"),
    test: UploadFile = File(..., description="待检循环 WAV"),
) -> dict:
    ref_raw = await _read_wav(reference, "参考录音")
    test_raw = await _read_wav(test, "待检录音")

    try:
        # 任何 AudioValidationError 都会在 store 写入前抛出，
        # 因此无效音频不会覆盖上一次有效结果
        return store.inspect(ref_raw, test_raw)
    except AudioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/track/last")
def last_track() -> dict:
    result = store.get_last_track()
    if result is None:
        raise HTTPException(status_code=404, detail="暂无啸声追踪结果")
    return result


@app.delete("/api/track/last")
def clear_track() -> dict[str, str]:
    store.clear_track()
    return {"status": "cleared"}


@app.post("/api/track/whistle")
async def track_whistle(
    audio: UploadFile = File(..., description="含送料啸声的 WAV"),
    start_time: str = Form(..., description="啸声清楚处的时刻（秒）"),
    start_freq: str = Form(..., description="该时刻的近似频率（Hz）"),
    max_rate_hz_s: str = Form(..., description="每秒最大频率变化量（Hz/s）"),
) -> dict:
    raw = await _read_wav(audio, "录音")
    t0 = _parse_number(start_time, "起点时刻")
    f0 = _parse_number(start_freq, "近似频率")
    rate = _parse_number(max_rate_hz_s, "每秒最大频率变化量")
    try:
        # 参数/起点不合法时在写入前抛出，上一次轨迹保留
        return store.track_whistle(raw, t0, f0, rate)
    except (AudioValidationError, TrackValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
