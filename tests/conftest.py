"""合成测试 WAV 与 TestClient 夹具。"""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.service import Store

SR = 16_000


def build_wav(
    impact_times: list[float],
    duration: float | None = None,
    sr: int = SR,
    gain: float = 0.8,
    burst_ms: int = 12,
    echoes: dict[float, float] | None = None,
    echo_gain: float = 0.7,
    noise_floor: float = 0.0,
    sample_width: int = 2,
    channels: int = 1,
    seed: int = 7,
) -> bytes:
    """生成带若干冲击噪声脉冲的 PCM WAV。

    echoes: {主冲击时刻: 回声延迟秒}，回声幅度约为主冲击的 40%。
    """
    rng = np.random.default_rng(seed)
    if duration is None:
        duration = max(impact_times) + 0.4 if impact_times else 1.0
    n = int(round(duration * sr))
    x = rng.standard_normal(n).astype(np.float32) * noise_floor
    burst_n = int(burst_ms / 1000 * sr)
    decay = np.exp(-np.linspace(0, 5, burst_n)).astype(np.float32)

    def put(t: float, amp: float) -> None:
        start = int(round(t * sr))
        end = min(n, start + burst_n)
        if start >= n or end <= 0:
            return
        seg = rng.standard_normal(end - start).astype(np.float32)
        x[start:end] += amp * seg * decay[: end - start]

    for t in impact_times:
        put(t, gain)
    for t, delay in (echoes or {}).items():
        put(t + delay, gain * echo_gain)

    x = np.clip(x, -1.0, 1.0)

    if sample_width == 2:
        pcm = (x * 32767).astype("<i2").tobytes()
        sampwidth = 2
    else:  # 8-bit unsigned PCM
        pcm = ((x + 1.0) * 127.5).clip(0, 255).astype(np.uint8).tobytes()
        sampwidth = 1
    if channels == 2:
        pcm = b"".join(bytes([b, b]) for b in pcm) if sampwidth == 1 else b""
        if sampwidth == 2:
            stereo = np.repeat((x * 32767).astype("<i2"), 2)
            pcm = stereo.tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.store", Store(tmp_path))
    return TestClient(app)


def post_inspect(client: TestClient, ref: bytes, test: bytes):
    return client.post(
        "/api/inspect",
        files={
            "reference": ("ref.wav", ref, "audio/wav"),
            "test": ("test.wav", test, "audio/wav"),
        },
    )


def whistle_signal(
    f0: float,
    f1: float | None = None,
    start: float = 0.0,
    end: float = 1.0,
    duration: float = 2.0,
    sr: int = SR,
    amp: float = 0.3,
    gaps: tuple[tuple[float, float], ...] = (),
    bursts: tuple[tuple[float, float, float], ...] = (),
    noise_floor: float = 0.0,
    extra_sines: tuple[tuple[float, float, float, float], ...] = (),
    seed: int = 11,
) -> np.ndarray:
    """单声道浮点波形：线性扫频的窄带啸声 + 可选静音缺口/宽带冲击/叠加正弦。

    extra_sines: (起始时刻, 结束时刻, 起始频率, 结束频率)
    bursts: (时刻, 时长秒, 幅度) 宽带噪声冲击，用来盖住啸声。
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * sr))
    t = np.arange(n, dtype=np.float64) / sr
    x = rng.standard_normal(n) * noise_floor
    f1 = f0 if f1 is None else f1
    def add_glide(a: float, b: float, fa: float, fb: float, gain: float) -> None:
        mask = ((t >= a) & (t <= b)).astype(np.float64)
        # 5 ms 余弦渐变，避免起止形成宽带咔嗒声
        ramp = max(1, int(0.005 * sr))
        kern = np.hanning(2 * ramp + 1)
        kern /= kern.sum()
        mask = np.convolve(mask, kern, mode="same")
        u = np.clip(t - a, 0.0, None)
        phase = 2 * np.pi * (fa * u + (fb - fa) * u * u / (2 * max(1e-6, b - a)))
        x[:] += gain * mask * np.sin(phase)

    add_glide(start, end, f0, f1, amp)
    for a, b, fa, fb in extra_sines:
        add_glide(a, b, fa, fb, amp)
    for a, b in gaps:
        mask = (t >= a) & (t < b)
        x[mask] = 0.0
    for time, dur, burst_amp in bursts:
        s = int(round(time * sr))
        e = min(n, s + int(round(dur * sr)))
        if s < n:
            x[s:e] += burst_amp * rng.standard_normal(e - s)
    return np.clip(x, -1.0, 1.0)


def wav_from_samples(x: np.ndarray, sr: int = SR) -> bytes:
    """float 波形 -> 16-bit PCM 单声道 WAV。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((x * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def build_whistle_wav(*args, **kwargs) -> bytes:
    sr = kwargs.pop("sr", SR)
    return wav_from_samples(whistle_signal(*args, sr=sr, **kwargs), sr=sr)

