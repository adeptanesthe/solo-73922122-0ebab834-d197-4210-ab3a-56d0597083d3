"""啸声追踪 API：参数校验、起点失败、无效输入不覆盖上次轨迹。"""
import pytest

from conftest import build_whistle_wav, wav_from_samples


CROSS_SIGNAL = dict(
    f0=1000, f1=1500, start=0.2, end=1.4, duration=1.8,
    extra_sines=((0.2, 1.4, 1500, 1000),),
)


def post_track(client, audio: bytes, start_time="0.3", start_freq="1046",
               rate="800", name="whistle.wav"):
    return client.post(
        "/api/track/whistle",
        files={"audio": (name, audio, "audio/wav")},
        data={
            "start_time": start_time,
            "start_freq": start_freq,
            "max_rate_hz_s": rate,
        },
    )


def test_track_empty_result(client):
    assert client.get("/api/track/last").status_code == 404


def test_track_crossing_returns_segments(client):
    audio = build_whistle_wav(**CROSS_SIGNAL)
    resp = post_track(client, audio)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["params"]["start_time"] == pytest.approx(0.3, abs=0.05)
    assert body["params"]["requested_start_freq"] == 1046.0
    assert body["audio"]["duration"] == 1.8
    kinds = [s["kind"] for s in body["segments"]]
    assert "track" in kinds and "ambiguous" in kinds
    assert body["summary"]["ambiguous_seconds"] > 0
    # 轨迹点全部落在 200~2000 Hz
    for seg in body["segments"]:
        for t, f in seg["points"]:
            assert 0.0 <= t <= 1.8
            assert 200.0 <= f <= 2000.0
    # 结果持久化，可再次获取
    assert client.get("/api/track/last").json()["id"] == body["id"]


def test_track_gap_marked_as_missing(client):
    audio = build_whistle_wav(
        f0=1000, f1=1300, start=0.2, end=1.75, duration=1.8,
        gaps=((0.9, 0.98),),
    )
    body = post_track(client, audio, "0.4", "1060", "1000").json()
    gaps = [s for s in body["segments"] if s["kind"] == "gap"]
    assert len(gaps) == 1
    assert gaps[0]["label"] == "缺测"
    assert gaps[0]["points"] == []
    assert body["summary"]["missing_seconds"] > 0


def test_track_long_gap_stops(client):
    audio = build_whistle_wav(
        f0=1000, f1=1200, start=0.2, end=1.75, duration=1.8,
        gaps=((0.9, 1.08),),
    )
    body = post_track(client, audio, "0.4", "1033", "600").json()
    assert body["track"]["stop_reason"] == "gap_too_long"
    assert body["track"]["stop_text"]
    assert body["track"]["stop_time"] < 1.0
    assert any("120" in w for w in body["warnings"])


def test_track_uploaded_audio_served_back(client):
    audio = build_whistle_wav(**CROSS_SIGNAL)
    body = post_track(client, audio).json()
    resp = client.get(body["audio"]["url"])
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


def test_track_invalid_input_keeps_last_result(client):
    good = post_track(client, build_whistle_wav(**CROSS_SIGNAL))
    assert good.status_code == 200
    good_id = good.json()["id"]

    def assert_kept():
        assert client.get("/api/track/last").json()["id"] == good_id

    cases = [
        (dict(start_time="abc"), "数字"),
        (dict(start_time=""), "数字"),
        (dict(start_time="-0.2"), "起点时刻"),
        (dict(start_time="9"), "起点时刻"),
        (dict(start_freq="80"), "近似频率"),
        (dict(start_freq="3000"), "近似频率"),
        (dict(rate="0"), "最大频率变化量"),
        (dict(rate="-5"), "最大频率变化量"),
        (dict(start_freq="99999"), "近似频率"),
    ]
    for overrides, keyword in cases:
        kwargs = dict(start_time="0.3", start_freq="1046", rate="800")
        kwargs.update(overrides)
        resp = post_track(client, build_whistle_wav(**CROSS_SIGNAL), **kwargs)
        assert resp.status_code == 400, (overrides, resp.text)
        assert keyword in resp.json()["detail"], (overrides, resp.json())
        assert_kept()

    # 起点找不到清晰谱峰（频率处没有峰）
    resp = post_track(client, build_whistle_wav(**CROSS_SIGNAL), start_freq="400")
    assert resp.status_code == 400
    assert "找不到清晰谱峰" in resp.json()["detail"]
    assert_kept()

    # 非 WAV
    resp = post_track(client, b"hello", name="notes.txt")
    assert resp.status_code == 400
    assert_kept()


def test_track_invalid_audio_format(client):
    # 8 kHz 采样率
    import numpy as np
    sr = 8000
    n = int(1.8 * sr)
    t = np.arange(n) / sr
    x = 0.3 * np.sin(2 * np.pi * 1200 * t)
    bad = wav_from_samples(x, sr=sr)
    resp = post_track(client, bad)
    assert resp.status_code == 400
    assert "采样率" in resp.json()["detail"]
    assert client.get("/api/track/last").status_code == 404


def test_track_missing_field(client):
    audio = build_whistle_wav(**CROSS_SIGNAL)
    resp = client.post(
        "/api/track/whistle",
        files={"audio": ("w.wav", audio, "audio/wav")},
        data={"start_time": "0.3", "start_freq": "1046"},
    )
    assert resp.status_code == 422  # FastAPI 必填字段缺失
    assert client.get("/api/track/last").status_code == 404


def test_track_does_not_disturb_inspection_result(client):
    """追踪结果与冲次核对结果互不覆盖。"""
    from conftest import build_wav, post_inspect

    inspect = post_inspect(
        client, build_wav([0.5, 1.0, 1.5, 2.0], duration=2.6),
        build_wav([0.5, 1.0, 1.5, 2.0], duration=2.6),
    )
    inspect_id = inspect.json()["id"]
    post_track(client, build_whistle_wav(**CROSS_SIGNAL))
    assert client.get("/api/result/last").json()["id"] == inspect_id


def test_track_clear(client):
    post_track(client, build_whistle_wav(**CROSS_SIGNAL))
    assert client.delete("/api/track/last").json() == {"status": "cleared"}
    assert client.get("/api/track/last").status_code == 404


def test_index_has_whistle_entry(client):
    html = client.get("/").text
    assert "啸声追踪" in html
    assert "/api/track/whistle" in html
