"""啸声追踪：谱峰检测、交叉、短中断续接与停止规则。"""
import pytest

from app.tracking import TrackValidationError, track_whistle
from conftest import whistle_signal


def _kinds(result):
    return [s.kind for s in result.segments]


def _points(result):
    return [
        (t, f) for s in result.segments if s.points for t, f in s.points
    ]


def _freq_at(result, t):
    """轨迹上最接近 t 的测量频率。"""
    pts = _points(result)
    return min(pts, key=lambda p: abs(p[0] - t))[1]


def test_track_rising_glide_follows_frequency():
    """300 Hz/s 上升扫频，追踪应贴合扫频曲线，并在录音结束后停止。"""
    x = whistle_signal(900, 1500, start=0.2, end=1.95, duration=2.0)
    res = track_whistle(x, 0.4, 960, 600)
    assert _kinds(res) == ["track"]
    for t in (0.4, 0.8, 1.2, 1.6):
        expected = 900 + 600 * (t - 0.2) / 1.75
        assert _freq_at(res, t) == pytest.approx(expected, abs=12.0)
    assert res.stop_reason == "end_of_recording"


def test_track_falling_glide_with_noise():
    x = whistle_signal(1800, 900, start=0.3, end=1.6, duration=2.0,
                       noise_floor=0.01)
    res = track_whistle(x, 0.5, 1680, 1500)
    assert _kinds(res) == ["track"]
    expected = 1800 - 900 * (0.5 - 0.3) / 1.3
    assert res.start_freq == pytest.approx(expected, abs=15.0)


def test_crossing_keeps_original_direction_and_flags_segment():
    """两条谱线交叉：保留起点方向，交叉后可成立的多走向区段标为待复听。"""
    x = whistle_signal(
        1000, 1500, start=0.2, end=1.4, duration=1.8,
        extra_sines=((0.2, 1.4, 1500, 1000),),
    )
    res = track_whistle(x, 0.3, 1046, 800)
    assert "ambiguous" in _kinds(res)
    # 起点向上，过交叉点（约 1250Hz/0.8s）后仍继续向上，而不是跟着另一条下降
    after = [(t, f) for t, f in _points(res) if t >= 0.95]
    assert after and after[0][1] > 1250
    assert res.stop_freq is not None and res.stop_freq > 1450
    assert res.ambiguous_seconds > 0


def test_crossing_direction_from_falling_start():
    x = whistle_signal(
        1000, 1500, start=0.2, end=1.4, duration=1.8,
        extra_sines=((0.2, 1.4, 1500, 1000),),
    )
    res = track_whistle(x, 1.1, 1166, 800)  # 从交叉后的下降支开始
    assert res.start_freq == pytest.approx(1125.0, abs=10.0)
    assert res.stop_freq is not None and res.stop_freq < 1050
    pts = [(t, f) for t, f in _points(res) if t <= 1.35]
    assert pts == sorted(pts, key=lambda p: -p[1]) or all(
        f2 <= f1 + 1 for (_, f1), (_, f2) in zip(pts, pts[1:])
    )


def test_short_gap_connected_with_missing_measurement():
    """80 ms 中断：两端符合变频上限且接法唯一 -> 续接，缺口标缺测。"""
    gap = 0.08
    x = whistle_signal(1000, 1300, start=0.2, end=1.75, duration=1.8,
                       gaps=((0.9, 0.9 + gap),))
    res = track_whistle(x, 0.4, 1060, 1000)
    kinds = _kinds(res)
    assert kinds == ["track", "gap", "track"]
    seg = res.segments[1]
    assert seg.end_time - seg.start_time == pytest.approx(gap, abs=0.02)
    assert seg.end_freq > seg.start_freq          # 缺口两端仍按原走向上升
    assert res.stop_reason == "end_of_recording"


def test_gap_of_exactly_120ms_is_connected():
    x = whistle_signal(1000, 1200, start=0.2, end=1.75, duration=1.8,
                       gaps=((0.9, 1.02),))
    res = track_whistle(x, 0.4, 1033, 600)
    assert "gap" in _kinds(res)
    assert res.stop_reason == "end_of_recording"


def test_gap_longer_than_120ms_stops_tracking():
    x = whistle_signal(1000, 1200, start=0.2, end=1.75, duration=1.8,
                       gaps=((0.9, 1.08),))
    res = track_whistle(x, 0.4, 1033, 600)
    assert res.stop_reason == "gap_too_long"
    assert "gap" not in _kinds(res)
    assert res.stop_time < 0.95
    # 中断之后不再产生轨迹点
    assert all(t < 0.95 for t, _ in _points(res))


def test_frequency_jump_beyond_limit_stops():
    """频率突变超出每秒最大变化量，按中断处理，超时即停止。"""
    x = whistle_signal(
        1200, 1250, start=0.1, end=1.0, duration=1.5,
        extra_sines=((1.0, 1.4, 1600, 1600),),
    )
    loose = track_whistle(x, 0.3, 1210, 200)
    assert loose.stop_reason == "gap_too_long"
    assert loose.stop_freq is not None and loose.stop_freq < 1300
    # 变频上限足够大时，同一段录音能跨过 350Hz 的跳变继续追上
    wide = track_whistle(x, 0.3, 1210, 5000)
    assert wide.stop_time > 1.3
    assert wide.stop_freq == pytest.approx(1600, abs=30)


def test_broadband_burst_short_gap_is_bridged():
    """冲击声盖住啸声 50 ms：谱峰短暂中断，仍按缺测续接。"""
    x = whistle_signal(1200, 1500, start=0.2, end=1.8, duration=2.0,
                       bursts=((0.95, 0.05, 5.0),))
    res = track_whistle(x, 0.4, 1262, 1000)
    assert "gap" in _kinds(res) or "ambiguous" in _kinds(res)
    assert res.stop_time > 1.7


def test_invalid_params_rejected():
    x = whistle_signal(1200, 1500, start=0.2, end=1.8, duration=2.0)
    with pytest.raises(TrackValidationError, match="起点时刻"):
        track_whistle(x, -0.1, 1262, 1000)
    with pytest.raises(TrackValidationError, match="起点时刻"):
        track_whistle(x, 5.0, 1262, 1000)
    with pytest.raises(TrackValidationError, match="近似频率"):
        track_whistle(x, 0.4, 100, 1000)
    with pytest.raises(TrackValidationError, match="近似频率"):
        track_whistle(x, 0.4, 2500, 1000)
    with pytest.raises(TrackValidationError, match="最大频率变化量"):
        track_whistle(x, 0.4, 1262, 0)
    with pytest.raises(TrackValidationError):
        track_whistle(x, float("nan"), 1262, 1000)


def test_no_clear_peak_at_start_rejected():
    x = whistle_signal(1200, 1500, start=0.2, end=1.8, duration=2.0,
                       noise_floor=0.01)
    with pytest.raises(TrackValidationError, match="找不到清晰"):
        track_whistle(x, 0.4, 400, 1000)       # 频率处没有谱峰
    with pytest.raises(TrackValidationError, match="找不到清晰"):
        track_whistle(x, 0.05, 1262, 1000)     # 啸声尚未开始


def test_impulse_only_start_rejected():
    """起点只有宽带冲击、没有持续窄带谱线时拒绝。"""
    x = whistle_signal(1200, 1500, start=0.2, end=1.8, duration=2.0,
                       bursts=((0.5, 0.02, 1.5),))
    with pytest.raises(TrackValidationError, match="找不到清晰"):
        track_whistle(x, 0.5, 900, 1000)


def test_noise_only_recording_rejected():
    x = whistle_signal(0, start=0.0, end=0.0, duration=2.0,
                       amp=0.0, noise_floor=0.02)
    with pytest.raises(TrackValidationError):
        track_whistle(x, 0.5, 1000, 1000)


def test_steady_tone():
    x = whistle_signal(900, end=1.9, duration=2.0, noise_floor=0.005)
    res = track_whistle(x, 0.4, 900, 500)
    assert _kinds(res) == ["track"]
    assert all(abs(f - 900) < 5 for _, f in _points(res))


def test_segments_cover_continuously_without_overlap():
    x = whistle_signal(1000, 1300, start=0.2, end=1.6, duration=1.8,
                       gaps=((0.9, 0.98),))
    res = track_whistle(x, 0.4, 1060, 1000)
    for a, b in zip(res.segments, res.segments[1:]):
        assert a.end_time <= b.start_time + 1e-6


def test_analysis_respects_band():
    """带外的强单音不进入 200~2000Hz 追踪范围。"""
    x = whistle_signal(1000, 1400, start=0.2, end=1.6, duration=1.8,
                       extra_sines=((0.1, 1.7, 3000, 3000),))
    res = track_whistle(x, 0.4, 1050, 1000)
    assert all(200 <= f <= 2000 for _, f in _points(res))


def test_gap_with_multiple_candidates_stops_with_review_hint():
    """中断两端有多个可接峰（接法不唯一）时停止，而不是武断续接。"""
    import app.tracking as T

    x = whistle_signal(1000, 1100, start=0.2, end=1.75, duration=1.8,
                       gaps=((0.9, 1.2),))
    frames = T.detect_peaks(x)
    times = T.frame_times(1.8)
    start = T.find_start_peak(frames, times, 0.4, 1030)
    for fi in range(len(frames)):
        if 0.99 <= float(times[fi]) <= 1.02:
            frames[fi] = [
                T.Peak(fi, float(times[fi]), 1090.0, 20.0, 10.0),
                T.Peak(fi, float(times[fi]), 1050.0, 20.0, 10.0),
            ]
    res = T.track_detections(frames, times, start, 1000.0)
    assert res.stop_reason == "gap_ambiguous"
    assert "gap" not in _kinds(res)
    assert all(t < 0.95 for t, _ in _points(res))


def test_single_frame_noise_spike_is_not_a_crossing():
    """单帧冒出的噪声峰不应把连续啸声误标为待复听。"""
    x = whistle_signal(900, end=1.9, duration=2.0, noise_floor=0.006)
    res = track_whistle(x, 0.4, 900, 500)
    assert _kinds(res) == ["track"]
