"""送料啸声（窄带谱峰）追踪。

从检修员指定的起点时刻与近似频率出发，在 200~2000 Hz 内逐帧寻找
窄带谱峰，只沿满足「每秒最大频率变化量」的走向前进：

* 谱峰交叉：若同一帧有多个都可从当前走向按变频上限接上的谱峰，
  取与近期走向预测最接近的一个继续，并把交叉附近多个走向都成立的
  区段标记为「待复听」；
* 短中断：谱峰中断不超过 120 ms 时，只有两端频率变化符合上限且
  可接的谱峰唯一时才续接，缺口记为「缺测」；
* 更长中断：停止追踪。

本模块只做检测与轨迹判定，WAV 校验复用 :mod:`app.audio`。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import SAMPLE_RATE

# ---- 分析参数 -------------------------------------------------------------
FMIN_HZ = 200.0
FMAX_HZ = 2000.0
FRAME_SEC = 0.064          # 64 ms 窗，保证窄带峰可分
HOP_SEC = 0.010            # 10 ms 帧移，轨迹时间分辨率
N_FFT = 2048               # 7.8125 Hz/bin（补零后）
FRAME_SAMPLES = int(round(FRAME_SEC * SAMPLE_RATE))
HOP_SAMPLES = int(round(HOP_SEC * SAMPLE_RATE))
BIN_HZ = SAMPLE_RATE / N_FFT

# 谱峰判定
SMOOTH_BINS = 9            # 局部背景平滑宽度（约 70 Hz）
PEAK_PROMINENCE_DB = 4.0   # 峰高于局部背景的幅度
PEAK_MIN_SNR_DB = 12.0     # 峰高于全帧底噪的幅度（排除白噪声起伏）
PEAK_DYNAMIC_RANGE_DB = 35.0  # 只保留比本帧最强分量弱不超过此值的峰
MAX_PEAKS_PER_FRAME = 10
PEAK_TRANSIENT_LEVEL_DB = 6.0  # 单帧峰与其相邻帧同伴的最大强度差

# 轨迹连接
LINK_TOL_HZ = 12.0         # 逐帧连接的频率测量容差（约 1.5 bin）
GAP_TOL_HZ = 12.0          # 跨缺口连接时同样允许的测量容差
MAX_GAP_SEC = 0.120        # 允许续接的最长谱峰中断
# 分析窗长 64 ms，啸声起止处的帧仍会部分含声，检测到的「缺席」通常比
# 真实中断长约半个窗；判定时先扣掉这部分窗展宽，使 120 ms 规则对得上
# 实际中断时长。
GAP_SMEAR_SEC = FRAME_SEC / 2
CROSS_MIN_SEPARATION_HZ = 15.0  # 两个候选峰至少差这么多，才算「两条走向」
AMBIGUITY_HORIZON_SEC = 0.12    # 另一条走向至少延续这么久，才算「均可成立」
AMBIGUITY_MERGE_SEC = 0.12      # 短于此的确定段并入相邻待复听区段
START_TIME_TOL_SEC = 0.050
START_FREQ_TOL_HZ = 60.0
START_SUPPORT_FRAMES = 2   # 起点峰至少在相邻两帧中出现，才算「清晰」


EPS = 1e-9

STOP_TEXT = {
    "end_of_recording": "录音结束",
    "gap_too_long": "谱峰中断超过 120 ms，追踪停止",
    "gap_ambiguous": "中断处有多个符合变频上限的谱峰，接法不唯一，追踪停止",
}

SEGMENT_LABELS = {
    "track": "轨迹",
    "ambiguous": "待复听",
    "gap": "缺测",
}


class TrackValidationError(ValueError):
    """追踪参数不合法或起点找不到清晰谱峰。"""


@dataclass
class Peak:
    frame: int
    time: float
    freq: float
    level_db: float
    prominence_db: float


@dataclass
class Segment:
    kind: str                       # track / ambiguous / gap
    start_time: float
    end_time: float
    start_freq: float | None
    end_freq: float | None
    points: list[tuple[float, float]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return SEGMENT_LABELS[self.kind]


@dataclass
class TrackingResult:
    start_time: float
    start_freq: float
    segments: list[Segment]
    stop_time: float
    stop_freq: float | None
    stop_reason: str
    duration: float
    ambiguous_seconds: float = 0.0
    missing_seconds: float = 0.0


# ---- 谱峰检测 -------------------------------------------------------------
def _frame_layout(n_samples: int) -> tuple[int, int]:
    """返回 (帧数, 起始填充)，帧中心对应原始录音时间 k*hop。"""
    pad = FRAME_SAMPLES // 2
    padded_len = n_samples + 2 * pad
    if padded_len < FRAME_SAMPLES:
        return 1, max(0, pad - (FRAME_SAMPLES - padded_len))
    n_frames = 1 + (padded_len - FRAME_SAMPLES) // HOP_SAMPLES
    return n_frames, pad


def frame_times(duration: float) -> np.ndarray:
    """与 :func:`detect_peaks` 对应的帧中心时刻（秒）。"""
    n_samples = int(round(duration * SAMPLE_RATE))
    n_frames, _ = _frame_layout(n_samples)
    return np.arange(n_frames, dtype=np.float64) * HOP_SEC


def detect_peaks(samples: np.ndarray) -> list[list[Peak]]:
    """逐帧检测 200~2000 Hz 内的窄带谱峰。

    先取 200~2000 Hz 的分析带，再按以下条件留峰：高于本帧底噪
    ``PEAK_MIN_SNR_DB``、比局部平滑谱突出 ``PEAK_PROMINENCE_DB``、
    且比本帧最强分量弱不超过 ``PEAK_DYNAMIC_RANGE_DB``（抑制窗函数
    泄漏和冲击声的宽带裙边）。返回与 :func:`frame_times` 等长的列表。
    """
    n_frames, pad = _frame_layout(len(samples))
    window = np.hanning(FRAME_SAMPLES).astype(np.float32)
    padded = np.pad(samples.astype(np.float32), (pad, pad), mode="constant")

    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    band = np.where((freqs >= FMIN_HZ) & (freqs <= FMAX_HZ))[0]
    if band.size < 3:
        return [[] for _ in range(n_frames)]
    lo, hi = int(band[0]), int(band[-1])
    band_freqs = freqs[lo : hi + 1]

    frames: list[list[Peak]] = []
    pad_bins = SMOOTH_BINS // 2
    for k in range(n_frames):
        start = k * HOP_SAMPLES
        frame = padded[start : start + FRAME_SAMPLES]
        if frame.shape[0] < FRAME_SAMPLES:
            break
        mag = np.abs(np.fft.rfft(frame * window, n=N_FFT))
        db = 20.0 * np.log10(mag[lo : hi + 1] + 1e-10)
        # 用平滑谱作为局部背景，宽带冲击不会形成「突出」的窄带峰
        smooth = np.convolve(
            np.pad(db, (pad_bins, pad_bins), mode="edge"),
            np.ones(SMOOTH_BINS) / SMOOTH_BINS,
            mode="valid",
        )
        floor = float(np.median(db))
        strongest = float(np.max(db))
        peaks: list[Peak] = []
        for i in range(1, len(db) - 1):
            level = float(db[i])
            if level < floor + PEAK_MIN_SNR_DB:
                continue
            if level < strongest - PEAK_DYNAMIC_RANGE_DB:
                continue
            prominence = level - float(smooth[i])
            if prominence < PEAK_PROMINENCE_DB:
                continue
            if not (level >= db[i - 1] and level >= db[i + 1]):
                continue
            # 抛物线插值，得到亚 bin 频率
            a, b, c = float(db[i - 1]), level, float(db[i + 1])
            denom = a - 2.0 * b + c
            delta = 0.0
            if abs(denom) > 1e-12:
                delta = 0.5 * (a - c) / denom
                delta = float(np.clip(delta, -0.5, 0.5))
            est = float(band_freqs[i] + delta * BIN_HZ)
            peaks.append(
                Peak(
                    frame=k,
                    time=k * HOP_SEC,
                    freq=est,
                    level_db=b,
                    prominence_db=prominence,
                )
            )
        peaks.sort(key=lambda p: p.level_db, reverse=True)
        # 只保留互不重叠的局部极大值：强峰两侧一个 bin 内的相邻极大值是
        # 同一个物理峰，不应被当成两条走向
        kept: list[Peak] = []
        for p in peaks:
            if any(abs(p.freq - q.freq) < CROSS_MIN_SEPARATION_HZ for q in kept):
                continue
            kept.append(p)
        frames.append(kept[:MAX_PEAKS_PER_FRAME])
    return _drop_transient_peaks(frames)


def _drop_transient_peaks(
    frames: list[list[Peak]], tol: float = START_FREQ_TOL_HZ
) -> list[list[Peak]]:
    """剔除只在一帧里冒出的谱峰（白噪声起伏、窗函数边带）。

    真实啸声谱线会连续若干帧存在；只在单帧出现的峰若进入追踪，
    会被误判成交叉或短中断。这里要求峰在前一帧或后一帧也有同伴，
    且该同伴的强度接近（排除强谱线的窗旁瓣）。
    """
    if len(frames) <= 1:
        return frames
    cleaned: list[list[Peak]] = []
    for k, peaks in enumerate(frames):
        neighbours: list[Peak] = []
        if k > 0:
            neighbours.extend(frames[k - 1])
        if k + 1 < len(frames):
            neighbours.extend(frames[k + 1])
        kept = [
            p
            for p in peaks
            if any(
                abs(q.freq - p.freq) <= tol
                and abs(q.level_db - p.level_db) <= PEAK_TRANSIENT_LEVEL_DB
                for q in neighbours
            )
        ]
        cleaned.append(kept)
    return cleaned


def _nearby_peak(peaks: list[Peak], freq: float, tol: float) -> Peak | None:
    found = [p for p in peaks if abs(p.freq - freq) <= tol]
    if not found:
        return None
    return min(found, key=lambda p: abs(p.freq - freq))


def _persistent(
    frames: list[list[Peak]], frame: int, freq: float, lookahead: int = 3
) -> bool:
    """该频率附近的谱峰是否在随后若干帧中持续出现（排除瞬时噪声/冲击）。"""
    seen = 0
    for fj in range(frame + 1, min(len(frames), frame + 1 + lookahead)):
        if _nearby_peak(frames[fj], freq, LINK_TOL_HZ) is not None:
            seen += 1
    return seen >= 2


def find_start_peak(
    frames: list[list[Peak]],
    times: np.ndarray,
    start_time: float,
    start_freq: float,
) -> Peak:
    """在指定时刻/频率附近找清晰、持续至少两帧的窄带谱峰。"""
    if len(frames) == 0:
        raise TrackValidationError("录音过短，无法分析频谱，起点处找不到清晰谱峰")
    candidates: list[Peak] = []
    for fi, peaks in enumerate(frames):
        t = float(times[fi])
        if abs(t - start_time) > START_TIME_TOL_SEC + EPS:
            continue
        for p in peaks:
            if abs(p.freq - start_freq) <= START_FREQ_TOL_HZ + EPS:
                candidates.append(p)
    if not candidates:
        raise TrackValidationError(
            f"起点处找不到清晰谱峰：在 {start_time:.3f}s ± "
            f"{START_TIME_TOL_SEC * 1000:.0f}ms、{start_freq:.0f}Hz ± "
            f"{START_FREQ_TOL_HZ:.0f}Hz 范围内未检测到窄带峰，"
            "请核对起点时刻与近似频率"
        )

    def support(p: Peak) -> int:
        """起点峰之后 ±2 帧内与该峰频率接近的帧数，用于排除瞬时噪声。"""
        n = 0
        for fj in range(max(0, p.frame - 2), min(len(frames), p.frame + 3)):
            if _nearby_peak(frames[fj], p.freq, LINK_TOL_HZ) is not None:
                n += 1
        return n

    clear = [
        p for p in candidates if support(p) >= START_SUPPORT_FRAMES
    ]
    if not clear:
        raise TrackValidationError(
            f"起点处找不到清晰、稳定的谱峰：{start_time:.3f}s 附近虽有峰，"
            "但只出现一帧，可能是冲击噪声；请换一个啸声清楚的时刻"
        )
    # 持续到之后的峰优先：起点后 30ms 内仍在同一频率的峰不是转瞬即逝的冲击
    continuing = [p for p in clear if _persistent(frames, p.frame, p.freq)]
    pool = continuing or clear

    def merged_within_range(p: Peak, tol: float) -> bool:
        """p 是否是两个相近分量的混合峰（交叉/贴近时不应作为起点）。"""
        for fj in range(max(0, p.frame - 1), min(len(frames), p.frame + 2)):
            near = [q for q in frames[fj] if abs(q.freq - p.freq) <= tol]
            if len(near) > 1 and all(
                abs(q.freq - p.freq) >= CROSS_MIN_SEPARATION_HZ for q in near
            ):
                return True
        return False

    # 交叉点的混合峰会同时满足两个走向，起点必须明确落在其中一条上
    clean = [p for p in pool if not merged_within_range(p, START_FREQ_TOL_HZ)]
    pool = clean or pool

    def score(p: Peak) -> tuple[float, float]:
        dist = (
            abs(p.time - start_time) / START_TIME_TOL_SEC
            + abs(p.freq - start_freq) / START_FREQ_TOL_HZ
        )
        return (dist, -p.level_db)

    chosen = min(pool, key=score)
    # 走向明确性：起点选定走向后，另一条谱峰若紧随其后同样存在，
    # 起点本身就在歧义区，不能作为可靠起点
    if merged_within_range(chosen, START_FREQ_TOL_HZ):
        raise TrackValidationError(
            f"起点处谱峰不清晰：{chosen.time:.3f}s / {chosen.freq:.0f}Hz 附近同时"
            "存在多条接近的谱线，无法确定啸声走向；请换一个谱线分开的起点"
        )
    return chosen


# ---- 轨迹连接 -------------------------------------------------------------
def _recent_slope(events: list[dict], window: int = 10) -> float:
    """最近一段轨迹的频率斜率（Hz/s）。

    用最小二乘拟合；交叉处的合并平台会把短期斜率拉平，因此窗口取
    约 100 ms，让交叉前的走向仍占主导。
    """
    pts = [(e["time"], e["freq"]) for e in events if e["kind"] == "point"][-window:]
    if len(pts) < 2:
        return 0.0
    ts = np.asarray([p[0] for p in pts], dtype=np.float64)
    fs = np.asarray([p[1] for p in pts], dtype=np.float64)
    ts = ts - ts[0]
    denom = float(np.sum(ts * ts))
    if denom <= 0:
        return 0.0
    return float(np.sum(ts * (fs - fs.mean())) / denom)


def _is_bridge(
    frames: list[list[Peak]], frame: int, freq: float, reach: int = 6
) -> bool:
    """当前帧是否落在两条谱线汇合（交叉）处：前后不远处都存在分歧峰。"""
    tol = START_FREQ_TOL_HZ
    left = any(
        len([q for q in frames[fj] if abs(q.freq - freq) <= tol]) > 1
        for fj in range(max(0, frame - reach), frame)
    )
    right = any(
        len([q for q in frames[fj] if abs(q.freq - freq) <= tol]) > 1
        for fj in range(frame + 1, min(len(frames), frame + reach + 1))
    )
    return left and right


def _choose_candidate(
    candidates: list[Peak], last_freq: float, predicted: float
) -> Peak:
    """选与近期走向外推最接近的峰，从而在交叉后保留原走向。"""
    return min(
        candidates,
        key=lambda p: (abs(p.freq - predicted), abs(p.freq - last_freq)),
    )


def _track_branch(
    frames: list[list[Peak]],
    times: np.ndarray,
    start_frame: int,
    freq: float,
    rate: float,
) -> dict[int, float]:
    """从某一帧的给定频率出发，沿最简单的「最近峰」规则试走一条支路。

    只用于判断交叉处的另一条走向是否同样成立，不参与主轨迹。
    返回 {帧号: 频率}，遇到中断即停止。
    """
    path: dict[int, float] = {}
    last_frame = start_frame
    last_freq = freq
    for fi in range(start_frame + 1, len(frames)):
        dt = float(times[fi]) - float(times[last_frame])
        if dt <= EPS or dt > MAX_GAP_SEC + EPS:
            break
        allowed = rate * dt + LINK_TOL_HZ
        candidates = [
            p for p in frames[fi] if abs(p.freq - last_freq) <= allowed + EPS
        ]
        if not candidates:
            break
        chosen = min(candidates, key=lambda p: abs(p.freq - last_freq))
        path[fi] = chosen.freq
        last_frame, last_freq = fi, chosen.freq
    return path


def _alternative_persists(
    frames: list[list[Peak]],
    times: np.ndarray,
    frame: int,
    alternatives: list[Peak],
    rate: float,
) -> bool:
    """另一条走向在交叉后是否仍持续到本帧之后，足以视为「均可成立」。"""
    for alt in alternatives:
        path = _track_branch(frames, times, frame, alt.freq, rate)
        if not path:
            continue
        end_time = float(times[max(path)])
        if end_time - float(times[frame]) >= AMBIGUITY_HORIZON_SEC - EPS:
            # 主轨迹在该时段是否也仍然存在（主轨迹仍在继续，说明两条都成立）
            return True
    return False


def _build_segments(events: list[dict]) -> list[Segment]:
    segments: list[Segment] = []
    run: list[dict] = []

    def flush() -> None:
        nonlocal run
        if not run:
            return
        points = [(e["time"], e["freq"]) for e in run]
        segments.append(
            Segment(
                kind=run[0]["status"],
                start_time=points[0][0],
                end_time=points[-1][0],
                start_freq=points[0][1],
                end_freq=points[-1][1],
                points=points,
            )
        )
        run = []

    for event in events:
        if event["kind"] == "gap":
            flush()
            segments.append(
                Segment(
                    kind="gap",
                    start_time=event["time"],
                    end_time=event["end_time"],
                    start_freq=event["freq"],
                    end_freq=event["end_freq"],
                    points=[],
                )
            )
        else:
            if run and run[0]["status"] != event["status"]:
                flush()
            run.append(event)
    flush()
    return _merge_short_track_segments(segments)


def _merge_short_track_segments(segments: list[Segment]) -> list[Segment]:
    """把夹在待复听区段之间的零碎确定段并入待复听，保持区段可读。"""
    merged: list[Segment] = []
    for seg in segments:
        if (
            merged
            and seg.kind == "track"
            and merged[-1].kind == "ambiguous"
            and seg.end_time - seg.start_time < AMBIGUITY_MERGE_SEC - EPS
        ):
            prev = merged[-1]
            prev.end_time = seg.end_time
            prev.end_freq = seg.end_freq
            prev.points.extend(seg.points)
            continue
        merged.append(seg)
    return merged


def track_detections(
    frames: list[list[Peak]],
    times: np.ndarray,
    start_peak: Peak,
    max_rate_hz_s: float,
) -> TrackingResult:
    """在已检测的逐帧谱峰上，从 ``start_peak`` 出发连接轨迹。"""
    events: list[dict] = [
        {
            "kind": "point",
            "time": start_peak.time,
            "freq": start_peak.freq,
            "status": "track",
        }
    ]
    last_time = start_peak.time
    last_freq = start_peak.freq
    stop_reason = "end_of_recording"
    in_gap = False
    gap_start_time = 0.0
    gap_start_freq = 0.0
    gap_ambiguous_seen = False
    ambiguous_until = -1.0

    for fi in range(start_peak.frame + 1, len(frames)):
        t = float(times[fi])
        dt = t - last_time
        if dt <= EPS:
            continue
        if in_gap and dt - GAP_SMEAR_SEC > MAX_GAP_SEC + EPS:
            stop_reason = "gap_ambiguous" if gap_ambiguous_seen else "gap_too_long"
            break

        peaks = frames[fi]
        tol = GAP_TOL_HZ if in_gap else LINK_TOL_HZ
        allowed = max_rate_hz_s * dt + tol
        candidates = [
            p for p in peaks if abs(p.freq - last_freq) <= allowed + EPS
        ]

        if not in_gap:
            if not candidates:
                in_gap = True
                gap_start_time = last_time
                gap_start_freq = last_freq
                gap_ambiguous_seen = False
                continue
            slope = _recent_slope(events)
            predicted = last_freq + slope * dt
            chosen = _choose_candidate(candidates, last_freq, predicted)
            others = [
                p
                for p in candidates
                if p is not chosen
                and abs(p.freq - chosen.freq) >= CROSS_MIN_SEPARATION_HZ - EPS
            ]
            bridge = _is_bridge(frames, fi, chosen.freq)
            if bridge or (
                others
                and _alternative_persists(
                    frames, times, fi, others, max_rate_hz_s
                )
            ):
                # 交叉处的合并峰及其后一段，多条走向都符合变频上限
                ambiguous_until = max(ambiguous_until, t + AMBIGUITY_HORIZON_SEC)
            events.append(
                {
                    "kind": "point",
                    "time": t,
                    "freq": chosen.freq,
                    "status": "ambiguous" if t <= ambiguous_until + EPS else "track",
                }
            )
            last_time, last_freq = t, chosen.freq
            continue

        # 缺口内：必须唯一且符合变频上限才续接
        if len(candidates) > 1:
            # 有多个可接峰 -> 接法不唯一，不能续接；继续等待，
            # 若直到超过 120ms 仍不唯一则停止
            gap_ambiguous_seen = True
            continue
        if candidates:
            chosen = candidates[0]
            events.append(
                {
                    "kind": "gap",
                    "time": gap_start_time,
                    "end_time": t,
                    "freq": gap_start_freq,
                    "end_freq": chosen.freq,
                }
            )
            events.append(
                {
                    "kind": "point",
                    "time": t,
                    "freq": chosen.freq,
                    # 中断后接法唯一，缺口两端已确定，不再沿用交叉处的待复听
                    "status": "track",
                }
            )
            last_time, last_freq = t, chosen.freq
            in_gap = False
            ambiguous_until = -1.0  # 缺口后的新起点不沿用缺口前的待复听窗口
        # 没有可接峰：继续等待，直到超过 120ms 或录音结束

    segments = _build_segments(events)
    ambiguous_seconds = sum(
        s.end_time - s.start_time for s in segments if s.kind == "ambiguous"
    )
    missing_seconds = sum(
        s.end_time - s.start_time for s in segments if s.kind == "gap"
    )
    duration = len(frames) * HOP_SEC if len(frames) else 0.0
    return TrackingResult(
        start_time=start_peak.time,
        start_freq=start_peak.freq,
        segments=segments,
        stop_time=last_time,
        stop_freq=last_freq,
        stop_reason=stop_reason,
        duration=duration,
        ambiguous_seconds=ambiguous_seconds,
        missing_seconds=missing_seconds,
    )


def validate_track_params(
    start_time: float,
    start_freq: float,
    max_rate_hz_s: float,
    duration: float,
) -> None:
    if not np.isfinite(start_time):
        raise TrackValidationError("起点时刻必须是有限数值")
    if not np.isfinite(start_freq):
        raise TrackValidationError("近似频率必须是有限数值")
    if not np.isfinite(max_rate_hz_s):
        raise TrackValidationError("每秒最大频率变化量必须是有限数值")
    if start_time < 0 or start_time > duration + EPS:
        raise TrackValidationError(
            f"起点时刻需在 0～{duration:.3f}s（录音时长）之间，当前为 "
            f"{start_time:.3f}s"
        )
    if start_freq < FMIN_HZ or start_freq > FMAX_HZ:
        raise TrackValidationError(
            f"近似频率需在 {FMIN_HZ:.0f}～{FMAX_HZ:.0f} Hz 之间，"
            f"当前为 {start_freq:.1f} Hz"
        )
    if max_rate_hz_s <= 0:
        raise TrackValidationError(
            f"每秒最大频率变化量需大于 0 Hz/s，当前为 {max_rate_hz_s:.3f}"
        )


def track_whistle(
    samples: np.ndarray,
    start_time: float,
    start_freq: float,
    max_rate_hz_s: float,
) -> TrackingResult:
    """完整流程：校验参数 -> 检测谱峰 -> 定位起点 -> 追踪。"""
    duration = len(samples) / SAMPLE_RATE
    validate_track_params(start_time, start_freq, max_rate_hz_s, duration)
    frames = detect_peaks(samples)
    times = frame_times(duration)[: len(frames)]
    start_peak = find_start_peak(frames, times, start_time, start_freq)
    result = track_detections(frames, times, start_peak, max_rate_hz_s)
    result.duration = duration  # 以录音真实时长为准（帧中心只在内部使用）
    return result
