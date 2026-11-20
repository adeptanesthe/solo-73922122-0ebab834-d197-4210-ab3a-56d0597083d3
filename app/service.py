"""检查服务与结果存储。

无效音频（格式错误、参考不达标等）直接抛错，由 API 层返回 400，
且完全不触碰已保存的上一次有效结果。
"""
from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio import (
    detect_impacts,
    parse_wav,
    validate_reference,
)
from .matching import align_impacts
from .tracking import STOP_TEXT, track_whistle as compute_whistle_track

MAX_UPLOAD_BYTES = 16_000 * 2 * 8 + 4096  # 8 秒 16k/16bit/单声道 + WAV 头余量


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.uploads_dir = data_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_result: dict[str, Any] | None = None
        self._last_track: dict[str, Any] | None = None

    def _save_wav(self, raw: bytes, prefix: str) -> str:
        digest = hashlib.sha256(raw).hexdigest()[:16]
        name = f"{prefix}_{digest}.wav"
        path = self.uploads_dir / name
        if not path.exists():
            path.write_bytes(raw)
        return f"/data/uploads/{name}"

    def inspect(self, reference_raw: bytes, test_raw: bytes) -> dict[str, Any]:
        # 先完整解析与检测，任何一步抛错都不会写入状态
        ref_samples = parse_wav(reference_raw)
        test_samples = parse_wav(test_raw)

        ref_det = detect_impacts(ref_samples)
        validate_reference(ref_det)  # 参考不达标 -> AudioValidationError
        test_det = detect_impacts(test_samples)

        alignment = align_impacts(ref_det.times, test_det.times)

        matched_ref = {i for i, _ in alignment.matched}
        matched_test = {j for _, j in alignment.matched}
        ref_url = self._save_wav(reference_raw, "ref")
        test_url = self._save_wav(test_raw, "test")

        reference_impacts = [
            {
                "index": i,
                "time": round(t, 4),
                "matched": i in matched_ref,
            }
            for i, t in enumerate(ref_det.times)
        ]
        test_impacts = [
            {
                "index": j,
                "time": round(t, 4),
                "matched": j in matched_test,
                "is_extra": j in alignment.extra_test_indices,
            }
            for j, t in enumerate(test_det.times)
        ]

        missed = [
            {
                "ref_index": i,
                "ref_time": round(ref_det.times[i], 4),
                "expected_time": round(
                    alignment.expected_missed_times.get(i, ref_det.times[i]), 4
                ),
            }
            for i in alignment.missed_ref_indices
        ]
        extras = [
            {
                "test_index": j,
                "time": round(test_det.times[j], 4),
            }
            for j in alignment.extra_test_indices
        ]
        matches = [
            {
                "ref_index": i,
                "test_index": j,
                "ref_time": round(ref_det.times[i], 4),
                "test_time": round(test_det.times[j], 4),
            }
            for i, j in alignment.matched
        ]

        verdict = "ok"
        if alignment.ambiguous:
            verdict = "ambiguous"
        elif alignment.miss_count or alignment.extra_count:
            verdict = "defect"

        warnings: list[str] = []
        if not test_det.times:
            warnings.append("待检录音未检测到任何冲击，全部参考冲次按漏冲处理")
        if test_det.suppressed_echoes:
            warnings.append(
                f"已抑制 {len(test_det.suppressed_echoes)} 个 40ms 内回声，不计多冲"
            )
        if alignment.ambiguous:
            warnings.append("存在多种同样符合节奏与最少漏/多冲规则的归属，请人工复听")

        result = {
            "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "ambiguous": alignment.ambiguous,
            "optimal_way_count": alignment.optimal_way_count,
            "summary": {
                "reference_impact_count": len(ref_det.times),
                "test_impact_count": len(test_det.times),
                "matched_count": len(alignment.matched),
                "miss_count": alignment.miss_count,
                "extra_count": alignment.extra_count,
                "suppressed_echo_count": len(test_det.suppressed_echoes),
            },
            "reference": {
                "url": ref_url,
                "duration": round(ref_det.duration, 4),
                "impacts": reference_impacts,
            },
            "test": {
                "url": test_url,
                "duration": round(test_det.duration, 4),
                "impacts": test_impacts,
            },
            "matches": matches,
            "missed": missed,
            "extras": extras,
            "timing_ratios": [round(r, 4) for r in alignment.timing_ratios],
            "warnings": warnings,
        }

        with self._lock:
            self._last_result = result
        return result

    def get_last(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._last_result is None else dict(self._last_result)

    def clear(self) -> None:
        with self._lock:
            self._last_result = None

    # ---- 送料啸声追踪 -----------------------------------------------------
    def track_whistle(
        self,
        raw: bytes,
        start_time: float,
        start_freq: float,
        max_rate_hz_s: float,
    ) -> dict[str, Any]:
        """从指定起点追踪啸声谱峰。

        与冲次核对一致：参数/音频/起点任何一步不合法都抛
        ``TrackValidationError``，且不写入状态，上一次轨迹原样保留。
        """
        samples = parse_wav(raw)  # 格式不合格 -> AudioValidationError
        tracking = compute_whistle_track(
            samples, start_time, start_freq, max_rate_hz_s
        )

        url = self._save_wav(raw, "whistle")
        duration = round(tracking.duration, 4)
        segments = [
            {
                "kind": seg.kind,
                "label": seg.label,
                "start_time": round(seg.start_time, 4),
                "end_time": round(seg.end_time, 4),
                "start_freq": None if seg.start_freq is None else round(seg.start_freq, 1),
                "end_freq": None if seg.end_freq is None else round(seg.end_freq, 1),
                "points": [
                    [round(t, 4), round(f, 1)] for t, f in seg.points
                ],
            }
            for seg in tracking.segments
        ]

        warnings: list[str] = []
        if tracking.stop_reason == "gap_ambiguous":
            warnings.append(
                "中断处有多个符合变频上限的谱峰，接法不唯一，按规则停止追踪，"
                "请人工复听中断位置附近"
            )
        if tracking.stop_reason == "gap_too_long":
            warnings.append("谱峰中断超过 120 ms，按规则停止追踪")
        if any(s["kind"] == "ambiguous" for s in segments):
            warnings.append("交叉附近存在多条可成立的走向，标出的区段请人工复听")

        result = {
            "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": {
                "start_time": round(tracking.start_time, 4),
                "start_freq": round(tracking.start_freq, 1),
                "max_rate_hz_s": round(max_rate_hz_s, 3),
                "requested_start_time": round(start_time, 4),
                "requested_start_freq": round(start_freq, 1),
                "fmin_hz": 200.0,
                "fmax_hz": 2000.0,
                "max_gap_ms": 120,
            },
            "audio": {"url": url, "duration": duration},
            "track": {
                "stop_time": round(tracking.stop_time, 4),
                "stop_freq": None if tracking.stop_freq is None else round(tracking.stop_freq, 1),
                "stop_reason": tracking.stop_reason,
                "stop_text": STOP_TEXT.get(tracking.stop_reason, tracking.stop_reason),
            },
            "segments": segments,
            "summary": {
                "segment_count": len(segments),
                "ambiguous_seconds": round(tracking.ambiguous_seconds, 3),
                "missing_seconds": round(tracking.missing_seconds, 3),
            },
            "warnings": warnings,
        }
        with self._lock:
            self._last_track = result
        return result

    def get_last_track(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._last_track is None else dict(self._last_track)

    def clear_track(self) -> None:
        with self._lock:
            self._last_track = None
