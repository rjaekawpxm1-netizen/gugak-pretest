# -*- coding: utf-8 -*-
"""AI 학습용 음원 가공 (ADR-004).

진단에서 나온 불균질성(샘플레이트·채널·음량·길이)을 각각 표준화한다:
  - to_mono     : 스테레오 → mono
  - resample    : 혼재 샘플레이트 → PROCESS_SR 통일
  - normalize   : 들쭉날쭉한 음량 → 목표 RMS로 정규화(피크 상한으로 클리핑 방지)
  - silence_segment : 긴 음원을 무음 지점 우선으로 15~30초 조각 분할
"""

import numpy as np
import soundfile as sf
import soxr

from .config import (PROCESS_SR, PROCESS_TARGET_RMS, PROCESS_PEAK_CEIL,
                     SILENCE_DB, AUDIO_LEN_MIN, AUDIO_LEN_MAX)

_HOP = 0.05      # 포락선 프레임(초)
_MIN_KEEP = 3.0  # 이보다 짧은 조각은 학습 부적합으로 버린다


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def resample(x: np.ndarray, sr_in: int, sr_out: int = PROCESS_SR) -> np.ndarray:
    return x if sr_in == sr_out else soxr.resample(x, sr_in, sr_out)


def normalize(x: np.ndarray,
              target_rms_db: float = PROCESS_TARGET_RMS,
              peak_ceil_db: float = PROCESS_PEAK_CEIL) -> np.ndarray:
    """목표 RMS로 스케일하되, 피크가 상한을 넘으면 상한에 맞춰 줄인다."""
    rms = float(np.sqrt(np.mean(x ** 2))) + 1e-12
    y = x * (10 ** (target_rms_db / 20) / rms)
    peak = float(np.max(np.abs(y))) + 1e-12
    ceil = 10 ** (peak_ceil_db / 20)
    if peak > ceil:
        y *= ceil / peak
    return y


def _envelope(x: np.ndarray, sr: int):
    win = max(1, int(sr * _HOP))
    n = (len(x) // win) * win
    if n == 0:
        return np.array([np.sqrt(np.mean(x ** 2))]), win
    amp = np.sqrt((x[:n].reshape(-1, win) ** 2).mean(axis=1))
    return amp, win


def silence_segment(x: np.ndarray, sr: int,
                    min_len: float = AUDIO_LEN_MIN, max_len: float = AUDIO_LEN_MAX,
                    silence_db: float = SILENCE_DB):
    """무음 인지 세그먼트 → [(시작샘플, 끝샘플), ...].

    앞뒤 무음을 잘라낸 뒤, [min_len, max_len] 구간 안에서 에너지가 가장 낮은
    지점을 컷으로 삼아 분할한다. 무음이 없으면 max_len에서 자른다.
    """
    amp, win = _envelope(x, sr)
    thr = 10 ** (silence_db / 20)
    nz = np.where(amp > thr)[0]
    if len(nz) == 0:
        return []
    start_f, end_f = int(nz[0]), int(nz[-1]) + 1
    hop_s = win / sr
    min_fr, max_fr = int(min_len / hop_s), int(max_len / hop_s)
    keep_fr = int(_MIN_KEEP / hop_s)

    segs = []
    cur = start_f
    while cur < end_f:
        if end_f - cur <= max_fr:                 # 남은 구간 ≤ max → 그대로 한 조각
            segs.append((cur, end_f))
            break
        window = amp[cur + min_fr: cur + max_fr]   # [min,max] 구간에서 최저 에너지 컷
        cut = cur + min_fr + int(np.argmin(window))
        segs.append((cur, cut))
        cur = cut

    segs = [(s, e) for s, e in segs if (e - s) >= keep_fr]  # 3초 미만 조각 제거
    return [(int(s * win), int(e * win)) for s, e in segs]


def process_file(path, sr_out: int = PROCESS_SR):
    """파일 1건 가공 → (가공신호, 출력sr, 세그먼트 샘플인덱스 리스트, 가공전 요약)."""
    data, sr = sf.read(str(path))
    before = {
        "sr": sr,
        "channels": data.shape[1] if data.ndim > 1 else 1,
        "duration": round(len(data) / sr, 2),
        "rms_db": round(float(20 * np.log10(np.sqrt(np.mean((to_mono(data)) ** 2)) + 1e-12)), 2),
    }
    x = normalize(resample(to_mono(data), sr, sr_out))
    segs = silence_segment(x, sr_out)
    return x, sr_out, segs, before