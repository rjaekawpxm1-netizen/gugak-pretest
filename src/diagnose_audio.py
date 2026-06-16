# -*- coding: utf-8 -*-
"""음원 품질 측정 + AI 학습 적합성 판정.

임의 가중치 합산 점수 대신 항목별 '적합/보완필요'를 투명하게 제시한다.
"""

from pathlib import Path
import numpy as np
import soundfile as sf

from .config import (AUDIO_LEN_MIN, AUDIO_LEN_MAX, SILENCE_DB, SILENCE_RATIO_WARN,
                     AI_TARGET_CHANNELS, AI_TARGET_SR)

_WIN_SEC = 0.05  # 에너지 포락선 윈도(초)


def _db(x: float) -> float:
    return round(float(20 * np.log10(x + 1e-12)), 2)


def measure(path: Path):
    """음원 1건 측정 → (속성 dict, 포락선 시간(s), 포락선 dBFS)."""
    info = sf.info(str(path))
    data, sr = sf.read(str(path))
    mono = data.mean(axis=1) if data.ndim > 1 else data

    dur = info.frames / info.samplerate
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono ** 2)))

    win = max(1, int(sr * _WIN_SEC))
    n = (len(mono) // win) * win
    env_amp = np.sqrt((mono[:n].reshape(-1, win) ** 2).mean(axis=1)) if n else np.array([0.0])
    env_db = 20 * np.log10(env_amp + 1e-12)
    times = np.arange(len(env_db)) * _WIN_SEC

    thr = 10 ** (SILENCE_DB / 20)
    nz = np.where(env_amp > thr)[0]
    lead = round(float(nz[0] * _WIN_SEC), 2) if len(nz) else round(dur, 2)
    trail = round(float((len(env_amp) - 1 - nz[-1]) * _WIN_SEC), 2) if len(nz) else round(dur, 2)

    props = {
        "file": Path(path).name,
        "format": f"{info.format}/{info.subtype}",
        "samplerate": info.samplerate,
        "channels": info.channels,
        "duration": round(dur, 2),
        "peak_db": _db(peak),
        "rms_db": _db(rms),
        "clipping": peak >= 0.99,
        "lead_silence": lead,
        "trail_silence": trail,
        "low_energy_ratio": round(float(np.mean(env_amp <= thr)) * 100, 1),
    }
    return props, times, env_db


def assess(props: dict):
    """속성 → 항목별 (항목, 판정, 사유) 리스트 + 종합 등급."""
    checks = []
    dur = props["duration"]

    len_ok = AUDIO_LEN_MIN <= dur <= AUDIO_LEN_MAX
    checks.append(("길이 적합성", "적합" if len_ok else "보완필요",
                   f"{dur}s (권장 {AUDIO_LEN_MIN}~{AUDIO_LEN_MAX}s)"
                   + ("" if len_ok else " → 세그먼트 분할 필요")))

    ch_ok = props["channels"] == AI_TARGET_CHANNELS
    checks.append(("채널", "적합" if ch_ok else "보완필요",
                   f"{props['channels']}ch" + ("" if ch_ok else " → mono 변환 필요")))

    sr_ok = props["samplerate"] in AI_TARGET_SR
    checks.append(("샘플레이트", "적합" if sr_ok else "확인",
                   f"{props['samplerate']}Hz" + ("" if sr_ok else f" → 학습표준 {AI_TARGET_SR} 리샘플 검토")))

    checks.append(("클리핑", "보완필요" if props["clipping"] else "적합",
                   f"peak {props['peak_db']}dBFS"))

    sil_ok = props["low_energy_ratio"] <= SILENCE_RATIO_WARN
    checks.append(("무음 비율", "적합" if sil_ok else "보완필요",
                   f"저에너지 {props['low_energy_ratio']}%"
                   + ("" if sil_ok else " → 무음 인지(silence-aware) 세그먼트 필요")))

    grade = "학습 적합" if all(c[1] == "적합" for c in checks) else "보완 후 활용"
    return checks, grade