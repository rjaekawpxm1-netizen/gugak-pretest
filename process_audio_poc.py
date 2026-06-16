# -*- coding: utf-8 -*-
"""AI 가공 PoC 실행 (ADR-004).

data/raw/audio_kogl1/ 의 음원 전체에 mono·리샘플·음량정규화·무음세그먼트를 적용하고,
가공된 조각을 data/processed/segments/<원본명>/ 에 저장, 결과를 manifest CSV로 남긴다.

실행(프로젝트 루트): python process_audio_poc.py
"""

import csv
import numpy as np
import soundfile as sf

from src.config import (AUDIO_KOGL1, SEGMENTS_DIR, PROCESS_MANIFEST, PROCESS_SR)
from src.process_audio import process_file


def _rms_db(x):
    return round(float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)), 2)


def main():
    files = sorted([*AUDIO_KOGL1.glob("*.wav"), *AUDIO_KOGL1.glob("*.mp3")])
    if not files:
        print("음원이 없습니다:", AUDIO_KOGL1)
        return
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in files:
        x, sr, segs, before = process_file(f)
        outdir = SEGMENTS_DIR / f.stem
        outdir.mkdir(parents=True, exist_ok=True)
        for i, (s, e) in enumerate(segs, 1):
            clip = x[s:e]
            outp = outdir / f"{f.stem}_seg{i:02d}.wav"
            sf.write(str(outp), clip, sr, subtype="PCM_16")
            rows.append({
                "source": f.name,
                "segment": outp.name,
                "idx": i,
                "start_sec": round(s / sr, 2),
                "dur_sec": round((e - s) / sr, 2),
                "src_sr": before["sr"],
                "src_ch": before["channels"],
                "src_rms_db": before["rms_db"],
                "out_sr": sr,
                "out_ch": 1,
                "out_rms_db": _rms_db(clip),
            })

    with open(PROCESS_MANIFEST, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    durs = [r["dur_sec"] for r in rows]
    print(f"가공 완료: 원본 {len(files)}개 → 세그먼트 {len(rows)}개")
    print(f"출력 통일: {PROCESS_SR}Hz / mono / 목표 -20dBFS")
    print(f"세그먼트 길이 {min(durs):.1f}~{max(durs):.1f}s "
          f"(15~30s 적합 {sum(15 <= d <= 30 for d in durs)}/{len(durs)})")
    print("manifest →", PROCESS_MANIFEST)


if __name__ == "__main__":
    main()