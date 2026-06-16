# -*- coding: utf-8 -*-
"""원본 파일 로딩만 담당. 진단·가공은 하지 않는다."""

from pathlib import Path
import pandas as pd
import soundfile as sf

from .config import ARCHIVE_LISTS, AUDIO_KOGL1, AUDIO_KOGL4, AUDIO_META

# 표준 파일명 (받은 파일을 이 이름으로 두면 됨)
# xlsx 는 내부가 zip 이라 전송 중 깨지기 쉬워, 둘 다 CSV(UTF-8 BOM)로 통일한다.
VR_FILE = "vr_list.csv"
EXHIBIT_FILE = "online_exhibit.csv"


def load_vr(path: Path | None = None) -> pd.DataFrame:
    """VR 영상 목록 CSV (UTF-8 BOM)."""
    return pd.read_csv(path or (ARCHIVE_LISTS / VR_FILE), encoding="utf-8-sig")


def load_exhibit(path: Path | None = None) -> pd.DataFrame:
    """온라인 전시 목록 CSV (UTF-8 BOM)."""
    return pd.read_csv(path or (ARCHIVE_LISTS / EXHIBIT_FILE), encoding="utf-8-sig")


def list_audio(kogl: int = 1) -> list[Path]:
    """공공누리 유형별 음원 파일 목록. 1유형=가공OK, 4유형=분석만."""
    folder = AUDIO_KOGL1 if kogl == 1 else AUDIO_KOGL4
    if not folder.exists():
        return []
    return sorted([*folder.glob("*.wav"), *folder.glob("*.mp3")])


def load_audio(path: Path):
    """음원 1건 로딩 → (mono 신호, 샘플레이트, soundfile.info)."""
    info = sf.info(str(path))
    data, sr = sf.read(str(path))
    mono = data.mean(axis=1) if data.ndim > 1 else data
    return mono, sr, info


def load_audio_metadata(path: Path | None = None) -> pd.DataFrame:
    """build_audio_metadata.py 가 만든 음원 메타데이터 CSV."""
    return pd.read_csv(path or AUDIO_META, encoding="utf-8-sig")