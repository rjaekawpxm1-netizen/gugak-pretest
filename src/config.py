# -*- coding: utf-8 -*-
"""중앙 설정값. 진단 기준을 바꾸려면 여기만 수정한다."""

from pathlib import Path

# ----- 경로 -----
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
AUDIO_KOGL1 = DATA_RAW / "audio_kogl1"      # 공공누리 1유형 = 가공 허용
AUDIO_KOGL4 = DATA_RAW / "audio_kogl4"      # 4유형 = 분석만, 가공 금지
ARCHIVE_LISTS = DATA_RAW / "archive_lists"  # vr_list.csv, online_exhibit.csv
PROCESSED = ROOT / "data" / "processed"
AUDIO_META = PROCESSED / "audio_metadata.csv"  # build_audio_metadata.py 산출
SEGMENTS_DIR = PROCESSED / "segments"          # 가공된 세그먼트 출력
PROCESS_MANIFEST = PROCESSED / "processing_manifest.csv"

# ----- AI 학습용 가공 기준 (ADR-004) -----
PROCESS_SR = 22050          # 통일 샘플레이트
PROCESS_TARGET_RMS = -20.0  # 음량 정규화 목표(dBFS)
PROCESS_PEAK_CEIL = -1.0    # 정규화 후 피크 상한(dBFS, 클리핑 방지)

# ----- 음원 진단 기준 (RFP: 평균 15~30초) -----
AUDIO_LEN_MIN = 15      # 초
AUDIO_LEN_MAX = 30      # 초
SILENCE_DB = -50        # 무음 판정 임계 (dBFS)
SILENCE_RATIO_WARN = 40 # 저에너지 비율 경고 임계 (%)

# ----- AI 학습용 권장 포맷 -----
AI_TARGET_CHANNELS = 1          # mono
AI_TARGET_SR = [16000, 22050]   # 권장 샘플레이트 후보

# ----- 공공누리 유형 (가공 가능 여부) -----
KOGL_MODIFY_OK = {1, 2}   # 변경 허용
KOGL_MODIFY_NO = {3, 4}   # 변경금지 → 가공 PoC 금지

# ----- 메타데이터 필수 항목 (개방DB/API 제공 전제) -----
REQUIRED_FIELDS_EXHIBIT = ["제목", "내용", "전시키워드", "disp_sdate", "disp_edate"]