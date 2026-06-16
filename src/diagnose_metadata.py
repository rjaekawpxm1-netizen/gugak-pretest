# -*- coding: utf-8 -*-
"""아카이브 목록(VR / 온라인전시) 메타데이터 품질진단.

화면(pages/)에서 diagnose_metadata(vr, ex)를 호출해 결과 표를 렌더링한다.
모든 수치는 실제 데이터에서 계산된다.
"""

import pandas as pd

from .config import REQUIRED_FIELDS_EXHIBIT

_PLANS = {
    "내용": "전시 설명 보완",
    "전시키워드": "검색·분류 키워드 보완",
    "disp_sdate": "전시 시작일 관리기준 수립",
    "disp_edate": "전시 종료일 관리기준 수립",
    "제목": "필수값 보완",
}


def _row(dataset, item, count, total, plan):
    return {
        "데이터셋": dataset,
        "진단항목": item,
        "건수": int(count),
        "비율%": round(count / total * 100, 1) if total else 0.0,
        "개선방안": plan,
    }


def diagnose_vr(vr: pd.DataFrame) -> pd.DataFrame:
    """VR 목록: 완전성 위주 (대체로 깨끗 → 즉시 개방 가능 근거)."""
    n = len(vr)
    rows = [_row("VR목록", f"{col} 누락", vr[col].isna().sum(), n,
                 "—" if vr[col].isna().sum() == 0 else "원천 확인")
            for col in vr.columns]
    return pd.DataFrame(rows)


def diagnose_exhibit(ex: pd.DataFrame) -> pd.DataFrame:
    """온라인 전시: 완전성 + 표준성 + URL 품질 + 컬럼매핑 진단."""
    n = len(ex)
    rows = []

    # 1) 필수값 완전성
    for col in REQUIRED_FIELDS_EXHIBIT:
        if col in ex.columns:
            miss = ex[col].isna().sum()
            rows.append(_row("온라인전시", f"{col} 누락", miss, n, _PLANS.get(col, "보완")))

    # 2) 전시링크: 누락이 아니라 컬럼 매핑 문제 (상세 URL은 마지막 컬럼에 완비)
    if "전시 링크" in ex.columns:
        miss = ex["전시 링크"].isna().sum()
        detail = ex.columns[-1]
        full = ex[detail].notna().sum()
        rows.append(_row("온라인전시", "전시링크 컬럼 결측(상세URL 별도 컬럼 존재)", miss, n,
                         f"컬럼 매핑 정정 ({detail}={full}/{n} 완비)"))

    # 3) 날짜 필드 오염: reg_date/mod_date 가 시각(HH:MM:SS) 형태
    for col in ["reg_date", "mod_date"]:
        if col in ex.columns:
            t = ex[col].astype(str).str.match(r"^\d{1,2}:\d{2}:\d{2}")
            if t.mean() > 0.5:
                rows.append(_row("온라인전시", f"{col} 날짜필드에 시각값 오염",
                                 t.sum(), n, "날짜 도메인 규칙 적용·재수집"))

    # 4) 썸네일 URL 결합 오류 (확장자 뒤에 경로가 또 붙음)
    if "썸네일 링크" in ex.columns:
        bad = ex["썸네일 링크"].astype(str).str.contains(
            r"\.(?:jpg|jpeg|png)/", case=False, regex=True)
        rows.append(_row("온라인전시", "썸네일 URL 결합 오류(두 경로 연결)", bad.sum(), n,
                         "thumb01/thumb02 컬럼 분리, 중복슬래시 제거"))

    # 5) 내부 저장경로 노출
    if "img_path" in ex.columns:
        internal = ex["img_path"].astype(str).str.startswith(
            ("/Odrive", "/newarchive", "/Edrive"))
        rows.append(_row("온라인전시", "내부 저장경로 노출(img_path)", internal.sum(), n,
                         "공개 URL과 내부경로 분리"))

    return pd.DataFrame(rows)


def diagnose_metadata(vr: pd.DataFrame, ex: pd.DataFrame):
    """두 목록 통합 진단 → (진단표, 요약 dict)."""
    findings = pd.concat([diagnose_vr(vr), diagnose_exhibit(ex)], ignore_index=True)
    summary = {
        "VR목록_건수": len(vr),
        "온라인전시_건수": len(ex),
        "총_진단항목": len(findings),
        "오류_발견_항목": int((findings["건수"] > 0).sum()),
    }
    return findings, summary
