# -*- coding: utf-8 -*-
"""Streamlit 진입점(홈). 실행: streamlit run app.py"""

import streamlit as st

st.set_page_config(page_title="국악 데이터 개방·AI학습 적합성 진단", layout="wide")

st.title("국악 데이터 개방·AI학습 적합성 사전진단")
st.caption("파일 품질 · 메타데이터 표준성 · 라벨링 준비도 · API 제공 적합성 사전진단")

st.markdown(
    """
    본 도구는 **실제 공개 데이터**를 기반으로 한 사전진단 프로토타입입니다. (가짜 데이터 미사용)

    - **음원 (사업 핵심 대상)**: 국악기 디지털음원 단음·악구·확장 (공공누리 1유형)
    - **아카이브 목록 (현행 개방 데이터, 보조)**: VR 영상 목록 · 온라인 전시 목록
      → RFP "데이터 개방 현황"에 해당. *기존 개방 데이터의 품질 문제*를 진단하는 근거로 사용.

    왼쪽 사이드바에서 화면을 선택하세요.
    """
)

st.info("화면 1: 데이터 현황 → 사이드바 'data overview'를 클릭하세요.")