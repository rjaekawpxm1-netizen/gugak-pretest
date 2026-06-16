# -*- coding: utf-8 -*-
"""화면 2: 메타데이터 품질진단 — 현행 개방 데이터의 품질 문제를 자동 진단·시각화."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.express as px

from src.loaders import load_vr, load_exhibit
from src.diagnose_metadata import diagnose_metadata

st.set_page_config(page_title="메타데이터 품질진단", layout="wide")
st.title("2. 메타데이터 품질진단")
st.caption("현행 개방 데이터(VR·온라인전시)의 완전성·표준성·URL·컬럼매핑을 자동 전수진단")

vr = load_vr()
ex = load_exhibit()
findings, summary = diagnose_metadata(vr, ex)
errors = findings[findings["건수"] > 0].sort_values("비율%", ascending=False)

k = st.columns(4)
k[0].metric("총 진단 항목", f"{summary['총_진단항목']}개")
k[1].metric("오류 발견 항목", f"{summary['오류_발견_항목']}개", delta_color="inverse")
k[2].metric("100% 누락 항목", f"{int((findings['비율%'] == 100).sum())}개", delta_color="inverse")
k[3].metric("점검 데이터", f"{len(vr) + len(ex)}건")

st.divider()

st.subheader("진단항목별 오류율")
fig = px.bar(errors, x="비율%", y="진단항목", orientation="h",
             color="데이터셋", text="건수",
             hover_data={"개선방안": True, "비율%": ":.1f"})
fig.update_layout(yaxis=dict(autorange="reversed"), height=460,
                  xaxis_title="오류율(%)", yaxis_title="")
st.plotly_chart(fig, use_container_width=True)

st.subheader("오류 유형별 개선방안")
st.caption("이 표가 제안서/PPT의 '품질개선 과제' 슬라이드로 바로 연결된다.")
st.dataframe(
    errors[["데이터셋", "진단항목", "건수", "비율%", "개선방안"]],
    hide_index=True, use_container_width=True,
)

with st.expander("전체 진단 항목 보기 (정상 포함)"):
    st.dataframe(findings, hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "현행 개방 데이터조차 전시기간 100%·날짜필드 100% 오염, 썸네일 URL 93.8% 결합오류 등 "
    "다수 결함이 존재한다. 신규 개방 데이터는 본 진단 규칙을 전수 적용하여 사전 차단한다."
)