# -*- coding: utf-8 -*-
"""화면 1: 데이터 현황 — 국악 데이터 구조를 이해했음을 보여주는 요약 화면."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 프로젝트 루트 보장

import streamlit as st
import pandas as pd
import plotly.express as px

from src.loaders import load_vr, load_exhibit, list_audio
from src.diagnose_metadata import diagnose_metadata

st.set_page_config(page_title="데이터 현황", layout="wide")
st.title("1. 데이터 현황")
st.caption("현행 개방 데이터(VR·온라인전시) + 음원 샘플의 보유현황과 진단 요약")

# ---------------- 데이터 로딩 ----------------
vr = load_vr()
ex = load_exhibit()
audio = list_audio(kogl=1)
findings, summary = diagnose_metadata(vr, ex)

# ---------------- KPI 카드 ----------------
k = st.columns(6)
k[0].metric("VR 영상", f"{len(vr)}건")
k[1].metric("온라인 전시", f"{len(ex)}건")
k[2].metric("음원 샘플(1유형)", f"{len(audio)}건")
k[3].metric("목록 데이터 합계", f"{len(vr) + len(ex)}건")
k[4].metric("진단 항목", f"{summary['총_진단항목']}개")
k[5].metric("오류 발견 항목", f"{summary['오류_발견_항목']}개", delta_color="inverse")

st.divider()

# ---------------- 차트 ----------------
col1, col2 = st.columns(2)

# 1) 데이터 유형별 건수
with col1:
    st.subheader("데이터 유형별 건수")
    type_df = pd.DataFrame({
        "유형": ["VR 영상", "온라인 전시", "음원 샘플"],
        "건수": [len(vr), len(ex), len(audio)],
    })
    fig = px.bar(type_df, x="유형", y="건수", text="건수", color="유형")
    fig.update_layout(showlegend=False, height=360)
    st.plotly_chart(fig, use_container_width=True)

# 2) 온라인 전시 조회수 Top 10
with col2:
    st.subheader("온라인 전시 조회수 Top 10")
    top = (ex[["제목", "viewcnt"]]
           .dropna(subset=["viewcnt"])
           .sort_values("viewcnt", ascending=False)
           .head(10))
    fig2 = px.bar(top, x="viewcnt", y="제목", orientation="h", text="viewcnt")
    fig2.update_layout(yaxis=dict(autorange="reversed"), height=360,
                       xaxis_title="조회수", yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)

# 3) 전시 키워드 빈도 Top 15
st.subheader("온라인 전시 키워드 빈도 (검색·분류 설계 근거)")
kw = (ex["전시키워드"].dropna()
      .str.split(",").explode().str.strip())
kw = kw[kw != ""]
if len(kw):
    kw_df = kw.value_counts().head(15).reset_index()
    kw_df.columns = ["키워드", "빈도"]
    fig3 = px.bar(kw_df, x="빈도", y="키워드", orientation="h", text="빈도")
    fig3.update_layout(yaxis=dict(autorange="reversed"), height=420, yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.write("키워드 데이터 없음")

st.divider()
st.caption(
    "VR 목록은 결측이 거의 없어 즉시 개방 가능하나, 온라인 전시는 "
    f"전시기간 100%·키워드 62.5% 누락 등 보완이 필요하다. "
    "상세 진단은 '화면 2: 메타데이터 품질진단' 참조."
)