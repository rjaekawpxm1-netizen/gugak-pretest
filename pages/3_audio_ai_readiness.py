# -*- coding: utf-8 -*-
"""화면 3: 음원 / AI 학습 적합성 — 코퍼스 불균질성 + 악기군별 특성 + 세그먼트 품질."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf

from src.loaders import load_audio_metadata
from src.diagnose_audio import measure, assess
from src.config import (AUDIO_KOGL1, PROCESS_MANIFEST, SILENCE_DB,
                        AUDIO_LEN_MIN, AUDIO_LEN_MAX)

st.set_page_config(page_title="음원 AI 학습 적합성", layout="wide")
st.title("3. 음원 / AI 학습 적합성")
st.caption("원천 음원 코퍼스의 불균질성과 악기군별 특성을 진단하여 AI 가공 범위를 도출")

# ── 데이터 로딩 ──────────────────────────────────────────────
try:
    meta = load_audio_metadata()
except FileNotFoundError:
    st.warning("먼저 `python build_audio_metadata.py` 를 실행하세요.")
    st.stop()

meta["instrument_group"] = meta["instrument_group"].fillna("미상").replace("", "미상")

mf_exists = PROCESS_MANIFEST.exists()
if mf_exists:
    mf = pd.read_csv(PROCESS_MANIFEST, encoding="utf-8-sig")
    mf2 = mf.merge(meta[["file", "instrument_group"]],
                   left_on="source", right_on="file", how="left")
    mf2["instrument_group"] = mf2["instrument_group"].fillna("미상")

# ── 코퍼스 KPI ───────────────────────────────────────────────
k = st.columns(6)
k[0].metric("음원 수", f"{len(meta)}건")
k[1].metric("샘플레이트 종류", f"{meta['samplerate'].nunique()}종")
k[2].metric("채널 종류", f"{meta['channels'].nunique()}종")
k[3].metric("길이 범위", f"{meta['duration_sec'].min():.0f}~{meta['duration_sec'].max():.0f}s")
k[4].metric("음량 편차", f"{meta['rms_db'].max()-meta['rms_db'].min():.0f}dB")
k[5].metric("길이 적합 비율",
            f"{meta['duration_sec'].between(AUDIO_LEN_MIN,AUDIO_LEN_MAX).mean()*100:.0f}%")

st.info("샘플레이트·채널·길이·음량이 모두 혼재 → 표준화 가공 없이는 AI 학습에 그대로 쓸 수 없음.")
st.divider()

# ── 탭 구성 ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 코퍼스 분포", "🎵 악기군별 특성", "✂️ 세그먼트 품질"])

# ====================================================== 탭1: 코퍼스 분포
with tab1:
    r1, r2 = st.columns(2)
    with r1:
        st.subheader("길이 분포")
        fig = px.histogram(meta, x="duration_sec", nbins=20)
        fig.add_vrect(x0=AUDIO_LEN_MIN, x1=AUDIO_LEN_MAX,
                      fillcolor="green", opacity=0.12, annotation_text="권장 15~30s")
        fig.update_layout(height=300, xaxis_title="길이(초)", yaxis_title="건수")
        st.plotly_chart(fig, use_container_width=True)
    with r2:
        st.subheader("샘플레이트 분포")
        sr = meta["samplerate"].astype(str).value_counts().reset_index()
        sr.columns = ["샘플레이트", "건수"]
        fig = px.bar(sr, x="샘플레이트", y="건수", text="건수", color="샘플레이트")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    r3, r4 = st.columns(2)
    with r3:
        st.subheader("음량(RMS) 분포")
        fig = px.histogram(meta, x="rms_db", nbins=20)
        fig.update_layout(height=300, xaxis_title="RMS(dBFS)", yaxis_title="건수")
        st.plotly_chart(fig, use_container_width=True)
    with r4:
        st.subheader("무음 비율 분포")
        fig = px.histogram(meta, x="low_energy_ratio", nbins=20)
        fig.update_layout(height=300, xaxis_title="저에너지 비율(%)", yaxis_title="건수")
        st.plotly_chart(fig, use_container_width=True)

# ====================================================== 탭2: 악기군별 특성
with tab2:
    st.subheader("악기군별 평균 특성")
    st.caption("악기마다 무음 비율·음량·길이 패턴이 다르다 → 악기 특성별 가공 방식 차등 적용 필요.")

    g = meta.groupby("instrument_group").agg(
        건수=("file", "count"),
        평균길이_s=("duration_sec", "mean"),
        평균음량_dBFS=("rms_db", "mean"),
        평균무음비율_pct=("low_energy_ratio", "mean"),
    ).round(1).reset_index()
    st.dataframe(g, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(g, x="instrument_group", y="평균무음비율_pct",
                     text="평균무음비율_pct", color="instrument_group",
                     title="악기군별 평균 무음 비율(%)")
        fig.update_layout(height=320, showlegend=False,
                          xaxis_title="악기군", yaxis_title="저에너지 비율(%)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(g, x="instrument_group", y="평균음량_dBFS",
                     text="평균음량_dBFS", color="instrument_group",
                     title="악기군별 평균 음량(RMS dBFS)")
        fig.update_layout(height=320, showlegend=False,
                          xaxis_title="악기군", yaxis_title="RMS(dBFS)")
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "타악기(장구)는 타격음 후 긴 여음으로 무음비율이 높고, "
        "현악기(거문고·가야금)는 발현 후 감쇠로 무음비율이 높다. "
        "관악기(대금)는 지속음이 많아 상대적으로 낮다. "
        "→ 고정창(fixed-window) 세그먼트보다 악기 특성 반영 무음인지 세그먼트가 필요하다."
    )

# ====================================================== 탭3: 세그먼트 품질
with tab3:
    if not mf_exists:
        st.warning("`python process_audio_poc.py` 를 먼저 실행하세요.")
    else:
        st.subheader("악기군별 세그먼트 적합성")
        st.caption("가공 후 세그먼트의 AI 학습 적합 비율 — 전문가 검수 우선순위 도출 근거.")

        fit  = mf2[mf2["dur_sec"].between(15,30)].groupby("instrument_group").size().rename("학습적합")
        short = mf2[(mf2["dur_sec"]>=3)&(mf2["dur_sec"]<15)].groupby("instrument_group").size().rename("보완필요")
        total = mf2.groupby("instrument_group").size().rename("세그먼트수")
        seg_g = pd.concat([total, fit, short], axis=1).fillna(0).astype(int).reset_index()
        seg_g["적합률%"] = (seg_g["학습적합"] / seg_g["세그먼트수"] * 100).round(1)
        st.dataframe(seg_g, hide_index=True, use_container_width=True)

        fig = px.bar(seg_g, x="instrument_group",
                     y=["학습적합", "보완필요"],
                     barmode="stack", text_auto=True,
                     color_discrete_map={"학습적합": "#4CAF50", "보완필요": "#FF9800"},
                     title="악기군별 세그먼트 적합성 (누적)")
        fig.update_layout(height=340, xaxis_title="악기군", yaxis_title="세그먼트 수",
                          legend_title="등급")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "보완필요(3~14s) 세그먼트가 많은 악기군이 전문가 검수 1순위다. "
            "단음·짧은 악구는 인접 조각과 병합하거나 별도 라벨 체계를 적용한다."
        )

st.divider()

# ── 파일별 상세 (포락선) ─────────────────────────────────────
st.divider()
st.subheader("파일별 상세 진단")
st.caption("에너지 포락선과 AI 적합성 항목별 판정. 원본 음원이 로컬에 있을 때 이용 가능.")

# metadata CSV에서 측정값을 그대로 표시 (원본 파일 없이도 동작)
sel = st.selectbox("음원 선택", meta["file"].tolist())
row = meta[meta["file"] == sel].iloc[0]

m = st.columns(5)
m[0].metric("길이", f"{row['duration_sec']}s")
m[1].metric("샘플레이트", f"{row['samplerate']}Hz")
m[2].metric("채널", f"{row['channels']}ch")
m[3].metric("RMS", f"{row['rms_db']}dBFS")
m[4].metric("저에너지", f"{row['low_energy_ratio']}%")

st.info(f"AI 학습 적합성 판정: **{row['ai_grade']}**")

# 에너지 포락선은 원본 파일 있을 때만
path = AUDIO_KOGL1 / sel
if path.exists():
    props, env_t, env_db = measure(path)
    checks, grade = assess(props)
    st.dataframe(pd.DataFrame(checks, columns=["항목", "판정", "사유"]),
                 hide_index=True, use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=env_t, y=env_db, mode="lines", name="에너지(dBFS)"))
    fig.add_hline(y=SILENCE_DB, line_dash="dash", line_color="red",
                  annotation_text=f"무음 임계 {SILENCE_DB}dBFS")
    fig.update_layout(height=340, xaxis_title="시간(초)", yaxis_title="dBFS", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("에너지 포락선 그래프는 로컬 실행 시 표시됩니다.")