# -*- coding: utf-8 -*-
"""화면 4: 가공 전후 비교 — ADR-004 결과 시각화 (제안서 마지막 캡처)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import soundfile as sf

from src.config import PROCESS_MANIFEST, SEGMENTS_DIR, AUDIO_KOGL1, PROCESS_SR

st.set_page_config(page_title="가공 전후 비교", layout="wide")
st.title("4. AI 가공 전후 비교 (ADR-004)")
st.caption("원본 음원(불균질) → 22050Hz·mono·음량정규화·무음세그먼트(15~30초) 표준 가공 결과")

if not PROCESS_MANIFEST.exists():
    st.warning("먼저 `python process_audio_poc.py` 를 실행하세요.")
    st.stop()

mf = pd.read_csv(PROCESS_MANIFEST, encoding="utf-8-sig")

# ---------------- 코퍼스 KPI ----------------
total_src = mf["source"].nunique()
total_seg = len(mf)
fit = mf[mf["dur_sec"].between(15, 30)]
short = mf[(mf["dur_sec"] >= 3) & (mf["dur_sec"] < 15)]
sr_unified = (mf["out_sr"] == PROCESS_SR).all()
ch_unified = (mf["out_ch"] == 1).all()

k = st.columns(6)
k[0].metric("원본 음원", f"{total_src}개")
k[1].metric("세그먼트", f"{total_seg}개")
k[2].metric("학습 적합(15~30s)", f"{len(fit)}개")
k[3].metric("보완(3~14s)", f"{len(short)}개")
k[4].metric("샘플레이트 통일", "✅ 22050Hz" if sr_unified else "❌")
k[5].metric("채널 통일", "✅ mono" if ch_unified else "❌")

# ---------------- 가공 전후 속성 비교 표 ----------------
st.divider()
st.subheader("가공 전후 속성 비교")
comp = pd.DataFrame({
    "항목": ["샘플레이트", "채널", "길이", "음량(RMS)"],
    "가공 전(원본)": [
        f"{sorted(mf['src_sr'].unique().tolist())} (3종 혼재)",
        f"mono+stereo 혼재",
        f"{mf['dur_sec'].min():.1f}~{mf['dur_sec'].max():.1f}s",
        f"{mf['src_rms_db'].min():.0f}~{mf['src_rms_db'].max():.0f}dBFS ({mf['src_rms_db'].max()-mf['src_rms_db'].min():.0f}dB 편차)",
    ],
    "가공 후(세그먼트)": [
        "22050Hz (통일)",
        "mono (통일)",
        f"{mf['dur_sec'].min():.1f}~{mf['dur_sec'].max():.1f}s",
        f"약 -20dBFS (정규화)",
    ],
    "가공 방법": ["리샘플(soxr)", "다운믹스", "무음인지 세그먼트", "RMS 정규화"],
})
st.dataframe(comp, hide_index=True, use_container_width=True)

# ---------------- 세그먼트 길이 분포 ----------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("세그먼트 길이 분포")
    fig = px.histogram(mf, x="dur_sec", nbins=20, color_discrete_sequence=["#2196F3"])
    fig.add_vrect(x0=15, x1=30, fillcolor="green", opacity=0.12,
                  annotation_text="15~30s 학습적합")
    fig.update_layout(height=320, xaxis_title="길이(초)", yaxis_title="조각수")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("AI 학습 적합성 분류")
    grade = pd.DataFrame({
        "등급": ["학습 적합(15~30s)", "보완필요(3~14s)", "제외(<3s)"],
        "건수": [len(fit), len(short), total_seg - len(fit) - len(short)],
        "기준": ["즉시 학습 가능", "단음·짧은악구, 병합 검토", "무음·불량"],
    })
    fig2 = px.bar(grade, x="등급", y="건수", text="건수", color="등급",
                  color_discrete_map={
                      "학습 적합(15~30s)": "#4CAF50",
                      "보완필요(3~14s)": "#FF9800",
                      "제외(<3s)": "#F44336",
                  })
    fig2.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- 파일별 Before/After 파형 ----------------
st.divider()
st.subheader("파일별 Before / After 파형")
sources = sorted(mf["source"].unique().tolist())
sel = st.selectbox("원본 음원 선택", sources)

src_path = AUDIO_KOGL1 / sel
seg_dir = SEGMENTS_DIR / Path(sel).stem
seg_files = sorted(seg_dir.glob("*.wav")) if seg_dir.exists() else []

if not src_path.exists():
    st.warning(f"원본 파일 없음: {src_path}")
    st.stop()

# Before: 원본 (최대 30초만 표시)
raw, sr_raw = sf.read(str(src_path))
raw_mono = raw.mean(axis=1) if raw.ndim > 1 else raw
preview_len = min(len(raw_mono), sr_raw * 30)
t_raw = np.arange(preview_len) / sr_raw
rms_before = round(float(20 * np.log10(np.sqrt(np.mean(raw_mono ** 2)) + 1e-12)), 1)

# After: 첫 번째 세그먼트
if seg_files:
    seg, sr_seg = sf.read(str(seg_files[0]))
    t_seg = np.arange(len(seg)) / sr_seg
    rms_after = round(float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)), 1)
else:
    seg = None

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Before** — 원본 앞 30초  \n"
                f"`{sr_raw}Hz / {'stereo' if raw.ndim > 1 else 'mono'} / {rms_before}dBFS`")
    fig3 = go.Figure(go.Scatter(x=t_raw, y=raw_mono[:preview_len],
                                mode="lines", line=dict(width=0.8, color="#1565C0")))
    fig3.update_layout(height=260, xaxis_title="시간(초)", yaxis_title="진폭",
                       margin=dict(t=10, b=40))
    st.plotly_chart(fig3, use_container_width=True)
with c2:
    if seg is not None:
        st.markdown(f"**After** — seg01 ({round(len(seg)/sr_seg,1)}s)  \n"
                    f"`{sr_seg}Hz / mono / {rms_after}dBFS`")
        fig4 = go.Figure(go.Scatter(x=t_seg, y=seg,
                                    mode="lines", line=dict(width=0.8, color="#2E7D32")))
        fig4.update_layout(height=260, xaxis_title="시간(초)", yaxis_title="진폭",
                           margin=dict(t=10, b=40))
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("세그먼트 없음 (짧거나 무음 파일)")

# 세그먼트 목록
if seg_files:
    seg_info = mf[mf["source"] == sel][["segment", "start_sec", "dur_sec", "out_rms_db"]]
    st.dataframe(seg_info.rename(columns={
        "segment": "세그먼트", "start_sec": "시작(s)", "dur_sec": "길이(s)", "out_rms_db": "RMS(dBFS)"}),
        hide_index=True, use_container_width=True)

st.divider()
st.caption(
    f"원본 {total_src}개(SR 3종·채널 혼재·길이 4~195s·음량 편차 ~40dB) → "
    f"세그먼트 {total_seg}개(22050Hz·mono·목표 -20dBFS·학습적합 {len(fit)}개). "
    "모든 가공은 공공누리 1유형(변경허용) 데이터에만 적용하였다."
)