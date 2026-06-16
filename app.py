# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf

from src.loaders import load_vr, load_exhibit, load_audio_metadata, list_audio
from src.diagnose_metadata import diagnose_metadata
from src.diagnose_audio import measure, assess
from src.config import (AUDIO_KOGL1, PROCESS_MANIFEST, SEGMENTS_DIR,
                        SILENCE_DB, AUDIO_LEN_MIN, AUDIO_LEN_MAX, PROCESS_SR)

# ── 전역 설정 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="국악 데이터 개방·AI학습 적합성 진단",
    page_icon="🎵", layout="wide",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #f8f9fa; }
.req-box {
    background:#f0f4ff; border-left:4px solid #3b5bdb;
    padding:.75rem 1rem; border-radius:4px;
    margin-bottom:1rem; font-size:.9rem;
}
.term-box {
    background:#f8f9fa; border:1px solid #dee2e6;
    padding:.7rem 1rem; border-radius:6px;
    margin-top:.5rem; font-size:.85rem; color:#495057;
}
.find-card {
    background:#fff; border:1px solid #dee2e6;
    border-radius:8px; padding:1rem 1.2rem;
}
.find-card h4 { margin:0 0 .4rem 0; font-size:1rem; }
.find-card p  { margin:0; font-size:.85rem; color:#495057; }
</style>
""", unsafe_allow_html=True)

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎵 국립국악원")
    st.markdown("**국악 데이터 개방·AI학습**  \n**적합성 사전진단 시스템**")
    st.caption("제안서 제출 전, 실제 공개 데이터를  \n직접 분석한 결과입니다.")
    st.divider()
    menu = st.radio("", [
        "📊 종합 대시보드",
        "📁 데이터 현황",
        "🔍 메타데이터 품질진단",
        "🎵 음원 AI 학습 적합성",
        "✂️ AI 가공 결과",
    ], label_visibility="collapsed")

# ── 공통 데이터 로딩 (캐시) ────────────────────────────────────
@st.cache_data
def get_vr():        return load_vr()
@st.cache_data
def get_exhibit():   return load_exhibit()
@st.cache_data
def get_meta():
    df = load_audio_metadata()
    df["instrument_group"] = df["instrument_group"].fillna("미상").replace("","미상")
    return df
@st.cache_data
def get_manifest():
    if not PROCESS_MANIFEST.exists(): return None
    return pd.read_csv(PROCESS_MANIFEST, encoding="utf-8-sig")

vr   = get_vr()
ex   = get_exhibit()
meta = get_meta()
mf   = get_manifest()

findings, summary = diagnose_metadata(vr, ex)
errors = findings[findings["건수"]>0].sort_values("비율%", ascending=False)

if mf is not None:
    mf2 = mf.merge(meta[["file","instrument_group"]], left_on="source", right_on="file", how="left")
    mf2["instrument_group"] = mf2["instrument_group"].fillna("미상")
    fit_cnt   = int(mf["dur_sec"].between(AUDIO_LEN_MIN, AUDIO_LEN_MAX).sum())
    short_cnt = int(((mf["dur_sec"]>=3)&(mf["dur_sec"]<15)).sum())
else:
    mf2 = None; fit_cnt = 0; short_cnt = 0

# ════════════════════════════════════════════════════════════════
# 0. 종합 대시보드
# ════════════════════════════════════════════════════════════════
if menu == "📊 종합 대시보드":
    st.markdown("📊 종합 대시보드")
    st.title("AI 데이터 사전 분석 종합 현황")
    st.caption("제안서 제출 전, 실제 공개 데이터를 직접 분석한 결과입니다.")

    st.markdown("""<div class="req-box">
    🗂️ <b>이 대시보드는 무엇인가요?</b><br>
    국립국악원이 공개한 음원·아카이브 데이터를 직접 내려받아 품질·AI 학습 적합성을 분석했습니다.
    아래 숫자와 발견은 모두 '하겠습니다'가 아니라 <b>이미 해본 결과</b>입니다.
    </div>""", unsafe_allow_html=True)

    k = st.columns(5)
    k[0].metric("분석 데이터", f"{len(vr)+len(ex)+len(meta)}건")
    k[1].metric("실제 음원", f"{len(meta)}건")
    k[2].metric("메타데이터 오류", f"{summary['오류_발견_항목']}종")
    k[3].metric("AI 가공 세그먼트", f"{len(mf) if mf is not None else 0}개")
    k[4].metric("학습 적합(15~30s)", f"{fit_cnt}개")

    st.divider()
    st.subheader("한눈에 보는 핵심 발견")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.error("🔴 즉시 개선 필요")
        st.markdown("""<div class="find-card">
        <h4>현행 개방 메타데이터 결함 9종</h4>
        <p>전시기간 100% 누락, 날짜필드 100% 오염,
        썸네일 URL 93.8% 결합오류, 내부경로 50% 노출.
        목록만 본 게 아니라 실제 필드값을 전수 점검했습니다.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.warning("🟡 가공 필수")
        rms_range = round(meta["rms_db"].max()-meta["rms_db"].min(), 0)
        st.markdown(f"""<div class="find-card">
        <h4>음원 불균질성 5종 확인</h4>
        <p>샘플레이트 {meta['samplerate'].nunique()}종 혼재, 음량 편차 {rms_range:.0f}dB,
        길이 {meta['duration_sec'].min():.0f}~{meta['duration_sec'].max():.0f}초,
        모노·스테레오 혼재. 표준화 없이 AI 학습 불가.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.success("🟢 가공 완료")
        seg_n = len(mf) if mf is not None else 0
        st.markdown(f"""<div class="find-card">
        <h4>실제 가공 파이프라인 검증</h4>
        <p>원본 {len(meta)}개 → 세그먼트 {seg_n}개 산출.
        22050Hz·mono·음량정규화 통일 완료.
        학습 적합(15~30s) {fit_cnt}개 확보.</p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("분석 결과 요약")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**메타데이터 오류율 Top 5**")
        top5 = errors.head(5)
        fig = px.bar(top5, x="비율%", y="진단항목", orientation="h",
                     color="비율%", color_continuous_scale="Reds", text="건수")
        fig.update_layout(height=280, showlegend=False, margin=dict(t=0,b=0),
                          coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**음원 샘플레이트 분포**")
        sr = meta["samplerate"].astype(str).value_counts().reset_index()
        sr.columns = ["샘플레이트","건수"]
        fig2 = px.pie(sr, names="샘플레이트", values="건수",
                      color_discrete_sequence=["#3b5bdb","#74c0fc","#a5d8ff"])
        fig2.update_layout(height=280, margin=dict(t=0,b=0))
        st.plotly_chart(fig2, use_container_width=True)
    with col3:
        st.markdown("**세그먼트 AI 학습 적합성**")
        if mf is not None:
            exc = len(mf)-fit_cnt-short_cnt
            gdf = pd.DataFrame({
                "등급":["학습 적합(15~30s)","보완필요(3~14s)","제외(<3s)"],
                "건수":[fit_cnt, short_cnt, exc]})
            fig3 = px.pie(gdf, names="등급", values="건수",
                          color_discrete_map={"학습 적합(15~30s)":"#40c057",
                                              "보완필요(3~14s)":"#fd7e14","제외(<3s)":"#fa5252"})
            fig3.update_layout(height=280, margin=dict(t=0,b=0))
            st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.markdown("""<div class="req-box">
    📋 <b>대응 요구사항</b>: OSR-001 (현황분석) · DQR-001~004 (품질진단·개선) · ADR-001~004 (AI데이터 가공) · ODR-001 (개방DB 분석)<br>
    본 사전분석 결과는 개방DB 설계(ODR), 오픈API 제공항목(APR), AI 라벨 체계(ADR), 품질개선 과제(DQR) 수립의 직접 입력값으로 활용합니다.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 1. 데이터 현황
# ════════════════════════════════════════════════════════════════
elif menu == "📁 데이터 현황":
    st.markdown("📁 데이터 현황")
    st.title("국악 데이터 보유현황을 직접 확인했습니다")
    st.markdown("""<div class="req-box">
    📋 <b>대응 요구사항: OSR-001 (업무 및 데이터 현황분석)</b><br>
    음원·아카이브 데이터의 보유현황, 파일 포맷, 분류체계, 개방 가능 여부를 분석하여
    개방DB 설계와 AI 학습용 데이터 가공 범위를 명확화합니다.
    </div>""", unsafe_allow_html=True)

    audio = list_audio(kogl=1)
    k = st.columns(5)
    k[0].metric("VR 영상 목록", f"{len(vr)}건")
    k[1].metric("온라인 전시 목록", f"{len(ex)}건")
    k[2].metric("음원 샘플(1유형)", f"{len(audio)}건")
    k[3].metric("목록 합계", f"{len(vr)+len(ex)}건")
    k[4].metric("오류 발견 항목", f"{summary['오류_발견_항목']}개")

    st.markdown("""<div class="term-box">
    💡 <b>공공누리 1유형이란?</b> 출처만 표시하면 자유롭게 이용·변경·가공할 수 있는 공개 라이선스입니다.
    음원 가공 PoC에 사용한 파일은 전부 이 유형입니다. 4유형(변경금지)은 분석만 가능합니다.
    </div>""", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("데이터 유형별 건수")
        st.caption("분석 대상 데이터를 유형별로 구분한 현황입니다.")
        tdf = pd.DataFrame({"유형":["VR 영상","온라인 전시","음원 샘플"],
                            "건수":[len(vr),len(ex),len(audio)]})
        fig = px.bar(tdf, x="유형", y="건수", text="건수", color="유형",
                     color_discrete_sequence=["#3b5bdb","#74c0fc","#a5d8ff"])
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("온라인 전시 조회수 Top 10")
        st.caption("조회수가 높은 전시가 API 우선 제공 후보입니다.")
        top = (ex[["제목","viewcnt"]].dropna(subset=["viewcnt"])
               .sort_values("viewcnt", ascending=False).head(10))
        fig2 = px.bar(top, x="viewcnt", y="제목", orientation="h", text="viewcnt",
                      color_discrete_sequence=["#3b5bdb"])
        fig2.update_layout(yaxis=dict(autorange="reversed"), height=320,
                           xaxis_title="조회수", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("온라인 전시 키워드 빈도")
    st.caption("키워드 빈도는 API 검색조건과 개방DB 분류체계 설계의 근거가 됩니다.")
    kw = ex["전시키워드"].dropna().str.split(",").explode().str.strip()
    kw = kw[kw != ""]
    if len(kw):
        kdf = kw.value_counts().head(15).reset_index()
        kdf.columns = ["키워드","빈도"]
        fig3 = px.bar(kdf, x="빈도", y="키워드", orientation="h", text="빈도",
                      color_discrete_sequence=["#3b5bdb"])
        fig3.update_layout(yaxis=dict(autorange="reversed"), height=360, yaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""<div class="term-box">
    💡 <b>이 화면이 제안서에서 말하는 것</b><br>
    단순히 "데이터가 있다"가 아니라, 실제 파일을 받아 건수·유형·키워드·조회수를 직접 확인했습니다.
    VR 목록은 결측이 없어 즉시 개방 가능하지만, 온라인 전시는 전시기간·키워드 등 보완이 필요합니다.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 2. 메타데이터 품질진단
# ════════════════════════════════════════════════════════════════
elif menu == "🔍 메타데이터 품질진단":
    st.markdown("🔍 메타데이터 품질진단")
    st.title("개방 데이터를 직접 열어 품질을 점검했습니다")
    st.markdown("""<div class="req-box">
    📋 <b>대응 요구사항: OSR-001 (현황분석) + DQR-001~004 (품질진단·개선계획·개선수행)</b><br>
    목록만 본 게 아니라 실제 CSV를 내려받아 결측·중복·이상값·URL 오류를 자동 전수 진단했습니다.
    현행 개방 데이터조차 이런 결함이 있다 → 신규 개방 데이터는 본 진단 규칙으로 사전 차단합니다.
    </div>""", unsafe_allow_html=True)

    k = st.columns(4)
    k[0].metric("총 진단 항목", f"{summary['총_진단항목']}개")
    k[1].metric("오류 발견 항목", f"{summary['오류_발견_항목']}개")
    k[2].metric("100% 누락 항목", f"{int((findings['비율%']==100).sum())}개")
    k[3].metric("점검 데이터", f"{len(vr)+len(ex)}건")

    st.markdown("""<div class="term-box">
    💡 <b>용어 설명</b><br>
    • <b>완전성(결측)</b>: 필수 항목이 비어있는 비율. 전시기간이 100% 비어있으면 날짜 기반 API 검색이 불가합니다.<br>
    • <b>URL 결합 오류</b>: 썸네일 이미지 주소 2개가 하나로 붙어있어 실제 이미지를 불러올 수 없는 상태.<br>
    • <b>내부 저장경로 노출</b>: 서버 내부 경로(/Odrive/...)가 공개 필드에 있어 보안 위험이 있는 상태.
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("진단항목별 오류율")
    st.caption("막대가 길수록 오류 비율이 높습니다. 빨간색 계열이 우선 개선 대상입니다.")
    fig = px.bar(errors, x="비율%", y="진단항목", orientation="h",
                 color="데이터셋", text="건수",
                 color_discrete_map={"VR목록":"#74c0fc","온라인전시":"#3b5bdb"},
                 hover_data={"개선방안":True,"비율%":":.1f"})
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420,
                      xaxis_title="오류율(%)", yaxis_title="", legend_title="데이터셋")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("오류 유형별 개선방안")
    st.caption("이 표가 제안서 '품질개선 과제' 슬라이드로 직접 연결됩니다.")
    st.dataframe(errors[["데이터셋","진단항목","건수","비율%","개선방안"]],
                 hide_index=True, use_container_width=True)

    with st.expander("전체 진단 항목 보기 (정상 포함)"):
        st.dataframe(findings, hide_index=True, use_container_width=True)

    st.markdown("""<div class="term-box">
    💡 <b>이 화면이 제안서에서 말하는 것</b><br>
    현행 개방 데이터조차 전시기간 100%·날짜필드 100% 오염·썸네일 URL 93.8% 결합오류 등
    다수 결함이 존재합니다. 신규 개방 데이터(음원 16,721건·이미지 1,685건)에는
    본 진단 규칙을 자동 전수 적용하여 이런 문제를 사전에 차단합니다.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 3. 음원 AI 학습 적합성
# ════════════════════════════════════════════════════════════════
elif menu == "🎵 음원 AI 학습 적합성":
    st.markdown("🎵 음원 AI 학습 적합성")
    st.title("음원 특성을 직접 측정해 AI 가공 범위를 도출했습니다")
    st.markdown("""<div class="req-box">
    📋 <b>대응 요구사항: ADR-001 (AI데이터 가공방안) · ADR-002 (AI데이터 설계) · ODR-001 (개방DB 분석)</b><br>
    음원 파일을 직접 내려받아 샘플레이트·채널·길이·음량·무음 비율을 측정했습니다.
    악기군마다 무음 패턴이 다르다는 사실을 데이터로 확인 → 국악 특화 가공 방식이 필요함을 입증합니다.
    </div>""", unsafe_allow_html=True)

    k = st.columns(6)
    k[0].metric("음원 수", f"{len(meta)}건")
    k[1].metric("샘플레이트 종류", f"{meta['samplerate'].nunique()}종")
    k[2].metric("채널 종류", f"{meta['channels'].nunique()}종")
    k[3].metric("길이 범위", f"{meta['duration_sec'].min():.0f}~{meta['duration_sec'].max():.0f}s")
    k[4].metric("음량 편차", f"{meta['rms_db'].max()-meta['rms_db'].min():.0f}dB")
    k[5].metric("길이 적합 비율",
                f"{meta['duration_sec'].between(AUDIO_LEN_MIN,AUDIO_LEN_MAX).mean()*100:.0f}%")

    st.markdown("""<div class="term-box">
    💡 <b>용어 설명</b><br>
    • <b>샘플레이트(Hz)</b>: 1초에 음원을 몇 번 측정했는지. 3종 혼재 시 AI 학습 전 통일 필수.<br>
    • <b>채널</b>: mono(1ch)는 소리 1개, stereo(2ch)는 좌우 2개. AI 학습은 보통 mono 사용.<br>
    • <b>RMS 음량(dBFS)</b>: 음원의 평균 크기. 값이 클수록 큰 소리. 편차가 크면 학습 편향 발생.<br>
    • <b>무음 비율</b>: 전체 길이 중 소리가 거의 없는 구간 비율. 국악은 발현음 후 여음이 길어 높습니다.
    </div>""", unsafe_allow_html=True)

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📊 코퍼스 분포", "🎸 악기군별 특성", "✂️ 세그먼트 품질"])

    with tab1:
        st.caption("50개 음원의 속성 분포. 분포가 불균질할수록 표준화 가공이 필요합니다.")
        r1, r2 = st.columns(2)
        with r1:
            fig = px.histogram(meta, x="duration_sec", nbins=20,
                               color_discrete_sequence=["#3b5bdb"], title="길이 분포")
            fig.add_vrect(x0=AUDIO_LEN_MIN, x1=AUDIO_LEN_MAX, fillcolor="green",
                          opacity=0.12, annotation_text="권장 15~30s")
            fig.update_layout(height=300, xaxis_title="길이(초)", yaxis_title="건수")
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            sr = meta["samplerate"].astype(str).value_counts().reset_index()
            sr.columns = ["샘플레이트","건수"]
            fig = px.bar(sr, x="샘플레이트", y="건수", text="건수", color="샘플레이트",
                         title="샘플레이트 분포 — 3종 혼재")
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        r3, r4 = st.columns(2)
        with r3:
            fig = px.histogram(meta, x="rms_db", nbins=20,
                               color_discrete_sequence=["#e03131"], title="음량(RMS) 분포 — 40dB 편차")
            fig.update_layout(height=300, xaxis_title="RMS(dBFS)", yaxis_title="건수")
            st.plotly_chart(fig, use_container_width=True)
        with r4:
            fig = px.histogram(meta, x="low_energy_ratio", nbins=20,
                               color_discrete_sequence=["#f08c00"], title="무음 비율 분포")
            fig.update_layout(height=300, xaxis_title="저에너지 비율(%)", yaxis_title="건수")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.caption("악기군마다 무음 비율·음량 패턴이 다릅니다. 이것이 국악 특화 가공이 필요한 이유입니다.")
        g = meta.groupby("instrument_group").agg(
            건수=("file","count"), 평균길이_s=("duration_sec","mean"),
            평균음량_dBFS=("rms_db","mean"), 평균무음비율_pct=("low_energy_ratio","mean"),
        ).round(1).reset_index()
        st.dataframe(g, hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(g, x="instrument_group", y="평균무음비율_pct",
                         text="평균무음비율_pct", color="instrument_group",
                         title="악기군별 평균 무음 비율(%)")
            fig.update_layout(height=320, showlegend=False, xaxis_title="악기군", yaxis_title="%")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(g, x="instrument_group", y="평균음량_dBFS",
                         text="평균음량_dBFS", color="instrument_group",
                         title="악기군별 평균 음량(dBFS)")
            fig.update_layout(height=320, showlegend=False, xaxis_title="악기군", yaxis_title="dBFS")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="term-box">
        💡 타악기(장구)는 타격 후 긴 여음으로 무음비율이 높고, 현악기(거문고·가야금)는 발현 후 감쇠로 높습니다.
        관악기(대금)는 지속음이라 상대적으로 낮습니다.
        → <b>악기 특성을 반영한 무음 인지(silence-aware) 세그먼트</b>가 필요합니다.
        </div>""", unsafe_allow_html=True)

    with tab3:
        if mf is None:
            st.warning("`python process_audio_poc.py` 를 먼저 실행하세요.")
        else:
            st.caption("가공 후 세그먼트가 AI 학습에 얼마나 적합한지 악기군별로 보여줍니다.")
            fit2  = mf2[mf2["dur_sec"].between(15,30)].groupby("instrument_group").size().rename("학습적합")
            short2 = mf2[(mf2["dur_sec"]>=3)&(mf2["dur_sec"]<15)].groupby("instrument_group").size().rename("보완필요")
            total2 = mf2.groupby("instrument_group").size().rename("세그먼트수")
            sg = pd.concat([total2,fit2,short2],axis=1).fillna(0).astype(int).reset_index()
            sg["적합률%"] = (sg["학습적합"]/sg["세그먼트수"]*100).round(1)
            st.dataframe(sg, hide_index=True, use_container_width=True)
            fig = px.bar(sg, x="instrument_group", y=["학습적합","보완필요"],
                         barmode="stack", text_auto=True,
                         color_discrete_map={"학습적합":"#40c057","보완필요":"#fd7e14"},
                         title="악기군별 세그먼트 적합성")
            fig.update_layout(height=340, xaxis_title="악기군", yaxis_title="세그먼트 수", legend_title="등급")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("파일별 상세 진단")
    st.caption("음원을 선택하면 측정값과 에너지 포락선(시간에 따른 소리 크기 변화)을 확인합니다.")
    sel = st.selectbox("음원 선택", meta["file"].tolist())
    row = meta[meta["file"]==sel].iloc[0]
    m2 = st.columns(5)
    m2[0].metric("길이", f"{row['duration_sec']}s")
    m2[1].metric("샘플레이트", f"{row['samplerate']}Hz")
    m2[2].metric("채널", f"{row['channels']}ch")
    m2[3].metric("RMS", f"{row['rms_db']}dBFS")
    m2[4].metric("저에너지", f"{row['low_energy_ratio']}%")
    st.info(f"AI 학습 적합성 판정: **{row['ai_grade']}**")

    src_path = AUDIO_KOGL1 / sel
    seg_dir  = SEGMENTS_DIR / Path(sel).stem
    seg_files = sorted(seg_dir.glob("*.wav")) if seg_dir.exists() else []
    wav_path = None; wav_label = ""
    if src_path.exists():   wav_path, wav_label = src_path, "원본"
    elif seg_files:         wav_path, wav_label = seg_files[0], "세그먼트 1번 조각"

    if wav_path:
        props, env_t, env_db = measure(wav_path)
        checks, grade = assess(props)
        st.dataframe(pd.DataFrame(checks, columns=["항목","판정","사유"]),
                     hide_index=True, use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=env_t, y=env_db, mode="lines",
                                 line=dict(color="#3b5bdb", width=1)))
        fig.add_hline(y=SILENCE_DB, line_dash="dash", line_color="red",
                      annotation_text=f"무음 임계({SILENCE_DB}dBFS)")
        fig.update_layout(height=320, xaxis_title="시간(초)", yaxis_title="에너지(dBFS)",
                          showlegend=False,
                          title=f"에너지 포락선 ({wav_label}) — 빨간 선 아래가 무음 구간")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="term-box">
        💡 <b>에너지 포락선이란?</b> 시간 흐름에 따라 소리의 크기가 어떻게 변하는지 보여주는 그래프입니다.
        빨간 점선 아래 구간은 무음으로, 이 구간을 인식해서 잘라야 쓸모 있는 AI 학습 조각을 만들 수 있습니다.
        </div>""", unsafe_allow_html=True)
    else:
        st.caption("에너지 포락선은 로컬 실행 시 표시됩니다.")

# ════════════════════════════════════════════════════════════════
# 4. AI 가공 결과
# ════════════════════════════════════════════════════════════════
elif menu == "✂️ AI 가공 결과":
    st.markdown("✂️ AI 가공 결과")
    st.title("음원을 직접 가공해 AI 학습 데이터를 만들었습니다")
    st.markdown("""<div class="req-box">
    📋 <b>대응 요구사항: ADR-004 (AI데이터 가공·음원) · ADR-008 (AI데이터 품질검수)</b><br>
    진단에서 발견한 불균질성(샘플레이트 3종·채널 혼재·음량 편차 40dB·길이 4~195초)을
    실제로 가공하여 표준화했습니다. 리샘플·다운믹스·음량정규화·무음인지세그먼트 4단계 파이프라인을
    직접 구현하고 검증했습니다.
    </div>""", unsafe_allow_html=True)

    if mf is None:
        st.warning("`python process_audio_poc.py` 를 먼저 실행하세요.")
        st.stop()

    total_src = mf["source"].nunique(); total_seg = len(mf)
    fit3  = mf[mf["dur_sec"].between(15,30)]
    short3 = mf[(mf["dur_sec"]>=3)&(mf["dur_sec"]<15)]

    k = st.columns(6)
    k[0].metric("원본 음원", f"{total_src}개")
    k[1].metric("세그먼트 산출", f"{total_seg}개")
    k[2].metric("학습 적합(15~30s)", f"{len(fit3)}개")
    k[3].metric("보완필요(3~14s)", f"{len(short3)}개")
    k[4].metric("샘플레이트 통일", "✅ 22050Hz" if (mf["out_sr"]==PROCESS_SR).all() else "❌")
    k[5].metric("채널 통일", "✅ mono" if (mf["out_ch"]==1).all() else "❌")

    st.markdown("""<div class="term-box">
    💡 <b>가공 4단계</b><br>
    ① <b>다운믹스</b>: stereo → mono<br>
    ② <b>리샘플</b>: 44100/48000/96000Hz → 22050Hz 통일<br>
    ③ <b>음량 정규화</b>: 들쭉날쭉한 음량 → -20dBFS 기준 통일 (학습 편향 방지)<br>
    ④ <b>무음 인지 세그먼트</b>: 긴 음원을 소리가 작아지는 지점에서 분할 → 15~30초 조각
    </div>""", unsafe_allow_html=True)

    st.divider()
    comp = pd.DataFrame({
        "항목":["샘플레이트","채널","길이","음량(RMS)"],
        "가공 전(원본)":[
            f"{sorted(mf['src_sr'].unique().tolist())} — {mf['src_sr'].nunique()}종 혼재",
            "mono+stereo 혼재",
            f"{mf['dur_sec'].min():.1f}~{mf['dur_sec'].max():.1f}s",
            f"{mf['src_rms_db'].min():.0f}~{mf['src_rms_db'].max():.0f}dBFS ({mf['src_rms_db'].max()-mf['src_rms_db'].min():.0f}dB 편차)",
        ],
        "가공 후(세그먼트)":["22050Hz 통일","mono 통일","3~30s","약 -20dBFS 정규화"],
        "가공 방법":["리샘플(soxr)","다운믹스","무음인지 세그먼트","RMS 정규화"],
    })
    st.subheader("가공 전후 속성 비교")
    st.dataframe(comp, hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(mf, x="dur_sec", nbins=20,
                           color_discrete_sequence=["#3b5bdb"], title="세그먼트 길이 분포")
        fig.add_vrect(x0=15, x1=30, fillcolor="green", opacity=0.12, annotation_text="학습 적합 15~30s")
        fig.update_layout(height=300, xaxis_title="길이(초)", yaxis_title="조각 수")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        exc3 = total_seg-len(fit3)-len(short3)
        gdf = pd.DataFrame({
            "등급":["학습 적합(15~30s)","보완필요(3~14s)","제외(<3s)"],
            "건수":[len(fit3),len(short3),exc3]})
        fig2 = px.bar(gdf, x="등급", y="건수", text="건수", color="등급",
                      color_discrete_map={"학습 적합(15~30s)":"#40c057",
                                          "보완필요(3~14s)":"#fd7e14","제외(<3s)":"#fa5252"},
                      title="AI 학습 적합성 분류")
        fig2.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("파일별 Before / After")
    st.caption("가공 전 원본 속성과 가공 후 세그먼트를 비교합니다. After 파형은 실제 가공된 음원입니다.")
    sources = sorted(mf["source"].unique().tolist())
    sel2 = st.selectbox("원본 음원 선택", sources)
    src_path2 = AUDIO_KOGL1 / sel2
    seg_dir2  = SEGMENTS_DIR / Path(sel2).stem
    seg_files2 = sorted(seg_dir2.glob("*.wav")) if seg_dir2.exists() else []
    sr2 = mf[mf["source"]==sel2].iloc[0]

    ba = pd.DataFrame({
        "항목":["샘플레이트","채널","음량(RMS)"],
        "가공 전":[f"{sr2['src_sr']}Hz",
                  f"{'stereo' if sr2['src_ch']==2 else 'mono'}({sr2['src_ch']}ch)",
                  f"{sr2['src_rms_db']}dBFS"],
        "가공 후":[f"{sr2['out_sr']}Hz","mono(1ch)","약 -20dBFS"],
    })
    st.dataframe(ba, hide_index=True, use_container_width=True)
    seg_info2 = mf[mf["source"]==sel2][["segment","start_sec","dur_sec","out_rms_db"]]
    st.dataframe(seg_info2.rename(columns={"segment":"세그먼트","start_sec":"시작(s)",
                                            "dur_sec":"길이(s)","out_rms_db":"RMS(dBFS)"}),
                 hide_index=True, use_container_width=True)

    if seg_files2:
        seg2, sr_seg2 = sf.read(str(seg_files2[0]))
        rms_a2 = round(float(20*np.log10(np.sqrt(np.mean(seg2**2))+1e-12)),1)
        t_seg2 = np.arange(len(seg2))/sr_seg2
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Before** — 원본  \n`{sr2['src_sr']}Hz / "
                        f"{'stereo' if sr2['src_ch']==2 else 'mono'} / {sr2['src_rms_db']}dBFS`")
            if src_path2.exists():
                raw2, sr_raw2 = sf.read(str(src_path2))
                rm2 = raw2.mean(axis=1) if raw2.ndim>1 else raw2
                pl2 = min(len(rm2), sr_raw2*30)
                fig3 = go.Figure(go.Scatter(x=np.arange(pl2)/sr_raw2, y=rm2[:pl2],
                                            mode="lines", line=dict(width=0.8, color="#1565C0")))
                fig3.update_layout(height=240, xaxis_title="시간(초)", yaxis_title="진폭",
                                   margin=dict(t=10,b=40), title="원본 파형(앞 30초)")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.caption("원본 파형은 로컬 실행 시 표시됩니다.")
        with c2:
            st.markdown(f"**After** — {seg_files2[0].name} ({round(len(seg2)/sr_seg2,1)}s)  \n"
                        f"`{sr_seg2}Hz / mono / {rms_a2}dBFS`")
            fig4 = go.Figure(go.Scatter(x=t_seg2, y=seg2,
                                        mode="lines", line=dict(width=0.8, color="#2E7D32")))
            fig4.update_layout(height=240, xaxis_title="시간(초)", yaxis_title="진폭",
                               margin=dict(t=10,b=40), title="가공 후 세그먼트 파형")
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.caption("세그먼트 파일이 없습니다.")

    st.divider()
    st.markdown(f"""<div class="term-box">
    💡 <b>이 화면이 제안서에서 말하는 것</b><br>
    원본 {total_src}개(SR 3종·채널 혼재·길이 4~195s·음량 편차 ~40dB)를 실제로 가공하여
    세그먼트 {total_seg}개(22050Hz·mono·목표 -20dBFS·학습적합 {len(fit3)}개)를 산출했습니다.
    모든 가공은 공공누리 1유형(변경허용) 데이터에만 적용하였습니다.
    본사업에서는 동일 파이프라인을 전체 음원 10,000건으로 확장 적용합니다.
    </div>""", unsafe_allow_html=True)