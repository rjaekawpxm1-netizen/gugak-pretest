# 국악 데이터 개방·AI학습 적합성 사전진단 (gugak-pretest)

AI 국악 음원·이미지 데이터 개방체계 구축 사업 제안서용 **사전진단 프로토타입**.
실제 공개 데이터로 품질·AI 학습 적합성을 진단하여, 개방DB 설계·AI 라벨 체계·오픈API
제공항목·품질개선 과제의 근거를 만든다. (가짜 데이터 사용 안 함)

## 대상 데이터
- **음원 (사업 핵심 대상)**: 국악기 디지털음원 단음·악구·확장, 공공누리 1유형, WAV
- **아카이브 목록 (현행 개방 데이터, 보조)**: VR 영상 목록, 온라인 전시 목록
  - RFP "데이터 개방 현황"에 해당. 신규 개방 대상이 아니라 *기존 개방 데이터 품질 진단* 근거로 사용.

## 구조
```
app.py                  Streamlit 진입점
src/                    진단 로직 (화면에서 import)
  config.py             기준값·경로·공공누리 유형
  loaders.py            파일 로딩
  diagnose_metadata.py  메타데이터 품질진단
  diagnose_audio.py     음원 품질·AI 적합성 진단
pages/                  Streamlit 화면 (하나씩 추가)
data/raw/               원본 (git 제외)
reports/                산출 리포트 (제안서 캡처용)
licenses/               파일별 공공누리 유형 추적
```

## 실행
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
