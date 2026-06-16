# -*- coding: utf-8 -*-
"""음원 폴더를 스캔해 메타데이터 CSV를 자동 생성한다.

- 오디오 측정값(길이/샘플레이트/채널/음량/무음/등급)은 전부 자동.
- 악기명/악기군은 파일명 부분일치로 best-effort 추출 (틀리면 CSV에서 수정).
- 장르(정악/산조/민요)도 추출. type(단음/악구/확장)은 길이 추정값 제시 → 사용자 확정.
- 코드형 파일명(m/s/w 등)은 매핑 불가 → 빈칸으로 두고 CODE_MAP에 규칙을 추가하면 자동 채움.

실행(프로젝트 루트): python build_audio_metadata.py
출력: data/processed/audio_metadata.csv
"""

import csv
from pathlib import Path

from src.config import AUDIO_KOGL1
from src.diagnose_audio import measure, assess

OUT = Path("data/processed/audio_metadata.csv")

# 악기 로마자(부분일치) → (악기명, 악기군). 긴 키부터 매칭한다.
ROMAN = {
    "daegeum": ("대금", "관악기"), "deageum": ("대금", "관악기"), "sogeum": ("소금", "관악기"),
    "danso": ("단소", "관악기"), "piri": ("피리", "관악기"), "taepyeongso": ("태평소", "관악기"),
    "saenghwang": ("생황", "관악기"),
    "gayageum": ("가야금", "현악기"), "geomungo": ("거문고", "현악기"), "geum": ("거문고", "현악기"),
    "haegeum": ("해금", "현악기"), "ajaeng": ("아쟁", "현악기"), "daejaeng": ("대쟁", "현악기"),
    "yanggeum": ("양금", "현악기"), "bipa": ("비파", "현악기"), "wolgeum": ("월금", "현악기"),
    "janggu": ("장구", "타악기"), "buk": ("북", "타악기"), "jing": ("징", "타악기"),
    "kkwaenggwari": ("꽹과리", "타악기"), "pyeonjong": ("편종", "타악기"), "pyeongyeong": ("편경", "타악기"),
}

# 코드형 접두 → (악기명, 악기군). 코드 의미를 알게 되면 여기에 추가.
# 예) "w2": ("대금","관악기"), "s2": ("가야금","현악기")  ← 실제 코드표 확인 후 입력
CODE_MAP: dict[str, tuple[str, str]] = {}


def parse_instrument(stem: str):
    s = stem.lower()
    # 1) 코드형 접두 우선 (예: w2-721-010 → 'w2')
    head = s.split("-")[0]
    if head in CODE_MAP:
        return CODE_MAP[head]
    # 2) 악기 로마자 부분일치 (긴 키부터)
    for key in sorted(ROMAN, key=len, reverse=True):
        if key in s:
            return ROMAN[key]
    return ("", "")


def parse_genre(stem: str) -> str:
    s = stem.lower()
    if "jungak" in s:
        return "정악"
    if "sanjo" in s:
        return "산조"
    if "아리랑" in stem:
        return "민요(아리랑)"
    return ""


def guess_type(duration: float) -> str:
    if duration < 3.5:
        return "단음"
    if duration <= 25:
        return "악구"
    return "확장"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = sorted([*AUDIO_KOGL1.glob("*.wav"), *AUDIO_KOGL1.glob("*.mp3")])
    if not files:
        print("음원이 없습니다:", AUDIO_KOGL1)
        return

    rows = []
    for f in files:
        props, _, _ = measure(f)
        _, grade = assess(props)
        name, group = parse_instrument(f.stem)
        rows.append({
            "file": f.name,
            "instrument_group": group,
            "instrument_name": name,
            "genre": parse_genre(f.stem),
            "type": "",
            "type_guess": guess_type(props["duration"]),
            "duration_sec": props["duration"],
            "samplerate": props["samplerate"],
            "channels": props["channels"],
            "rms_db": props["rms_db"],
            "low_energy_ratio": props["low_energy_ratio"],
            "ai_grade": grade,
        })

    with open(OUT, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    matched = sum(1 for r in rows if r["instrument_name"])
    print(f"총 {len(rows)}개 → {OUT}")
    print(f"악기명 자동매칭 {matched}/{len(rows)}개 (미매칭은 CSV에서 직접 입력)")
    print("유형 추정 분포:", {t: sum(r["type_guess"] == t for r in rows)
                          for t in ["단음", "악구", "확장"]})


if __name__ == "__main__":
    main()