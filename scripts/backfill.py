"""
초기 이력 채우기 스크립트 (1회만 실행하면 됨)

collect_static.py는 매일 딱 하루치씩만 쌓기 때문에, 서비스를 막 시작하면 percentile을
계산할 과거 데이터가 없어 지표가 전부 0점 아니면 100점으로만 나온다(cold start 문제).

이 스크립트는 yfinance가 한 번 호출할 때 어차피 받아오는 1년치 시세를 이용해서,
과거 약 190여 거래일치의 raw 지표를 소급 계산해 docs/data/history.json에 채워 넣는다.

대상 5개 지표 (yfinance 기반 - 한 번의 다운로드로 전체 과거 계산 가능):
  - KOSPI/KOSDAQ 모멘텀, 변동성, 환율 스트레스, 안전자산 수요

제외 3개 지표 (KRX 기반 - 날짜별로 따로 조회해야 해서 비용/차단 위험이 큼):
  - 시장 폭, 외국인 수급, 주가 강도
  → collect_static.py가 매일 실행되면서 자연히 쌓인다.

지표 중 일부만 그날 결측이어도(야후 데이터 자체의 자잘한 누락 등) 그 지표만 빼고
나머지로 그날을 계산한다 - 5개가 전부 없는 날만 통으로 건너뛴다.

8개 지표가 다 채워진 "완전한" 실제 수집일은 그대로 보존하고, 그 외(백필된 날 / 일부
지표만 있는 날)는 매번 최신 가중치로 다시 계산한다.

실행: GitHub Actions의 "Backfill History" 워크플로를 Run workflow로 실행.
"""
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

YF_KOSPI = "^KS11"
YF_KOSDAQ = "^KQ11"
YF_USDKRW = "KRW=X"
YF_GOLD = "GC=F"

MOMENTUM_WINDOW = 60
VOL_WINDOW = 20
FX_WINDOW = 5
SAFE_HAVEN_WINDOW = 60

GREEDY_WHEN_HIGH = {
    "kospi_momentum": True,
    "kosdaq_momentum": True,
    "volatility": False,
    "fx_stress": False,
    "safe_haven": True,
}
INDICATOR_LABELS = {
    "kospi_momentum": "KOSPI 모멘텀",
    "kosdaq_momentum": "KOSDAQ 모멘텀",
    "volatility": "변동성 (KOSPI)",
    "fx_stress": "환율 스트레스",
    "safe_haven": "안전자산 수요",
}
# 대시보드/일일 스크립트와 동일한 가중치 중 백필 대상 5개만 사용 (합 기준으로 재정규화됨)
# 2026-08-20 재캘리브레이션: 7월 말 폭락 때 환율(SK하이닉스 나스닥 상장 이슈로 독자 강세)이
# 주가 폭락 신호를 과도하게 희석시키는 게 확인되어, 환율 비중을 낮추고 가격 지표에 더 실었다.
INDICATOR_WEIGHTS = {
    "volatility": 0.20,
    "kospi_momentum": 0.17,
    "kosdaq_momentum": 0.14,
    "fx_stress": 0.07,
    "safe_haven": 0.07,
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "docs" / "data" / "history.json"


def percentile_score(value: float, history: list) -> float:
    if not history:
        return 50.0
    s = sorted(history)
    return round(bisect_left(s, value) / len(s) * 100, 1)


def label_for(score: float) -> str:
    if score < 25:
        return "극단적 공포"
    if score < 45:
        return "공포"
    if score <= 55:
        return "중립"
    if score <= 75:
        return "탐욕"
    return "극단적 탐욕"


def _close(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        # 최신 yfinance는 단일 종목이어도 MultiIndex 컬럼을 줘서 df["Close"]가
        # Series가 아니라 1개짜리 DataFrame으로 나온다 - 진짜 Series로 변환
        close = close.iloc[:, 0]
    return close.dropna()


def build_raw_series() -> pd.DataFrame:
    """KOSPI 거래일을 기준 달력으로 삼아 5개 지표의 raw 시계열을 만든다."""
    kospi = _close(YF_KOSPI)
    kosdaq = _close(YF_KOSDAQ).reindex(kospi.index, method="ffill")
    usdkrw = _close(YF_USDKRW).reindex(kospi.index, method="ffill")
    gold = _close(YF_GOLD).reindex(kospi.index, method="ffill")

    kospi_ma = kospi.rolling(MOMENTUM_WINDOW).mean()
    kosdaq_ma = kosdaq.rolling(MOMENTUM_WINDOW).mean()
    log_ret = np.log(kospi / kospi.shift(1))

    df = pd.DataFrame({
        "kospi_momentum": (kospi - kospi_ma) / kospi_ma * 100,
        "kosdaq_momentum": (kosdaq - kosdaq_ma) / kosdaq_ma * 100,
        "volatility": log_ret.rolling(VOL_WINDOW).std() * np.sqrt(252) * 100,
        "fx_stress": (usdkrw - usdkrw.shift(FX_WINDOW)) / usdkrw.shift(FX_WINDOW) * 100,
        "safe_haven": (
            (kospi - kospi.shift(SAFE_HAVEN_WINDOW)) / kospi.shift(SAFE_HAVEN_WINDOW) * 100
            - (gold - gold.shift(SAFE_HAVEN_WINDOW)) / gold.shift(SAFE_HAVEN_WINDOW) * 100
        ),
    })
    return df  # dropna 하지 않음 - 지표 중 일부만 결측이어도 나머지로 그날을 계산하기 위함


def load_history() -> list:
    if DATA_PATH.exists():
        try:
            return json.loads(DATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_history(records: list):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    existing = load_history()
    existing_by_date = {r["date"]: r for r in existing}

    raw_df = build_raw_series()
    print(f"소급 계산 가능한 거래일 수: {len(raw_df)}")

    per_indicator_hist = {k: [] for k in GREEDY_WHEN_HIGH}
    computed_count = 0

    for ts, row in raw_df.iterrows():
        date_str = ts.strftime("%Y-%m-%d")

        components, scores = [], {}
        for key in GREEDY_WHEN_HIGH:
            raw_value = row[key]
            if pd.isna(raw_value):
                continue  # 이 지표만 그날 결측 - 나머지 지표로 계속 진행
            value = round(float(raw_value), 2)
            hist = per_indicator_hist[key] + [value]
            pct = percentile_score(value, hist)
            score = pct if GREEDY_WHEN_HIGH[key] else round(100 - pct, 1)
            scores[key] = score
            components.append({
                "key": key, "label": INDICATOR_LABELS[key],
                "raw": value, "score": score, "verdict": label_for(score),
            })
            per_indicator_hist[key].append(value)  # 다음 날 계산을 위해 이력에 추가(당일 시점까지만 사용 = 미래참조 없음)

        if not components:
            continue  # 5개 지표가 그날 전부 결측이면 진짜로 건너뜀

        prior = existing_by_date.get(date_str)
        if prior and len(prior.get("components", [])) >= 8:
            continue  # 8개 지표가 다 채워진 "완전한" 실제 수집일은 건드리지 않음

        total_w, acc = 0.0, 0.0
        for k, w in INDICATOR_WEIGHTS.items():
            if k in scores:
                acc += scores[k] * w
                total_w += w
        composite = round(acc / total_w, 1) if total_w else 50.0

        existing_by_date[date_str] = {
            "date": date_str, "composite_score": composite,
            "verdict": label_for(composite), "components": components,
        }
        computed_count += 1

    merged = sorted(existing_by_date.values(), key=lambda r: r["date"])
    save_history(merged)

    print(f"✅ 백필 완료 - {computed_count}일 (재)계산, 전체 {len(merged)}일")


if __name__ == "__main__":
    main()
