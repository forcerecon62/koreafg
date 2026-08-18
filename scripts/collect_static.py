"""
K-FGI 일일 수집 + 점수화 스크립트 (완전 독립형 - 다른 파일에 의존하지 않음)

GitHub Actions에서 매일 실행되어:
  1. 8개 지표의 raw값을 KRX(pykrx) / Yahoo Finance(yfinance)에서 수집하고
  2. 과거 기록(docs/data/history.json) 대비 percentile로 0~100점을 매긴 뒤
  3. 오늘자 기록을 docs/data/history.json에 추가 저장한다.

docs/index.html이 이 JSON 파일을 읽어 대시보드를 그린다.
"""
import sys
import json
import time
from bisect import bisect_left
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yfinance as yf
from pykrx import stock

# ── 설정 ──────────────────────────────────────────────────
TIMEZONE = "Asia/Seoul"
LOOKBACK_DAYS = 252  # percentile 계산에 사용할 과거 거래일 수 (약 1년)
MAX_RECORDS = 800    # history.json에 보관할 최대 일수 (약 3년)

YF_KOSPI = "^KS11"
YF_KOSDAQ = "^KQ11"
YF_USDKRW = "KRW=X"
YF_GOLD = "GC=F"

TOP_N_BY_CAP = 100        # 주가 강도 지표에 사용할 시총 상위 표본 수
NEAR_BAND_PCT = 5.0        # 52주 고/저가 대비 근접 판정 기준(%)

INDICATOR_WEIGHTS = {
    "volatility": 0.18,
    "market_breadth": 0.17,
    "kospi_momentum": 0.15,
    "fx_stress": 0.13,
    "kosdaq_momentum": 0.12,
    "foreign_flow": 0.10,
    "price_strength": 0.08,
    "safe_haven": 0.07,
}  # 합 1.00 — 2026-08-18 yasun.gg 대비 캘리브레이션 반영

INDICATOR_LABELS = {
    "kospi_momentum": "KOSPI 모멘텀",
    "kosdaq_momentum": "KOSDAQ 모멘텀",
    "volatility": "변동성 (KOSPI)",
    "fx_stress": "환율 스트레스",
    "market_breadth": "시장 폭 (KS-KQ)",
    "foreign_flow": "외국인 5일 수급",
    "price_strength": "주가 강도 (52주)",
    "safe_haven": "안전자산 수요",
}

# True: raw값이 클수록 탐욕(점수↑) / False: raw값이 클수록 공포(점수↓, 반전)
GREEDY_WHEN_HIGH = {
    "kospi_momentum": True,
    "kosdaq_momentum": True,
    "volatility": False,
    "fx_stress": False,
    "market_breadth": True,
    "foreign_flow": True,
    "price_strength": True,
    "safe_haven": True,
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "docs" / "data" / "history.json"


# ── 날짜 유틸 ─────────────────────────────────────────────
def now_kst() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


def today_str(fmt: str = "%Y%m%d") -> str:
    return now_kst().strftime(fmt)


def days_ago_str(n: int, fmt: str = "%Y%m%d") -> str:
    return (now_kst() - timedelta(days=n)).strftime(fmt)


def _retry(fn, tries=3, delay=4):
    """KRX 서버가 간헐적으로 빈 응답을 줄 때(Expecting value 에러) 잠깐 쉬었다가 재시도."""
    last_err = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i < tries - 1:
                print(f"[RETRY] {i+1}/{tries} 실패({e}), {delay}초 후 재시도")
                time.sleep(delay)
    raise last_err


# ── 지표 수집 ─────────────────────────────────────────────
def collect_market_index() -> dict:
    """KOSPI/KOSDAQ 모멘텀(60일 이평 괴리율), KOSPI 변동성(20일 실현변동성 연율화)"""
    out = {}
    try:
        for key, ticker in (("kospi_momentum", YF_KOSPI), ("kosdaq_momentum", YF_KOSDAQ)):
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            close = df["Close"].dropna()
            ma = close.rolling(60).mean()  # 60거래일(약 1분기) 이평 — 급락 후 하루 반등에 과민반응 방지
            out[key] = round(float((close.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1] * 100), 2)

        df = yf.download(YF_KOSPI, period="1y", interval="1d", progress=False, auto_adjust=True)
        close = df["Close"].dropna()
        log_ret = np.log(close / close.shift(1)).dropna().tail(20)
        out["volatility"] = round(float(log_ret.std() * np.sqrt(252) * 100), 2)
    except Exception as e:
        print(f"[WARN] market_index 수집 실패: {e}")
    return out


def collect_fx() -> dict:
    """원/달러 5일 변동률(%). 클수록(원화 약세 심화) 공포."""
    out = {}
    try:
        df = yf.download(YF_USDKRW, period="6mo", interval="1d", progress=False, auto_adjust=True)
        close = df["Close"].dropna()
        last, prior = float(close.iloc[-1]), float(close.iloc[-6])
        out["fx_stress"] = round((last - prior) / prior * 100, 2)
    except Exception as e:
        print(f"[WARN] fx_stress 수집 실패: {e}")
    return out


def collect_safe_haven() -> dict:
    """KOSPI 60일 수익률 - 금 60일 수익률 스프레드(모멘텀과 윈도우 통일). 낮을수록(주식이 금 대비 부진) 공포."""
    out = {}
    try:
        def n_day_return(ticker, window=60):
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            close = df["Close"].dropna()
            return (float(close.iloc[-1]) - float(close.iloc[-1 - window])) / float(close.iloc[-1 - window]) * 100

        out["safe_haven"] = round(n_day_return(YF_KOSPI) - n_day_return(YF_GOLD), 2)
    except Exception as e:
        print(f"[WARN] safe_haven 수집 실패: {e}")
    return out


def collect_breadth() -> dict:
    """코스피+코스닥 전종목 중 상승-하락 종목 비율(%). 낮을수록 공포."""
    out = {}
    try:
        df = _retry(lambda: stock.get_market_ohlcv_by_ticker(today_str(), market="ALL", alternative=True))
        chg = df["등락률"]
        adv, dec = int((chg > 0).sum()), int((chg < 0).sum())
        total = adv + dec
        if total:
            out["market_breadth"] = round((adv - dec) / total * 100, 2)
    except Exception as e:
        print(f"[WARN] market_breadth 수집 실패: {e}")
    return out


def collect_foreign_flow() -> dict:
    """최근 5거래일 외국인 순매수 대금 합계(억원). 클수록 탐욕."""
    out = {}
    try:
        df = _retry(lambda: stock.get_market_net_purchases_of_equities(
            days_ago_str(7), today_str(), market="ALL", investor="외국인"
        ))
        out["foreign_flow"] = round(float(df["순매수거래대금"].sum()) / 100_000_000, 1)
    except Exception as e:
        print(f"[WARN] foreign_flow 수집 실패: {e}")
    return out


def collect_strength() -> dict:
    """시총 상위 N종목 중 52주 신고가 근접 - 신저가 근접 비율(%). 낮을수록 공포."""
    out = {}
    try:
        date_s = today_str()
        cap_df = _retry(lambda: stock.get_market_cap_by_ticker(date_s, market="ALL", alternative=True))
        tickers = list(cap_df.sort_values("시가총액", ascending=False).index[:TOP_N_BY_CAP])

        near_high, near_low, counted = 0, 0, 0
        fromdate = days_ago_str(365)
        for ticker in tickers:
            try:
                ohlcv = _retry(lambda t=ticker: stock.get_market_ohlcv_by_date(fromdate, date_s, t), tries=2, delay=2)
                if ohlcv.empty:
                    continue
                hi, lo = float(ohlcv["고가"].max()), float(ohlcv["저가"].min())
                last = float(ohlcv["종가"].iloc[-1])
                if hi > 0 and (hi - last) / hi * 100 <= NEAR_BAND_PCT:
                    near_high += 1
                if lo > 0 and (last - lo) / lo * 100 <= NEAR_BAND_PCT:
                    near_low += 1
                counted += 1
            except Exception:
                continue
        if counted:
            out["price_strength"] = round((near_high - near_low) / counted * 100, 2)
    except Exception as e:
        print(f"[WARN] price_strength 수집 실패: {e}")
    return out


COLLECTORS = [
    collect_market_index, collect_fx, collect_safe_haven,
    collect_breadth, collect_foreign_flow, collect_strength,
]


# ── 점수화 ────────────────────────────────────────────────
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


# ── 저장소 I/O ────────────────────────────────────────────
def load_history() -> list:
    if DATA_PATH.exists():
        try:
            return json.loads(DATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("[WARN] 기존 history.json 파싱 실패 - 빈 배열로 시작합니다")
    return []


def save_history(records: list):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 메인 ──────────────────────────────────────────────────
def main():
    today = date.today().isoformat()
    history = [r for r in load_history() if r["date"] != today]

    raw = {}
    for fn in COLLECTORS:
        try:
            raw.update(fn())
        except Exception as e:
            print(f"[WARN] {fn.__name__} 전체 실패: {e}")

    if not raw:
        print("[ERROR] 모든 지표 수집 실패 - 이번 실행은 기록하지 않습니다")
        sys.exit(1)

    print("수집된 raw 지표:", raw)

    per_indicator_hist = {k: [] for k in raw}
    for rec in history[-LOOKBACK_DAYS:]:
        for c in rec.get("components", []):
            if c["key"] in per_indicator_hist:
                per_indicator_hist[c["key"]].append(c["raw"])

    components, scores = [], {}
    for key, value in raw.items():
        pct = percentile_score(value, per_indicator_hist[key] + [value])
        score = pct if GREEDY_WHEN_HIGH.get(key, True) else round(100 - pct, 1)
        scores[key] = score
        components.append({
            "key": key, "label": INDICATOR_LABELS.get(key, key),
            "raw": value, "score": score, "verdict": label_for(score),
        })

    total_w, acc = 0.0, 0.0
    for k, w in INDICATOR_WEIGHTS.items():
        if k in scores:
            acc += scores[k] * w
            total_w += w
    composite = round(acc / total_w, 1) if total_w else 50.0

    history.append({
        "date": today, "composite_score": composite,
        "verdict": label_for(composite), "components": components,
    })
    history.sort(key=lambda r: r["date"])
    save_history(history[-MAX_RECORDS:])

    print(f"✅ {today} 저장 완료 - 합성지수 {composite}점 ({label_for(composite)})")


if __name__ == "__main__":
    main()
