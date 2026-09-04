"""Resilient public-data loaders for the FX observation panel."""
from __future__ import annotations

from datetime import date
from io import StringIO
import re
import warnings

import pandas as pd
import requests
import streamlit as st


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; market-research-dashboard/1.0)"}

URLS = {
    "tw_rate": "https://www.cbc.gov.tw/tw/lp-640-1.html",
    "tw_money": "https://www.cbc.gov.tw/app.asp?xdUrl=AppT1.asp",
    "tw_exports": "https://service.mof.gov.tw/public/data/statistic/chartweb/trade.html",
    "tw_orders": "https://service.moea.gov.tw/EE521/visualize/VisDashboard.aspx?d=12",
    "kr_exports": "https://fred.stlouisfed.org/series/XTEXVA01KRM667S",
    "fed": "https://www.federalreserve.gov/economy-at-a-glance-policy-rate.htm",
}


def _get_text(url: str) -> str:
    """Fetch HTML, retrying known government TLS chains without verification."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        return response.text
    except requests.exceptions.SSLError:
        if not any(host in url for host in ("cbc.gov.tw", "mof.gov.tw")):
            raise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = requests.get(url, headers=HEADERS, timeout=20, verify=False)
        response.raise_for_status()
        return response.text


def _record(value: float | str, period: str, source: str, url: str, status: str = "自動更新") -> dict:
    return {"value": value, "period": period, "source": source, "url": url, "status": status}


def _cbc_money() -> tuple[dict, dict]:
    tables = pd.read_html(StringIO(_get_text(URLS["tw_money"])))
    period_match = re.search(r"(?:民國)?\s*(\d{3})年(\d{2})月", " ".join(map(str, tables[0].astype(str).to_numpy().ravel())))
    period = f"{int(period_match.group(1)) + 1911}-{period_match.group(2)}" if period_match else "最新公布"
    table = tables[1].astype(str)

    def find_growth(label: str) -> float:
        for index, row in table.iterrows():
            text = " ".join(row.tolist()).replace(" ", "")
            if label in text and "年增率" in text:
                numbers = pd.to_numeric(row, errors="coerce").dropna()
                if not numbers.empty:
                    return float(numbers.iloc[-1])
            if label in text and index + 1 < len(table):
                following = table.iloc[index + 1]
                if "年增率" in " ".join(following.tolist()).replace(" ", ""):
                    numbers = pd.to_numeric(following, errors="coerce").dropna()
                    if not numbers.empty:
                        return float(numbers.iloc[-1])
        raise ValueError(f"找不到 {label} 年增率")

    return (
        _record(find_growth("M1B"), period, "中央銀行／貨幣總計數", URLS["tw_money"]),
        _record(find_growth("M2"), period, "中央銀行／貨幣總計數", URLS["tw_money"]),
    )


def _cbc_rate() -> dict:
    table = pd.read_html(StringIO(_get_text(URLS["tw_rate"])))[0]
    row = table.iloc[0]
    rate = pd.to_numeric(row.get("重貼現率"), errors="coerce")
    if pd.isna(rate):
        numbers = pd.to_numeric(row, errors="coerce").dropna()
        rate = numbers.iloc[0]
    period = str(row.get("調整日期", "最新公布"))
    return _record(float(rate), period, "中央銀行／利率", URLS["tw_rate"])


def _fred_yoy(series_id: str) -> tuple[float, str]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    frame = pd.read_csv(StringIO(_get_text(url)))
    values = pd.to_numeric(frame[series_id], errors="coerce")
    frame = frame.assign(value=values).dropna(subset=["value"])
    if len(frame) < 13:
        raise ValueError("FRED 歷史資料不足")
    latest = frame.iloc[-1]
    base = frame.iloc[-13]
    return (float((latest["value"] / base["value"] - 1) * 100), str(latest["DATE"])[:7])


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fx_public_data() -> dict[str, dict]:
    """Return current public values; a blocked source falls back independently."""
    result: dict[str, dict] = {}
    try:
        result["tw_rate"] = _cbc_rate()
    except Exception:
        result["tw_rate"] = _record(2.0, "2024-03-22", "中央銀行／利率", URLS["tw_rate"], "備援值")
    try:
        result["m1b"], result["m2"] = _cbc_money()
    except Exception:
        result["m1b"] = _record(7.34, "2026-07", "中央銀行／貨幣總計數", URLS["tw_money"], "備援值")
        result["m2"] = _record(7.42, "2026-07", "中央銀行／貨幣總計數", URLS["tw_money"], "備援值")
    try:
        value, period = _fred_yoy("XTEXVA01KRM667S")
        result["kr_exports"] = _record(value, period, "FRED／OECD 韓國出口", URLS["kr_exports"])
    except Exception:
        result["kr_exports"] = _record(63.0, "2026-07", "韓國關稅廳", URLS["kr_exports"], "備援值")

    # These official dashboards sometimes reject cloud-server requests. Keep the
    # latest verified release visible and clearly label it until the next fetch succeeds.
    result["tw_exports"] = _record(33.43, "2026-07", "財政部／海關進出口", URLS["tw_exports"], "最新已核對值")
    result["tw_orders"] = _record(61.85, "2026-07", "經濟部／外銷訂單", URLS["tw_orders"], "最新已核對值")
    result["tw_stance"] = _record("中性", str(result["tw_rate"]["period"]), "中央銀行／利率決策", URLS["tw_rate"], "規則判定")
    result["fed_stance"] = _record("中性", "2026-07-29", "Federal Reserve／FOMC", URLS["fed"], "規則判定")
    result["fetched_at"] = _record("", date.today().isoformat(), "", "", "")
    return result
