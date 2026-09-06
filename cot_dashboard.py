"""CFTC Commitments of Traders positioning dashboard."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st


COT_MARKETS = {
    "S&P 500": ("E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE", "淨多代表大型美股偏多"),
    "Nasdaq 100": ("NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE", "淨多代表科技股偏多"),
    "美元指數": ("USD INDEX - ICE FUTURES U.S.", "淨多代表美元偏強"),
    "日圓": ("JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", "淨多代表日圓偏強"),
    "日經平均（美元計價）": ("NIKKEI STOCK AVERAGE - CHICAGO MERCANTILE EXCHANGE", "美元計價日經期貨；淨多代表日股偏多"),
    "日經平均（日圓計價）": ("NIKKEI STOCK AVERAGE YEN DENOM - CHICAGO MERCANTILE EXCHANGE", "日圓計價日經期貨；可與美元計價合約比較匯率影響"),
    "MSCI EAFE": ("MSCI EAFE - ICE FUTURES U.S.", "歐洲、澳洲及遠東已開發市場綜合部位，不等同純亞洲"),
    "MSCI Emerging Markets": ("MSCI EM INDEX - ICE FUTURES U.S.", "新興市場綜合部位，可作亞洲新興市場風險偏好代理"),
    "黃金": ("GOLD - COMMODITY EXCHANGE INC.", "淨多代表黃金偏多"),
    "WTI原油": ("CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE", "淨多代表油價偏多；使用ICE WTI合約"),
    "美國10年債": ("UST 10Y NOTE - CHICAGO BOARD OF TRADE", "淨多代表債券價格偏多、殖利率偏降"),
}


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_cot_history() -> pd.DataFrame:
    start = (pd.Timestamp.now().normalize() - pd.DateOffset(years=3)).strftime("%Y-%m-%dT00:00:00.000")
    market_names = [market for market, _ in COT_MARKETS.values()]
    quoted = ",".join("'" + name.replace("'", "''") + "'" for name in market_names)
    params = {
        "$select": ",".join([
            "report_date_as_yyyy_mm_dd", "market_and_exchange_names", "open_interest_all",
            "noncomm_positions_long_all", "noncomm_positions_short_all",
            "comm_positions_long_all", "comm_positions_short_all",
        ]),
        "$where": f"report_date_as_yyyy_mm_dd >= '{start}' AND futonly_or_combined='FutOnly' AND market_and_exchange_names IN ({quoted})",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": 5000,
    }
    response = requests.get(
        "https://publicreporting.cftc.gov/resource/srt6-5q2f.json",
        params=params,
        headers={"User-Agent": "global-market-dashboard/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    raw = pd.DataFrame(response.json())
    if raw.empty:
        raise ValueError("CFTC returned no COT observations")
    rename = {
        "report_date_as_yyyy_mm_dd": "日期", "market_and_exchange_names": "CFTC市場",
        "open_interest_all": "未平倉量", "noncomm_positions_long_all": "非商業多單",
        "noncomm_positions_short_all": "非商業空單", "comm_positions_long_all": "商業多單",
        "comm_positions_short_all": "商業空單",
    }
    raw = raw.rename(columns=rename)
    raw["日期"] = pd.to_datetime(raw["日期"], errors="coerce")
    for column in ["未平倉量", "非商業多單", "非商業空單", "商業多單", "商業空單"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    reverse = {market: label for label, (market, _) in COT_MARKETS.items()}
    raw["資產"] = raw["CFTC市場"].map(reverse)
    raw = raw.dropna(subset=["日期", "資產", "未平倉量"]).sort_values(["資產", "日期"])
    raw["非商業淨部位"] = raw["非商業多單"] - raw["非商業空單"]
    raw["商業淨部位"] = raw["商業多單"] - raw["商業空單"]
    raw["非商業淨部位/OI%"] = raw["非商業淨部位"] / raw["未平倉量"].replace(0, np.nan) * 100
    raw["淨部位週變化"] = raw.groupby("資產")["非商業淨部位"].diff()
    raw["三年百分位%"] = raw.groupby("資產")["非商業淨部位"].rank(pct=True) * 100
    return raw


def _signal(percentile: float, weekly_change: float) -> str:
    if percentile >= 80:
        base = "🟢 強勢淨多／偏擁擠"
    elif percentile >= 60:
        base = "🟢 偏多"
    elif percentile <= 20:
        base = "🔴 強勢淨空／偏擁擠"
    elif percentile <= 40:
        base = "🔴 偏空"
    else:
        base = "⚪ 中性"
    direction = "加碼" if weekly_change > 0 else "減碼" if weekly_change < 0 else "持平"
    return f"{base}｜{direction}"


def render_cot_dashboard() -> None:
    st.header("CFTC COT 市場部位")
    st.caption("COT是每週部位報告，報告日通常為週二，發布時間晚於持倉截點；它適合觀察資金方向與擁擠度，不是即時買賣訊號。")
    try:
        with st.spinner("正在載入 CFTC 官方 COT 資料…"):
            history = fetch_cot_history()
    except Exception:
        st.warning("CFTC公開資料目前暫時無法取得，請稍後重新整理。")
        st.link_button("開啟 CFTC COT 官方頁面", "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm")
        return

    latest = history.groupby("資產", as_index=False).tail(1).copy()
    latest["市場狀況"] = latest.apply(lambda row: _signal(row["三年百分位%"], row["淨部位週變化"]), axis=1)
    latest["解讀"] = latest["資產"].map({label: note for label, (_, note) in COT_MARKETS.items()})
    newest = latest["日期"].max()
    latest["資料新鮮度"] = np.where((newest-latest["日期"]).dt.days <= 14, "✅ 最新週期", "⚠️ 舊資料／低頻申報")
    latest = latest.sort_values("三年百分位%", ascending=False)
    bullish = int((latest["三年百分位%"] >= 60).sum())
    bearish = int((latest["三年百分位%"] <= 40).sum())
    crowded = int(((latest["三年百分位%"] >= 80) | (latest["三年百分位%"] <= 20)).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新報告日", newest.strftime("%Y-%m-%d"), border=True)
    c2.metric("偏多資產", f"{bullish}項", border=True)
    c3.metric("偏空資產", f"{bearish}項", border=True)
    c4.metric("極端／擁擠", f"{crowded}項", border=True)

    st.subheader("主要市場最新部位")
    st.dataframe(
        latest[["資產", "日期", "資料新鮮度", "非商業淨部位", "淨部位週變化", "非商業淨部位/OI%", "三年百分位%", "市場狀況", "解讀"]],
        hide_index=True, width="stretch",
        column_config={
            "日期": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "非商業淨部位": st.column_config.NumberColumn(format="%d"),
            "淨部位週變化": st.column_config.NumberColumn(format="%+d"),
            "非商業淨部位/OI%": st.column_config.NumberColumn(format="%.2f%%"),
            "三年百分位%": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        },
    )

    selected = st.selectbox("選擇資產查看歷史部位", list(COT_MARKETS), key="cot_selected_market")
    selected_history = history.loc[history["資產"] == selected].tail(156).copy()
    long = selected_history.melt(
        id_vars=["日期"], value_vars=["非商業淨部位", "商業淨部位"],
        var_name="交易人分類", value_name="淨部位",
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#94a3b8", strokeDash=[4, 4]).encode(y="y:Q")
    lines = alt.Chart(long).mark_line(strokeWidth=2.4).encode(
        x=alt.X("日期:T", title="報告日期"), y=alt.Y("淨部位:Q", title="淨部位（口）", scale=alt.Scale(zero=False)),
        color=alt.Color("交易人分類:N", title=None),
        tooltip=[alt.Tooltip("日期:T", format="%Y-%m-%d"), "交易人分類:N", alt.Tooltip("淨部位:Q", format=",.0f")],
    )
    st.altair_chart((lines + zero).properties(height=390, title=f"{selected}｜非商業與商業淨部位").interactive(), width="stretch")

    percentile_chart = alt.Chart(selected_history).mark_area(line={"color": "#f59e0b"}, color="#f59e0b", opacity=.22).encode(
        x=alt.X("日期:T", title="報告日期"), y=alt.Y("三年百分位%:Q", scale=alt.Scale(domain=[0, 100]), title="三年百分位 %"),
        tooltip=[alt.Tooltip("日期:T", format="%Y-%m-%d"), alt.Tooltip("三年百分位%:Q", format=".1f"), alt.Tooltip("淨部位週變化:Q", format="+,.0f")],
    ).properties(height=250, title="非商業淨部位三年百分位")
    st.altair_chart(percentile_chart.interactive(), width="stretch")

    row = latest.loc[latest["資產"] == selected].iloc[0]
    if row["三年百分位%"] >= 80:
        st.warning(f"{selected}投機淨多位於三年高檔，方向仍偏多，但反向消息出現時須留意多單擁擠與快速平倉。")
    elif row["三年百分位%"] <= 20:
        st.warning(f"{selected}投機淨部位位於三年低檔，方向偏空；若基本面改善，也要留意空單回補。")
    else:
        st.info(f"{selected}部位未達三年極端區，應搭配價格趨勢、美元、利率與庫存資料判讀。")

    st.markdown("[CFTC COT官方報告](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)｜[CFTC Legacy Futures Only資料集](https://publicreporting.cftc.gov/resource/srt6-5q2f)｜[CFTC資料說明](https://www.cftc.gov/MarketReports/CommitmentsofTraders/AbouttheCOTReports/index.htm)")
    st.caption("資料採Legacy Futures Only：非商業部位常作投機資金代理、商業部位常含避險需求，但分類不等同單一交易策略。")
