from __future__ import annotations

from io import StringIO

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st


BLSSERIES = {
    "CUSR0000SA0": "CPI",
    "CUSR0000SA0L1E": "核心CPI",
    "LNS14000000": "失業率",
    "CES0000000001": "非農就業",
}


@st.cache_data(ttl=21600, show_spinner=False)
def _fred(series_id: str, label: str) -> pd.DataFrame:
    start_date = (pd.Timestamp.now().normalize() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    response = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series_id, "cosd": start_date},
        headers={"User-Agent": "global-market-dashboard/1.0"},
        timeout=15,
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    frame.columns = ["Date", label]
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame[label] = pd.to_numeric(frame[label], errors="coerce")
    return frame.dropna().sort_values("Date")


@st.cache_data(ttl=21600, show_spinner=False)
def _bls_macro() -> dict[str, pd.DataFrame]:
    """Fetch CPI and labor data in one request from the official BLS API."""
    this_year = pd.Timestamp.now().year
    response = requests.post(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        json={"seriesid": list(BLSSERIES), "startyear": str(this_year - 3), "endyear": str(this_year)},
        headers={"User-Agent": "global-market-dashboard/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise ValueError("BLS request did not succeed")
    output: dict[str, pd.DataFrame] = {}
    for series in payload.get("Results", {}).get("series", []):
        label = BLSSERIES.get(series.get("seriesID"))
        if not label:
            continue
        rows = []
        for row in series.get("data", []):
            period = str(row.get("period", ""))
            if not period.startswith("M") or period == "M13":
                continue
            rows.append({
                "Date": pd.Timestamp(int(row["year"]), int(period[1:]), 1),
                label: pd.to_numeric(str(row.get("value", "")).replace(",", ""), errors="coerce"),
            })
        output[label] = pd.DataFrame(rows).dropna().sort_values("Date")
    if not all(label in output and not output[label].empty for label in BLSSERIES.values()):
        raise ValueError("BLS returned incomplete CPI or labor series")
    return output


@st.cache_data(ttl=21600, show_spinner=False)
def _ny_fed_effr() -> pd.DataFrame:
    end_date = pd.Timestamp.now().normalize()
    response = requests.get(
        "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json",
        params={"startDate": (end_date - pd.DateOffset(months=3)).strftime("%Y-%m-%d"), "endDate": end_date.strftime("%Y-%m-%d"), "type": "rate"},
        headers={"User-Agent": "global-market-dashboard/1.0"}, timeout=25,
    )
    response.raise_for_status()
    frame = pd.DataFrame(response.json().get("refRates", [])).rename(
        columns={"effectiveDate": "Date", "percentRate": "有效聯邦基金利率"}
    )
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["有效聯邦基金利率"] = pd.to_numeric(frame["有效聯邦基金利率"], errors="coerce")
    return frame.dropna(subset=["Date", "有效聯邦基金利率"]).sort_values("Date")


@st.cache_data(ttl=21600, show_spinner=False)
def _rate_inputs() -> dict[str, pd.DataFrame]:
    data = _bls_macro()
    try:
        data["初領失業金"] = _fred("ICSA", "初領失業金")
    except Exception:
        data["初領失業金"] = pd.DataFrame(columns=["Date", "初領失業金"])
    try:
        data["有效聯邦基金利率"] = _ny_fed_effr()
    except Exception:
        data["有效聯邦基金利率"] = pd.DataFrame(columns=["Date", "有效聯邦基金利率"])
    return data


def _last(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].iloc[-1])


def _date(frame: pd.DataFrame) -> str:
    return frame["Date"].iloc[-1].strftime("%Y-%m-%d")


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0, 100))


def render_us_rate_dashboard(default_rate: float = 4.0) -> None:
    st.header("美國升降息風險與市場壓力")
    st.caption(
        "OIS 反映市場定價，CPI與就業反映 Fed 的雙重使命。本頁將三者分開計分；"
        "OIS 報價通常來自授權終端，因此由使用者輸入，不偽裝成免費即時報價。"
    )

    try:
        data = _rate_inputs()
        cpi = data["CPI"].copy()
        core = data["核心CPI"].copy()
        cpi["CPI年增%"] = cpi["CPI"].pct_change(12) * 100
        core["核心CPI年增%"] = core["核心CPI"].pct_change(12) * 100
        cpi_yoy = float(cpi["CPI年增%"].dropna().iloc[-1])
        core_yoy = float(core["核心CPI年增%"].dropna().iloc[-1])
        unemployment = _last(data["失業率"], "失業率")
        payroll = data["非農就業"].copy()
        payroll["月增千人"] = payroll["非農就業"].diff()
        payroll_3m = float(payroll["月增千人"].dropna().tail(3).mean())
        claims_4w = float(data["初領失業金"]["初領失業金"].tail(4).mean() / 1000) if not data["初領失業金"].empty else np.nan
        effective_rate = _last(data["有效聯邦基金利率"], "有效聯邦基金利率") if not data["有效聯邦基金利率"].empty else float(default_rate)
        source_ok = True
    except Exception:
        source_ok = False
        cpi_yoy = core_yoy = unemployment = payroll_3m = claims_4w = np.nan
        effective_rate = float(default_rate)

    with st.form("us_ois_settings", border=True):
        st.markdown("**OIS／Fed 會議設定**")
        c1, c2, c3 = st.columns(3)
        current_mid = c1.number_input("目前政策利率中點 %", value=float(round(effective_rate, 2)), step=0.01)
        ois_rate = c2.number_input("下次會議 OIS 隱含政策利率 %", value=float(round(effective_rate, 2)), step=0.01)
        step_bps = c3.number_input("單次利率變動幅度 bps", min_value=5, max_value=100, value=25, step=5)
        st.form_submit_button("更新升降息判斷", type="primary", width="stretch")

    implied_bps = (ois_rate - current_mid) * 100
    hike_prob = _clip_score(max(implied_bps, 0) / step_bps * 100)
    cut_prob = _clip_score(max(-implied_bps, 0) / step_bps * 100)
    hold_prob = _clip_score(100 - hike_prob - cut_prob)

    inflation_pressure = _clip_score(
        50 + (cpi_yoy - 2) * 14 + (core_yoy - 2) * 18
    ) if source_ok else 50.0
    claims_component = (230 - claims_4w) * 0.12 if pd.notna(claims_4w) else 0
    labor_tightness = _clip_score(
        50 + (4.2 - unemployment) * 16 + (payroll_3m - 100) * 0.08 + claims_component
    ) if source_ok else 50.0
    ois_hawkishness = _clip_score(50 + implied_bps * 2)
    hike_pressure = _clip_score(ois_hawkishness * 0.50 + inflation_pressure * 0.30 + labor_tightness * 0.20)
    cut_pressure = _clip_score(100 - hike_pressure)

    cards = st.columns(5)
    cards[0].metric("OIS升息機率", f"{hike_prob:.1f}%", f"{implied_bps:+.1f} bps", border=True)
    cards[1].metric("OIS維持機率", f"{hold_prob:.1f}%", border=True)
    cards[2].metric("OIS降息機率", f"{cut_prob:.1f}%", border=True)
    cards[3].metric("綜合升息壓力", f"{hike_pressure:.1f}/100", border=True)
    cards[4].metric("綜合降息壓力", f"{cut_pressure:.1f}/100", border=True)

    if hike_pressure >= 65:
        st.error("偏鷹：OIS、通膨與就業組合顯示升息或高利率維持更久的壓力較高。")
    elif cut_pressure >= 65:
        st.success("偏鴿：OIS與基本面組合顯示降息壓力較高，但仍須留意通膨反彈。")
    else:
        st.info("中性／分歧：市場較偏向維持利率，或 OIS 與通膨、就業訊號互相抵銷。")

    if not source_ok:
        st.warning("BLS資料暫時無法取得；目前僅顯示OIS輸入結果，CPI與就業分數使用中性值。")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CPI年增", f"{cpi_yoy:.2f}%", _date(cpi), border=True)
    k2.metric("核心CPI年增", f"{core_yoy:.2f}%", _date(core), border=True)
    k3.metric("失業率", f"{unemployment:.2f}%", _date(data["失業率"]), border=True)
    claims_text = f"初領4週均 {claims_4w:,.0f}千" if pd.notna(claims_4w) else "初領資料暫缺"
    k4.metric("非農3月均增", f"{payroll_3m:,.0f}千人", claims_text, border=True)

    score_frame = pd.DataFrame({
        "構面": ["OIS偏鷹度", "通膨壓力", "就業緊俏度"],
        "分數": [ois_hawkishness, inflation_pressure, labor_tightness],
        "權重": [50, 30, 20],
    })
    bars = alt.Chart(score_frame).mark_bar(cornerRadiusEnd=5).encode(
        x=alt.X("分數:Q", scale=alt.Scale(domain=[0, 100]), title="壓力分數"),
        y=alt.Y("構面:N", sort=None, title=None),
        color=alt.Color("分數:Q", scale=alt.Scale(domain=[0, 50, 100], range=["#22c55e", "#f59e0b", "#ef4444"]), legend=None),
        tooltip=["構面:N", alt.Tooltip("分數:Q", format=".1f"), alt.Tooltip("權重:Q", format=".0f")],
    ).properties(height=190, title="升息壓力拆解")
    st.altair_chart(bars, width="stretch")

    inflation = pd.merge(cpi[["Date", "CPI年增%"]], core[["Date", "核心CPI年增%"]], on="Date", how="outer")
    inflation_long = inflation.tail(72).melt("Date", var_name="指標", value_name="年增率%")
    inflation_chart = alt.Chart(inflation_long.dropna()).mark_line(point=True).encode(
        x=alt.X("Date:T", title="日期"), y=alt.Y("年增率%:Q", title="年增率 %"), color="指標:N",
        tooltip=[alt.Tooltip("Date:T", title="日期"), "指標:N", alt.Tooltip("年增率%:Q", format=".2f")],
    ).properties(height=300, title="美國 CPI 與核心 CPI")
    st.altair_chart(inflation_chart.interactive(), width="stretch")

    st.markdown("**市場影響速覽**")
    st.dataframe(pd.DataFrame([
        {"情境": "升息／高利率更久", "美債": "短端殖利率偏升、價格承壓", "美元": "通常偏強", "美股": "高估值成長股折現壓力較大", "黃金": "實質利率上升時較不利"},
        {"情境": "維持利率", "美債": "等待數據、區間震盪", "美元": "視相對利差", "美股": "由獲利與估值主導", "黃金": "視實質利率與避險需求"},
        {"情境": "降息／寬鬆", "美債": "殖利率偏降、價格受惠", "美元": "通常偏弱", "美股": "軟著陸利多；衰退式降息未必", "黃金": "通常受惠於實質利率下降"},
    ]), hide_index=True, width="stretch")

    st.caption("OIS機率採單次、單一幅度的兩狀態換算；若市場定價多次行動或含期限溢酬，不能視為官方機率。所有分數均為研究用方向模型。")
    st.markdown("[BLS CPI與就業資料](https://www.bls.gov/data/)｜[FRED初領失業金](https://fred.stlouisfed.org/series/ICSA)｜[New York Fed EFFR](https://www.newyorkfed.org/markets/reference-rates/effr)｜[New York Fed SOFR](https://www.newyorkfed.org/markets/reference-rates/sofr)")
