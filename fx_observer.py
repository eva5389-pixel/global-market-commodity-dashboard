"""USD/TWD observation model used by the market scenario dashboards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from fx_data_sources import fetch_fx_public_data


STANCE_SCORE = {"偏鷹（抑制通膨／阻貶）": 3.0, "中性": 0.0, "偏鴿（支持景氣／寬鬆）": -3.0}


def calculate_twd_score(values: dict[str, float | str]) -> pd.DataFrame:
    """Return factor contributions; positive means TWD appreciation pressure."""
    rate_gap = float(values["美國利率"]) - float(values["台灣利率"])
    liquidity_gap = float(values["M1B年增率"]) - float(values["M2年增率"])
    export_average = (float(values["台灣出口年增率"]) + float(values["外銷訂單年增率"])) / 2
    export_advantage = float(values["台灣出口年增率"]) - float(values["韓國出口年增率"])
    rows = [
        ("台美利差", -np.clip((rate_gap - 1.0) * 2.4, -8, 8), f"{rate_gap:+.2f} 個百分點"),
        ("M1B－M2", np.clip(liquidity_gap * 0.45, -4, 4), f"{liquidity_gap:+.2f} 個百分點"),
        ("台灣出口／外銷訂單", np.clip(export_average * 0.22, -6, 6), f"平均 {export_average:+.2f}%"),
        ("台灣相對韓國出口", np.clip(export_advantage * 0.18, -4, 4), f"差距 {export_advantage:+.2f} 個百分點"),
        ("外資台股買賣超", np.clip(float(values["外資買賣超億元"]) / 120, -5, 5), f"{float(values['外資買賣超億元']):+,.0f} 億元"),
        ("台灣央行態度", STANCE_SCORE[str(values["台灣央行態度"])], str(values["台灣央行態度"])),
        ("Fed態度", -STANCE_SCORE[str(values["Fed態度"])], str(values["Fed態度"])),
        ("美元指數1M", np.clip(-float(values["美元指數1M變動"]) * 0.7, -5, 5), f"{float(values['美元指數1M變動']):+.2f}%"),
    ]
    frame = pd.DataFrame(rows, columns=["因子", "台幣方向分", "目前狀態"])
    frame["方向"] = np.where(frame["台幣方向分"] >= 0, "支持台幣升值", "造成台幣貶值壓力")
    return frame


def render_twd_observer(
    us_yield: float,
    usd_month_change: float,
    foreign_flow: float,
    live_dates: dict[str, str] | None = None,
) -> None:
    st.header("台幣匯率觀察模板")
    st.caption("正分代表台幣升值壓力，負分代表台幣貶值壓力；這是方向模型，不是匯價點位預測。")

    public = fetch_fx_public_data()
    live_dates = live_dates or {}
    automatic = st.toggle("自動載入最新公開資料", value=True, help="欄位仍可直接修改；修改後按下更新匯率判讀。")
    if automatic:
        st.success(f"已載入公開資料；本次檢查日期：{public['fetched_at']['period']}。各指標發布頻率不同，請以表格中的資料日期為準。")

    with st.form("twd_fx_observer"):
        a, b, c, d = st.columns(4)
        with a:
            us_rate = st.number_input("美國利率／10年債殖利率 %", value=float(round(us_yield, 2)), step=0.05)
            tw_rate = st.number_input("台灣政策利率 %", value=float(public["tw_rate"]["value"] if automatic else 2.0), step=0.05)
        with b:
            m1b = st.number_input("台灣 M1B 年增率 %", value=float(public["m1b"]["value"] if automatic else 4.0), step=0.1)
            m2 = st.number_input("台灣 M2 年增率 %", value=float(public["m2"]["value"] if automatic else 5.0), step=0.1)
        with c:
            tw_exports = st.number_input("台灣出口年增率 %", value=float(public["tw_exports"]["value"] if automatic else 10.0), step=0.5)
            export_orders = st.number_input("台灣外銷訂單年增率 %", value=float(public["tw_orders"]["value"] if automatic else 8.0), step=0.5)
            kr_exports = st.number_input("韓國出口年增率 %", value=float(public["kr_exports"]["value"] if automatic else 6.0), step=0.5)
        with d:
            foreign = st.number_input("外資台股買賣超（億元）", value=float(round(foreign_flow, 1)), step=10.0)
            usd_change = st.number_input("美元指數近1月變動 %", value=float(round(usd_month_change, 2)), step=0.1)
            tw_stance = st.selectbox("台灣央行態度", list(STANCE_SCORE), index=list(STANCE_SCORE).index(str(public["tw_stance"]["value"])) if automatic else 1)
            fed_stance = st.selectbox("Fed態度", list(STANCE_SCORE), index=list(STANCE_SCORE).index(str(public["fed_stance"]["value"])) if automatic else 1)
        submitted = st.form_submit_button("更新匯率判讀", type="primary", width="stretch")

    values = {
        "美國利率": us_rate, "台灣利率": tw_rate, "M1B年增率": m1b, "M2年增率": m2,
        "台灣出口年增率": tw_exports, "外銷訂單年增率": export_orders, "韓國出口年增率": kr_exports,
        "外資買賣超億元": foreign, "美元指數1M變動": usd_change,
        "台灣央行態度": tw_stance, "Fed態度": fed_stance,
    }
    detail = calculate_twd_score(values)
    score = float(detail["台幣方向分"].sum())
    if score >= 5:
        label, note = "台幣偏升", "美元兌台幣偏向下行"
    elif score <= -5:
        label, note = "台幣偏貶", "美元兌台幣偏向上行"
    else:
        label, note = "區間震盪", "升貶力量大致抵銷"

    k1, k2, k3 = st.columns(3)
    k1.metric("台幣方向分", f"{score:+.1f}", label, border=True)
    k2.metric("台美利差", f"{us_rate - tw_rate:+.2f}%", "擴大不利台幣", border=True)
    k3.metric("M1B－M2", f"{m1b - m2:+.2f}%", "正值代表資金活性較強", border=True)
    st.info(f"模型結論：**{label}**；{note}。請再搭配外資實際淨匯入與央行盤中調節確認。")
    st.bar_chart(detail, x="因子", y="台幣方向分", color="方向", horizontal=True)
    st.dataframe(
        detail,
        hide_index=True,
        column_config={"台幣方向分": st.column_config.NumberColumn(format="%+.2f")},
        width="stretch",
    )
    sources = [
        {"指標": "美國10年債殖利率", "目前值": f"{us_yield:.2f}%", "資料日期": live_dates.get("us_yield", "最新交易日"), "來源": "Yahoo Finance／美國公債", "狀態": "自動更新", "連結": "https://finance.yahoo.com/quote/%5ETNX/"},
        {"指標": "美元指數近1月", "目前值": f"{usd_month_change:+.2f}%", "資料日期": live_dates.get("usd", "最新交易日"), "來源": "Yahoo Finance／ICE美元指數", "狀態": "自動更新", "連結": "https://finance.yahoo.com/quote/DX-Y.NYB/"},
        {"指標": "外資台股買賣超", "目前值": f"{foreign_flow:+,.1f}億元", "資料日期": live_dates.get("foreign", "最新交易日"), "來源": "臺灣證券交易所", "狀態": "自動更新", "連結": "https://www.twse.com.tw/zh/trading/foreign/bfi82u.html"},
    ]
    labels = {
        "tw_rate": "台灣政策利率", "m1b": "台灣 M1B 年增率", "m2": "台灣 M2 年增率",
        "tw_exports": "台灣出口年增率", "tw_orders": "台灣外銷訂單年增率", "kr_exports": "韓國出口年增率",
        "tw_stance": "台灣央行態度", "fed_stance": "Fed態度",
    }
    for key, label_text in labels.items():
        item = public[key]
        suffix = "%" if key not in {"tw_stance", "fed_stance"} else ""
        sources.append({"指標": label_text, "目前值": f"{item['value']}{suffix}", "資料日期": item["period"], "來源": item["source"], "狀態": item["status"], "連結": item["url"]})
    st.subheader("資料來源與更新日期")
    st.dataframe(
        pd.DataFrame(sources), hide_index=True, width="stretch",
        column_config={"連結": st.column_config.LinkColumn("官方／原始資料連結", display_text="開啟來源 ↗")},
    )
    st.caption("自動值會快取 6 小時。『最新已核對值』代表官方網站阻擋雲端程式存取，系統保留最近一次核對資料並附上官方連結；所有欄位仍可手動覆寫。")


def _render_result(currency: str, quote_note: str, rows: list[tuple[str, float, str]]) -> None:
    detail = pd.DataFrame(rows, columns=["因子", "方向分", "目前狀態"])
    detail["方向"] = np.where(detail["方向分"] >= 0, f"支持{currency}升值", f"造成{currency}貶值壓力")
    score = float(detail["方向分"].sum())
    label = f"{currency}偏強" if score >= 5 else f"{currency}偏弱" if score <= -5 else "區間震盪"
    st.metric(f"{currency}方向分", f"{score:+.1f}", label, border=True)
    st.info(f"模型結論：**{label}**。{quote_note}")
    st.bar_chart(detail, x="因子", y="方向分", color="方向", horizontal=True)
    st.dataframe(
        detail, hide_index=True, width="stretch",
        column_config={"方向分": st.column_config.NumberColumn(format="%+.2f")},
    )


def render_usd_observer(usd_month_change: float) -> None:
    st.subheader("美元匯率觀察")
    st.caption("衡量美元整體強弱；正分代表美元偏強。")
    with st.form("usd_fx_observer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            fed = st.selectbox("Fed態度", list(STANCE_SCORE), index=1, key="usd_fed")
            dxy = st.number_input("美元指數近1月變動 %", value=float(round(usd_month_change, 2)), step=0.1, key="usd_dxy")
        with c2:
            growth_gap = st.number_input("美國相對其他國家成長差 %", value=0.5, step=0.1)
            inflation_gap = st.number_input("美國通膨相對目標差 %", value=0.5, step=0.1)
        with c3:
            risk = st.slider("全球避險程度", 0, 100, 50)
            twin_deficit = st.number_input("美國財政＋經常帳赤字壓力（0–10）", 0.0, 10.0, 5.0, 0.5)
        st.form_submit_button("更新美元判讀", type="primary", width="stretch")
    rows = [
        ("Fed態度", STANCE_SCORE[fed], fed),
        ("美元既有動能", np.clip(dxy * 0.8, -5, 5), f"{dxy:+.2f}%"),
        ("美國成長優勢", np.clip(growth_gap * 1.5, -4, 4), f"{growth_gap:+.2f}%"),
        ("通膨／高利率預期", np.clip(inflation_gap * 1.2, -3, 3), f"{inflation_gap:+.2f}%"),
        ("避險需求", np.clip((risk - 50) / 10, -5, 5), f"{risk}/100"),
        ("雙赤字壓力", -np.clip(twin_deficit * 0.45, 0, 4.5), f"{twin_deficit:.1f}/10"),
    ]
    _render_result("美元", "美元偏強通常使非美貨幣承壓。", rows)


def render_jpy_observer(us_yield: float, usd_month_change: float) -> None:
    st.subheader("日圓匯率觀察")
    st.caption("正分代表日圓升值；USD/JPY 通常反向下跌。")
    with st.form("jpy_fx_observer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            jp_rate = st.number_input("日本利率 %", value=0.75, step=0.05)
            us_rate = st.number_input("美國利率／10年債殖利率 %", value=float(round(us_yield, 2)), step=0.05, key="jpy_us")
        with c2:
            boj = st.selectbox("日本央行態度", list(STANCE_SCORE), index=1)
            wages = st.number_input("日本實質薪資年增率 %", value=0.0, step=0.1)
            inflation = st.number_input("日本核心通膨 %", value=2.0, step=0.1)
        with c3:
            risk = st.slider("全球避險程度", 0, 100, 50, key="jpy_risk")
            oil = st.number_input("油價近1月變動 %", value=0.0, step=0.5)
            dxy = st.number_input("美元指數近1月變動 %", value=float(round(usd_month_change, 2)), step=0.1, key="jpy_dxy")
        st.form_submit_button("更新日圓判讀", type="primary", width="stretch")
    gap = us_rate - jp_rate
    rows = [
        ("美日利差", -np.clip((gap - 2.0) * 2.0, -7, 7), f"{gap:+.2f} 個百分點"),
        ("日本央行態度", STANCE_SCORE[boj], boj),
        ("實質薪資", np.clip(wages * 0.8, -3, 3), f"{wages:+.2f}%"),
        ("通膨正常化", np.clip((inflation - 2.0) * 0.8, -2, 2), f"{inflation:.2f}%"),
        ("避險需求", np.clip((risk - 50) / 15, -3.3, 3.3), f"{risk}/100"),
        ("能源進口成本", -np.clip(oil * 0.25, -3, 3), f"{oil:+.2f}%"),
        ("美元壓力", -np.clip(dxy * 0.5, -4, 4), f"{dxy:+.2f}%"),
    ]
    _render_result("日圓", "日圓偏強＝USD/JPY偏向下行；日圓偏弱＝USD/JPY偏向上行。", rows)


def render_cny_observer(us_yield: float, usd_month_change: float) -> None:
    st.subheader("人民幣匯率觀察")
    st.caption("正分代表人民幣升值；USD/CNY與USD/CNH通常反向下跌。")
    with st.form("cny_fx_observer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            cn_rate = st.number_input("中國政策利率代理 %", value=1.40, step=0.05)
            us_rate = st.number_input("美國利率／10年債殖利率 %", value=float(round(us_yield, 2)), step=0.05, key="cny_us")
            pboc = st.selectbox("中國人民銀行態度", list(STANCE_SCORE), index=1)
        with c2:
            exports = st.number_input("中國出口年增率 %", value=5.0, step=0.5)
            pmi = st.number_input("中國製造業 PMI", value=50.0, step=0.1)
            property_pressure = st.slider("房地產／信用壓力", 0, 100, 50)
        with c3:
            capital_flow = st.number_input("資本流入強度（-10至10）", -10.0, 10.0, 0.0, 0.5)
            fixing_support = st.number_input("央行中間價支撐（0–10）", 0.0, 10.0, 5.0, 0.5)
            dxy = st.number_input("美元指數近1月變動 %", value=float(round(usd_month_change, 2)), step=0.1, key="cny_dxy")
        st.form_submit_button("更新人民幣判讀", type="primary", width="stretch")
    gap = us_rate - cn_rate
    rows = [
        ("中美利差", -np.clip((gap - 1.0) * 1.8, -7, 7), f"{gap:+.2f} 個百分點"),
        ("人行態度", STANCE_SCORE[pboc], pboc),
        ("出口結匯", np.clip(exports * 0.3, -4, 4), f"{exports:+.2f}%"),
        ("製造業景氣", np.clip((pmi - 50) * 0.8, -4, 4), f"{pmi:.1f}"),
        ("房地產／信用壓力", -np.clip((property_pressure - 30) / 14, -2, 5), f"{property_pressure}/100"),
        ("跨境資本流", np.clip(capital_flow * 0.5, -5, 5), f"{capital_flow:+.1f}/10"),
        ("中間價支撐", np.clip(fixing_support * 0.4, 0, 4), f"{fixing_support:.1f}/10"),
        ("美元壓力", -np.clip(dxy * 0.6, -4, 4), f"{dxy:+.2f}%"),
    ]
    _render_result("人民幣", "人民幣偏強＝USD/CNY偏向下行；離岸CNH通常波動較大。", rows)


def render_fx_observers(us_yield: float, usd_month_change: float, foreign_flow: float, live_dates: dict[str, str] | None = None) -> None:
    st.header("主要匯率觀察")
    twd_tab, usd_tab, jpy_tab, cny_tab = st.tabs(["🇹🇼 台幣", "🇺🇸 美元", "🇯🇵 日圓", "🇨🇳 人民幣"])
    with twd_tab:
        render_twd_observer(us_yield, usd_month_change, foreign_flow, live_dates)
    with usd_tab:
        render_usd_observer(usd_month_change)
    with jpy_tab:
        render_jpy_observer(us_yield, usd_month_change)
    with cny_tab:
        render_cny_observer(us_yield, usd_month_change)
