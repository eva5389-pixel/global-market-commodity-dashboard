"""USD/TWD observation model used by the market scenario dashboards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


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


def render_twd_observer(us_yield: float, usd_month_change: float, foreign_flow: float) -> None:
    st.header("台幣匯率觀察模板")
    st.caption("正分代表台幣升值壓力，負分代表台幣貶值壓力；這是方向模型，不是匯價點位預測。")

    with st.form("twd_fx_observer"):
        a, b, c, d = st.columns(4)
        with a:
            us_rate = st.number_input("美國利率／10年債殖利率 %", value=float(round(us_yield, 2)), step=0.05)
            tw_rate = st.number_input("台灣政策利率 %", value=2.00, step=0.05)
        with b:
            m1b = st.number_input("台灣 M1B 年增率 %", value=4.0, step=0.1)
            m2 = st.number_input("台灣 M2 年增率 %", value=5.0, step=0.1)
        with c:
            tw_exports = st.number_input("台灣出口年增率 %", value=10.0, step=0.5)
            export_orders = st.number_input("台灣外銷訂單年增率 %", value=8.0, step=0.5)
            kr_exports = st.number_input("韓國出口年增率 %", value=6.0, step=0.5)
        with d:
            foreign = st.number_input("外資台股買賣超（億元）", value=float(round(foreign_flow, 1)), step=10.0)
            usd_change = st.number_input("美元指數近1月變動 %", value=float(round(usd_month_change, 2)), step=0.1)
            tw_stance = st.selectbox("台灣央行態度", list(STANCE_SCORE), index=1)
            fed_stance = st.selectbox("Fed態度", list(STANCE_SCORE), index=1)
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
    st.caption("自動帶入：美國10年債殖利率、美元指數近1月變動、台灣三大法人合計。M1B、M2、出口、外銷訂單、韓國出口與央行態度採手動更新，避免發布頻率不同造成誤判。")
