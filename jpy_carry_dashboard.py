"""Japan rate, yen and carry-trade risk monitor."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


MARKET_SYMBOLS = {
    "美元／日圓": "JPY=X",
    "美元指數": "DX-Y.NYB",
    "日經225": "^N225",
    "日本銀行ETF": "1615.T",
    "日本出口股ETF": "1625.T",
    "VIX": "^VIX",
}


@st.cache_data(ttl=1800, max_entries=8, show_spinner=False)
def fetch_japan_carry_market(period: str = "1y") -> pd.DataFrame:
    """Return normalized market series; individual symbol failure does not stop the page."""
    rows: list[pd.DataFrame] = []
    for label, symbol in MARKET_SYMBOLS.items():
        try:
            raw = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close = pd.to_numeric(raw["Close"], errors="coerce").dropna()
            if close.empty:
                continue
            frame = close.rename("收盤價").reset_index()
            frame.columns = ["日期", "收盤價"]
            frame["指標"] = label
            frame["區間漲跌%"] = (frame["收盤價"] / frame["收盤價"].iloc[0] - 1) * 100
            rows.append(frame)
        except Exception:
            continue
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["日期", "收盤價", "指標", "區間漲跌%"]
    )


def implied_step_probability(current_rate: float, implied_rate: float, step_bps: int, direction: str) -> float:
    """Two-state meeting probability inferred from an OIS-implied policy rate."""
    step = max(step_bps, 1) / 100
    move = implied_rate - current_rate
    probability = move / step if direction == "升息" else -move / step
    return float(np.clip(probability * 100, 0, 100))


def carry_risk_score(
    boj_hike_prob: float,
    fed_cut_prob: float,
    us_jp_gap: float,
    jpy_1m: float,
    vix: float,
    leverage: float,
) -> tuple[float, pd.DataFrame]:
    """Transparent 0–100 risk score; higher means greater unwind pressure."""
    rows = [
        ("日銀升息機率", boj_hike_prob * 0.25, f"{boj_hike_prob:.1f}%"),
        ("Fed降息機率", fed_cut_prob * 0.15, f"{fed_cut_prob:.1f}%"),
        ("美日利差收斂", np.clip((3.5 - us_jp_gap) / 3.5 * 20, 0, 20), f"{us_jp_gap:.2f}個百分點"),
        ("日圓近1月升值", np.clip(max(jpy_1m, 0) * 2.0, 0, 20), f"{jpy_1m:+.2f}%"),
        ("市場波動", np.clip((vix - 12) / 28 * 10, 0, 10), f"VIX {vix:.1f}"),
        ("槓桿／擁擠度", np.clip(leverage, 0, 10), f"{leverage:.1f}/10"),
    ]
    detail = pd.DataFrame(rows, columns=["風險因子", "風險分", "目前狀態"])
    return float(np.clip(detail["風險分"].sum(), 0, 100)), detail


def render_jpy_carry_dashboard(default_us_rate: float = 4.0, default_dxy_change: float = 0.0) -> None:
    st.header("日圓、日銀 OIS 與 carry trade 風險")
    st.caption("OIS 欄位使用市場隱含政策利率自行換算單次升／降息機率；若隱含多次行動或含期限溢酬，須改用完整利率路徑，不可把本頁視為交易所官方機率。")

    with st.form("jpy_ois_carry_form", border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**日本 OIS／政策設定**")
            jp_rate = st.number_input("日本目前政策利率 %", value=0.75, step=0.05)
            jp_ois = st.number_input("下次會議 OIS 隱含政策利率 %", value=0.85, step=0.01)
            jp_step = st.selectbox("日本單次升息幅度", [10, 15, 25], index=2, format_func=lambda x: f"{x} bps")
        with c2:
            st.markdown("**美元 OIS／Fed 設定**")
            us_rate = st.number_input("美國目前政策利率／代理 %", value=float(round(default_us_rate, 2)), step=0.05)
            us_ois = st.number_input("下次會議 OIS 隱含政策利率 %", value=max(float(round(default_us_rate, 2)) - 0.10, 0.0), step=0.01)
            us_step = st.selectbox("Fed 單次降息幅度", [25, 50], format_func=lambda x: f"{x} bps")
        with c3:
            st.markdown("**市場壓力設定**")
            jpy_1m = st.number_input("日圓近1月升值幅度 %", value=2.0, step=0.5, help="日圓升值填正數；若以USD/JPY計算，應將其跌幅方向反轉。")
            vix = st.number_input("VIX", value=18.0, min_value=0.0, step=1.0)
            leverage = st.slider("Carry trade 槓桿／擁擠度", 0.0, 10.0, 5.0, 0.5)
        st.form_submit_button("更新 OIS 與風險判斷", type="primary", width="stretch")

    boj_prob = implied_step_probability(jp_rate, jp_ois, jp_step, "升息")
    fed_cut_prob = implied_step_probability(us_rate, us_ois, us_step, "降息")
    rate_gap = us_rate - jp_rate
    risk, risk_detail = carry_risk_score(boj_prob, fed_cut_prob, rate_gap, jpy_1m, vix, leverage)
    risk_label = "高風險" if risk >= 70 else "中高風險" if risk >= 50 else "中性" if risk >= 30 else "低風險"

    with st.container(horizontal=True):
        st.metric("日銀下次升息機率", f"{boj_prob:.1f}%", f"OIS {jp_ois:.2f}%", border=True)
        st.metric("Fed 下次降息機率", f"{fed_cut_prob:.1f}%", f"OIS {us_ois:.2f}%", border=True)
        st.metric("美日政策利差", f"{rate_gap:.2f}%", "收斂會削弱carry報酬", border=True)
        st.metric("Carry trade 風險", f"{risk:.0f}/100", risk_label, border=True)

    if risk >= 70:
        st.error("日圓升值、日美利差收斂與風險波動可能形成共振；注意槓桿部位平倉、全球股票下跌及高息貨幣回吐。")
    elif risk >= 50:
        st.warning("平倉風險升高，市場可能從出口股、高估值科技與高息貨幣，轉向日本銀行、內需或低槓桿資產。")
    else:
        st.info("目前模型未顯示全面性平倉壓力，但日銀政策意外或日圓急升仍可能快速改變風險。")

    st.subheader("風險因子拆解")
    st.bar_chart(risk_detail, x="風險因子", y="風險分", horizontal=True)
    st.dataframe(risk_detail, hide_index=True, width="stretch", column_config={"風險分": st.column_config.NumberColumn(format="%.1f")})

    st.subheader("各國相對利率與資金方向")
    rate_table = st.data_editor(
        pd.DataFrame([
            {"市場": "美國", "政策／短率%": us_rate}, {"市場": "日本", "政策／短率%": jp_rate},
            {"市場": "歐元區", "政策／短率%": 2.00}, {"市場": "英國", "政策／短率%": 3.75},
            {"市場": "澳洲", "政策／短率%": 3.60}, {"市場": "台灣", "政策／短率%": 2.00},
        ]),
        hide_index=True, width="stretch", key="relative_rate_editor",
        column_config={"政策／短率%": st.column_config.NumberColumn(format="%.2f%%", min_value=-2.0, max_value=20.0, step=0.05)},
    )
    rate_table["相對日本利差%"] = pd.to_numeric(rate_table["政策／短率%"], errors="coerce") - jp_rate
    rate_table["Carry吸引力"] = pd.cut(rate_table["相對日本利差%"], [-np.inf, 1, 2.5, np.inf], labels=["低", "中", "高"])
    st.dataframe(rate_table, hide_index=True, width="stretch", column_config={"相對日本利差%": st.column_config.NumberColumn(format="%+.2f%%")})

    dxy_score = np.clip(default_dxy_change * 0.7 + (us_rate - us_ois) * 8, -10, 10)
    st.subheader("對市場的傳導")
    impacts = pd.DataFrame([
        {"市場／資產": "日本出口股", "日圓升值": "偏空：海外獲利換算縮水", "日本升息": "偏空至中性：估值與融資成本上升", "Carry平倉": "偏空"},
        {"市場／資產": "日本銀行／保險", "日圓升值": "中性", "日本升息": "溫和升息通常偏多：利差改善", "Carry平倉": "短線可能隨大盤震盪"},
        {"市場／資產": "日本內需／進口", "日圓升值": "偏多：進口成本下降", "日本升息": "視負債與需求而定", "Carry平倉": "相對抗跌"},
        {"市場／資產": "全球科技／高估值股", "日圓升值": "可能反映去槓桿", "日本升息": "全球資金成本上升", "Carry平倉": "偏空且波動放大"},
        {"市場／資產": "高息貨幣／新興市場", "日圓升值": "融資貨幣回補", "日本升息": "利差優勢下降", "Carry平倉": "匯率與股市同時承壓"},
        {"市場／資產": "美元指數", "日圓升值": "日圓權重使DXY承壓", "Fed偏鷹／少降息": "支撐美元", "Carry平倉": "初期美元避險與日圓回補可能並存"},
    ])
    st.dataframe(impacts, hide_index=True, width="stretch")
    st.caption(f"美元方向輔助分：{dxy_score:+.1f}；正值偏美元強。美元OIS更偏鷹會支撐DXY，但若日圓急升，DXY可能因日圓權重而受到反向影響。")

    st.subheader("行情交叉驗證")
    history = fetch_japan_carry_market()
    if history.empty:
        st.warning("Yahoo Finance目前未回傳行情；OIS輸入與風險試算仍可使用。")
    else:
        selected = st.multiselect("顯示指標", list(MARKET_SYMBOLS), default=["美元／日圓", "美元指數", "日經225", "VIX"])
        chart_data = history.loc[history["指標"].isin(selected)]
        hover = alt.selection_point(nearest=True, on="pointerover", fields=["日期"], empty=False)
        base = alt.Chart(chart_data).encode(x=alt.X("日期:T", title="日期"), color=alt.Color("指標:N", title="指標"))
        lines = base.mark_line(strokeWidth=2.3).encode(y=alt.Y("區間漲跌%:Q", title="起點至今漲跌（%）"))
        points = base.mark_circle(size=80).encode(
            y="區間漲跌%:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("日期:T", format="%Y-%m-%d"), "指標:N", alt.Tooltip("收盤價:Q", format=",.3f"), alt.Tooltip("區間漲跌%:Q", format="+.2f")],
        ).add_params(hover)
        st.altair_chart((lines + points).properties(height=420).interactive(bind_y=False), width="stretch")

    st.subheader("資料與機率查核")
    st.markdown("[日本銀行金融政策會議](https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm)｜[日本銀行利率與市場資料](https://www.boj.or.jp/en/statistics/market/index.htm)｜[紐約聯準銀行市場隱含政策利率說明](https://www.newyorkfed.org/research/policy/rstar)｜[CME FedWatch](https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html)｜[ICE美元指數](https://www.ice.com/products/194/US-Dollar-Index-Futures)")
    st.caption("OIS報價通常來自授權終端或券商。此頁不偽造即時OIS：請輸入你取得的隱含政策利率；模型只做透明換算與市場傳導分析。")
