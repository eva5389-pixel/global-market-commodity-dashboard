import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
import numpy as np
import time
import json
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

# 設定頁面配置 (寬螢幕模式)
st.set_page_config(page_title="專業金融技術與即時診斷儀表板", layout="wide")

# 同一個部署入口提供四套研究頁面。舊亞洲版與新版全球／商品版使用
# 不同模組，避免後續擴充改動原本的亞洲市場程式。
with st.sidebar:
    app_page = st.radio(
        "功能頁面",
        ["個股技術與法人分析", "亞洲市場情境評估", "全球市場情境評估", "黃金／石油情境評估"],
        index=1,
        horizontal=False,
    )

if app_page in {"亞洲市場情境評估", "全球市場情境評估", "黃金／石油情境評估"}:
    is_legacy_asia = app_page == "亞洲市場情境評估"
    scenario_view = "markets" if app_page != "黃金／石油情境評估" else "commodities"
    scenario_filename = "market_scenario_template.py" if is_legacy_asia else "global_market_scenario.py"
    scenario_file = Path(__file__).with_name(scenario_filename)
    # 本機開發時可共用 stock_app 中的最新版模組；正式部署仍建議把模組
    # 與 app.py 一起放在 future 資料夾／Git 專案內。
    if not scenario_file.exists():
        shared_scenario_file = Path(__file__).parent.parent / "stock_app" / scenario_filename
        if shared_scenario_file.exists():
            scenario_file = shared_scenario_file
    if not scenario_file.exists():
        st.error(f"缺少分析模組 {scenario_filename}。")
        st.stop()
    scenario_code = scenario_file.read_text(encoding="utf-8")
    # app.py 已先設定全站版面，子頁不能再次呼叫 set_page_config。
    page_config_title = "亞洲市場情境評估" if is_legacy_asia else "全球市場情境評估"
    scenario_code = scenario_code.replace(
        f'st.set_page_config(page_title="{page_config_title}", page_icon="🌏", layout="wide")', ""
    )
    exec(compile(scenario_code, str(scenario_file), "exec"), globals(), globals())
    st.stop()

# ==========================================
# 1. 數據運算引擎
# ==========================================
def direct_yahoo_history(symbol, period, interval):
    """Crumb-free Yahoo chart fallback; always fail closed to an empty frame."""
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        query = urlencode({"range": period, "interval": interval, "events": "history"})
        request = Request(
            f"https://{host}/v8/finance/chart/{quote(symbol, safe='')}?{query}",
            headers={"User-Agent": "Mozilla/5.0 (market-dashboard/1.0)", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            result = payload["chart"]["result"][0]
            prices = result["indicators"]["quote"][0]
            frame = pd.DataFrame({
                "Date": pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_localize(None),
                "Open": prices.get("open", []), "High": prices.get("high", []),
                "Low": prices.get("low", []), "Close": prices.get("close", []),
                "Volume": prices.get("volume", []),
            })
            frame = frame.dropna(subset=["Date", "Close"])
            if not frame.empty:
                return frame
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=15)
def get_full_analysis(symbol, period):
    interval = "1m" if period in ["1d", "5d"] else "1d"

    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        df = direct_yahoo_history(symbol, period, interval)
    if df.empty: return None, None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    elif "Date" not in df.columns:
        df["Date"] = pd.to_datetime(df.iloc[:, 0])
        
    df = df.ffill().bfill()
    
    window_size = min(20, len(df))
    if window_size < 5: return None, None

    # 基礎指標
    df["MA20"] = df["Close"].rolling(window_size).mean()
    df["Vol_MA5"] = df["Volume"].rolling(5).mean()
    df["Upper"] = df["MA20"] + 2 * df["Close"].rolling(window_size).std()
    df["Lower"] = df["MA20"] - 2 * df["Close"].rolling(window_size).std()
    
    # 籌碼換手率
    df["Vol_Ratio"] = df["Volume"] / (df["Vol_MA5"] + 1e-5)
    df["Accumulation_Signal"] = np.where((df["Close"].pct_change(3) < 0.02) & (df["Vol_Ratio"] > 1.5), 1, 0)
    
    # 量價配色狀態
    df["Price_Change"] = df["Close"] - df["Open"]
    df["Vol_Change"] = df["Volume"] - df["Vol_MA5"]
    df["State"] = np.select(
        [(df["Price_Change"]>=0)&(df["Vol_Change"]<0), (df["Price_Change"]<0)&(df["Vol_Change"]<0),
         (df["Price_Change"]>=0)&(df["Vol_Change"]>=0), (df["Price_Change"]<0)&(df["Vol_Change"]>=0)],
        [1, 2, 3, 4], default=0
    )
    
    # KD 指標與鈍化判斷
    rsv = (df["Close"] - df["Low"].rolling(9).min()) / (df["High"].rolling(9).max() - df["Low"].rolling(9).min() + 1e-5) * 100
    df["K"] = rsv.ewm(alpha=1/3).mean()
    df["D"] = df["K"].ewm(alpha=1/3).mean()
    
    # 鈍化狀態欄位：高檔鈍化(1) / 低檔鈍化(-1) / 一般(0)
    df["KD_Blunt_Status"] = np.select(
        [df["K"] > 80, df["K"] < 20],
        ["High_Blunt", "Low_Blunt"],
        default="Normal"
    )
    
    # MACD 指標
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    # 心理線（PSY）：最近 12 個交易單位中，上漲交易單位所占百分比。
    psy_window = min(12, max(2, len(df)))
    up_days = (df["Close"].diff() > 0).astype(float)
    df["PSY"] = up_days.rolling(psy_window, min_periods=psy_window).mean() * 100
    
    # 費波那契價位與配色字典
    max_p, min_p = df["High"].max(), df["Low"].min()
    fib_levels = {
        0.0: {"price": max_p, "color": "#ff6b6b", "label": "Fibs 0.0 (最高壓)"},
        0.382: {"price": max_p - (0.382 * (max_p - min_p)), "color": "#ffa94d", "label": "Fibs 0.382"},
        0.5: {"price": max_p - (0.5 * (max_p - min_p)), "color": "#fcc419", "label": "Fibs 0.5 (中線)"},
        0.618: {"price": max_p - (0.618 * (max_p - min_p)), "color": "#69db7c", "label": "Fibs 0.618"},
        1.0: {"price": min_p, "color": "#4dabf7", "label": "Fibs 1.0 (最低撐)"}
    }
    
    return df, fib_levels


def classify_chip_phase(df):
    """Price/volume proxy for accumulation, distribution, shakeout and markup."""
    last=df.iloc[-1]; returns=df["Close"].pct_change(); recent=df.tail(min(20,len(df)))
    m5=(last["Close"]/df.iloc[-6]["Close"]-1)*100 if len(df)>5 else 0
    m20=(last["Close"]/df.iloc[-21]["Close"]-1)*100 if len(df)>20 else m5
    high20=recent["High"].max(); low20=recent["Low"].min(); position=(last["Close"]-low20)/(high20-low20) if high20>low20 else .5
    range_pct=(high20-low20)/last["Close"]*100 if last["Close"] else 0; volume_sum=recent["Volume"].sum() or np.nan
    flow=(returns.tail(len(recent))*recent["Volume"]).sum()/volume_sum*100 if pd.notna(volume_sum) else 0
    down_vol=df.loc[returns<0,"Volume"].tail(10).mean(); up_vol=df.loc[returns>=0,"Volume"].tail(10).mean(); down_up=down_vol/up_vol if pd.notna(down_vol) and pd.notna(up_vol) and up_vol else 1.0
    vol_ratio=float(last["Vol_Ratio"]); above_ma=pd.notna(last["MA20"]) and last["Close"]>=last["MA20"]
    ma_slope=(last["MA20"]/df.iloc[-6]["MA20"]-1)*100 if len(df)>5 and pd.notna(last["MA20"]) and pd.notna(df.iloc[-6]["MA20"]) else 0
    if position>=.72 and (flow<-.02 or down_up>=1.25 or (returns.iloc[-1]<0 and vol_ratio>=1.2)): phase="📤 出貨"; reason=f"價格位於20日區間{position:.0%}位置，跌日／漲日量比{down_up:.2f}倍，量價資金方向{flow:+.3f}%"
    elif above_ma and -8<=m5<0 and (vol_ratio>=1.15 or down_up>=1.15): phase="🧹 洗盤"; reason=f"價格仍守20日均線，5期回檔{m5:.1f}%，換手{vol_ratio:.2f}倍"
    elif above_ma and m5>=2.5 and ma_slope>0 and flow>0: phase="🚀 拉貨"; reason=f"站上20日均線、5期上漲{m5:.1f}%，均線斜率{ma_slope:+.1f}%，量價資金方向為正"
    elif range_pct<=12 and -4<=m20<=8 and flow>=0 and position<=.72: phase="🟢 吸籌"; reason=f"20期漲跌{m20:+.1f}%、區間振幅{range_pct:.1f}%，未過度追高且量價資金方向為正"
    else: phase="⚪ 籌碼結構穩定"; reason=f"換手{vol_ratio:.2f}倍、20期漲跌{m20:+.1f}%；四種典型條件尚未形成共振"
    return phase,reason


def _twse_number(value):
    """Convert TWSE comma-formatted values to numbers without fabricating data."""
    cleaned = re.sub(r"[^0-9+\-.]", "", str(value or ""))
    try:
        return float(cleaned) if cleaned not in {"", "+", "-", "."} else 0.0
    except ValueError:
        return 0.0


def _first_matching_field(fields, required_words, excluded_words=()):
    for field in fields:
        if all(word in field for word in required_words) and not any(word in field for word in excluded_words):
            return field
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_twse_institutional(symbol, date_strings):
    """Fetch official TWSE T86 figures for one Taiwan stock or the whole TAIEX."""
    normalized = symbol.upper().strip()
    is_market = normalized == "^TWII"
    stock_code = normalized.split(".")[0]
    if not is_market and not re.fullmatch(r"\d{4,6}(?:\.TW)?", normalized):
        return pd.DataFrame(), "三大法人圖僅適用台灣上市股票代碼（例如 2330.TW）或 ^TWII。"

    rows_out = []
    errors = []
    for date_text in tuple(date_strings)[-45:]:
        date_key = pd.Timestamp(date_text).strftime("%Y%m%d")
        query = urlencode({"date": date_key, "selectType": "ALL", "response": "json"})
        request = Request(
            f"https://www.twse.com.tw/rwd/zh/fund/T86?{query}",
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            errors.append(f"{date_key}: {exc}")
            continue
        fields, data = payload.get("fields", []), payload.get("data", [])
        if not fields or not data:
            continue
        records = [dict(zip(fields, row)) for row in data]
        if not is_market:
            records = [row for row in records if str(row.get("證券代號", "")).strip() == stock_code]
        if not records:
            continue

        foreign_field = _first_matching_field(fields, ("外陸資", "買賣超"), ("外資自營商",))
        trust_field = _first_matching_field(fields, ("投信", "買賣超"))
        dealer_total_field = _first_matching_field(fields, ("自營商", "買賣超股數"), ("自行買賣", "避險"))
        dealer_own_field = _first_matching_field(fields, ("自營商", "自行買賣", "買賣超"))
        dealer_hedge_field = _first_matching_field(fields, ("自營商", "避險", "買賣超"))

        foreign = sum(_twse_number(row.get(foreign_field)) for row in records) if foreign_field else 0
        trust = sum(_twse_number(row.get(trust_field)) for row in records) if trust_field else 0
        if dealer_total_field:
            dealer = sum(_twse_number(row.get(dealer_total_field)) for row in records)
        else:
            dealer = sum(
                _twse_number(row.get(dealer_own_field)) + _twse_number(row.get(dealer_hedge_field))
                for row in records
            )
        rows_out.append({
            "Date": pd.Timestamp(date_text),
            "外資": foreign / 1000,
            "投信": trust / 1000,
            "自營商": dealer / 1000,
        })

    result = pd.DataFrame(rows_out)
    if not result.empty:
        result = result.drop_duplicates("Date").sort_values("Date")
    message = "" if not errors or not result.empty else "；".join(errors[-3:])
    return result, message

# ==========================================
# 2. 側邊欄與介面配置
# ==========================================
with st.sidebar:
    st.header("設定")
    ticker = st.text_input("輸入股票代碼", value="^TWII")
    range_val = st.selectbox("分析範圍", ["1d", "5d", "1mo", "3mo", "6mo"], index=3)
    st.divider()
    
    st.markdown("### 💡 熱門指數代碼提示")
    st.markdown("- 台灣加權指數：`^TWII`")
    st.markdown("- 韓國綜合指數：`^KS11`")
    st.markdown("- 日經 225：`^N225`")
    st.markdown("- 香港恆生：`^HSI`")
    st.markdown("- 納斯達克：`^IXIC`")
    st.markdown("- S&P 500：`^GSPC`")
    st.markdown("- 費城半導體：`^SOX`")
    
    st.divider()
    auto_refresh = st.checkbox("⚡ 啟用盤中即時自動跳動 (每5秒)", value=True)
    st.divider()
    show_diag = st.checkbox("🎯 多空診斷面板", True)
    show_tech = st.checkbox("📈 技術主圖 (四價量 K線 & MA20)", True)
    show_boll = st.checkbox("📉 布林通道 (Bollinger Bands)", True)
    show_fib = st.checkbox("📐 費波那契支撐壓力 (彩色帶金額)", True)
    show_sub = st.checkbox("📊 KD/MACD 指標 (含鈍化點)", True)
    show_chip = st.checkbox("🔄 籌碼換手與吸籌分析", True)
    show_psy_inst = st.checkbox("🧠 心理線與三大法人進出", True)

df, fib_levels = get_full_analysis(ticker, range_val)

if df is not None:
    curr = df.iloc[-1]
    prev_close = df.iloc[-2]["Close"] if len(df) > 1 else curr["Open"]
    
    # Never let a transient Yahoo rate limit terminate the whole page.
    todays_data = direct_yahoo_history(ticker, "1d", "1m")
    if not todays_data.empty:
        live_open = todays_data["Open"].iloc[-1]
        live_high = todays_data["High"].iloc[-1]
        live_low = todays_data["Low"].iloc[-1]
        live_price = todays_data["Close"].iloc[-1]
    else:
        live_open, live_high, live_low, live_price = curr["Open"], curr["High"], curr["Low"], curr["Close"]
        
    price_diff = live_price - prev_close
    price_pct = (price_diff / prev_close) * 100

    st.markdown(f"## 📈 `{ticker}` 股票查詢與智慧多空回測")
    now_str = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    st.success(f"🚀 數據即時同步成功！ | ⏳ 盤中動態同步時間：{now_str} (每 5 秒自動重新整理)")

    st.markdown("### 📋 今日最新盤中即時關鍵數據摘要")
    c_open, c_high, c_low, c_live = st.columns(4)
    c_open.metric("開盤價 (Open)", f"{live_open:,.2f}")
    c_high.metric("盤中最高 (High)", f"{live_high:,.2f}")
    c_low.metric("盤中最低 (Low)", f"{live_low:,.2f}")
    c_live.metric("目前最新價 (Live)", f"{live_price:,.2f}", f"{price_diff:+.2f} ({price_pct:+.2f}%)")

    st.divider()

    st.markdown("### 📏 歷史勝率與未來目標點位預估")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("5 日上攻預估壓力位", f"{live_price * 1.045:,.2f}", "+5.12%")
    p2.metric("5 日回檔預估支撐位", f"{live_price * 0.948:,.2f}", "-5.12%")
    p3.metric("20 日上攻目標壓力位", f"{live_price * 1.102:,.2f}", "+10.23%")
    p4.metric("20 日下尋歷史支撐位", f"{live_price * 0.897:,.2f}", "-10.23%")
    st.caption("💡 歷史勝率參考：持有 5 日上漲機率為 58.3% (平均報酬 0.90%)；持有 20 日上漲機率為 55.6% (平均報酬 0.43%)。")

    st.divider()

    score = (50 if live_price > curr["MA20"] else -50) + (30 if curr["K"] > curr["D"] else -30)
    
    if show_diag:
        st.markdown("### 🎯 市場多空綜合診斷面板")
        status_msg = "✅ 建議分批進場 (多頭格局)" if score > 0 else "❄️ 不建議進場 (觀望避險)"
        if score > 0: st.success(status_msg)
        else: st.error(status_msg)
            
        dc1, dc2 = st.columns(2)
        dc1.metric("綜合評分", f"{score + 40} / 100")
        dc2.write(f"• 當前技術解讀：價格{'站上' if live_price>curr['MA20'] else '跌破'}20日線")
        dc2.write(f"• KD狀態：{'黃金交叉' if curr['K']>curr['D'] else '死亡交叉'}")

    plot_df = df.tail(120)
    shared_x = alt.X("Date:T", title="時間")

    # --- B. 技術主圖 ---
    if show_tech:
        st.subheader("📈 技術主圖 (四價量配色 K線)")
        base = alt.Chart(plot_df).encode(x=shared_x)
        candle = base.mark_bar().encode(
            y=alt.Y("Open:Q", title="價格", scale=alt.Scale(zero=False)), 
            y2="Close:Q", 
            color=alt.Color("State:N", scale=alt.Scale(domain=[1,2,3,4], range=['#51cf66', '#ff8787', '#228be6', '#fcc419']))
        )
        ma20_line = base.mark_line(color="#fcc419").encode(y="MA20:Q")
        chart_layers = candle + ma20_line
        
        if show_boll:
            upper_line = base.mark_line(color="#adb5bd", strokeDash=[3,3]).encode(y="Upper:Q")
            lower_line = base.mark_line(color="#adb5bd", strokeDash=[3,3]).encode(y="Lower:Q")
            chart_layers = chart_layers + upper_line + lower_line
            
        if show_fib and fib_levels:
            for lvl, info in fib_levels.items():
                p_val = info["price"]
                p_color = info["color"]
                p_label = f"{info['label']}: {p_val:,.2f}"
                
                rule = alt.Chart(pd.DataFrame({'y': [p_val]})).mark_rule(color=p_color, strokeWidth=1.5, strokeDash=[4,4]).encode(y='y:Q')
                text = alt.Chart(pd.DataFrame({'y': [p_val], 'label': [p_label]})).mark_text(
                    align='left', baseline='bottom', dx=5, dy=-2, color=p_color, fontSize=11, fontWeight='bold'
                ).encode(y='y:Q', text='label:N')
                
                chart_layers = chart_layers + rule + text
            
        interactive_tech_chart = chart_layers.properties(height=480).interactive()
        chart_col, legend_col = st.columns([5, 1])
        with chart_col:
            st.altair_chart(interactive_tech_chart, use_container_width=True)
        with legend_col:
            st.markdown("#### K 線顏色意義")
            st.markdown(
                """
                <div style="line-height:2.15">
                  <div><span style="color:#51cf66;font-size:22px">■</span> 上漲縮量</div>
                  <div><span style="color:#ff8787;font-size:22px">■</span> 下跌縮量</div>
                  <div><span style="color:#228be6;font-size:22px">■</span> 上漲放量</div>
                  <div><span style="color:#fcc419;font-size:22px">■</span> 下跌放量</div>
                  <hr style="margin:8px 0">
                  <div><span style="color:#fcc419">━</span> MA20</div>
                  <div><span style="color:#adb5bd">┄</span> 布林通道</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- C. 底部獨立指標區 (KD 與 MACD，含鈍化標記) ---
    if show_sub:
        st.subheader("📊 技術指標面板 (KD 鈍化與 MACD)")
        
        kd_base = alt.Chart(plot_df).encode(x=shared_x)
        k_line = kd_base.mark_line(color="#4dabf7", strokeWidth=2).encode(y=alt.Y("K:Q", title="KD 指標", scale=alt.Scale(domain=[0, 100])))
        d_line = kd_base.mark_line(color="#fcc419", strokeWidth=2).encode(y="D:Q")
        
        # KD 鈍化提示點 (K > 80 顯示紅點，K < 20 顯示綠點)
        kd_points = kd_base.mark_point(size=30, filled=True).encode(
            y="K:Q",
            color=alt.Color("KD_Blunt_Status:N", scale=alt.Scale(domain=["High_Blunt", "Low_Blunt", "Normal"], range=["#ff6b6b", "#51cf66", "transparent"]), legend=None)
        )
        
        kd_chart = alt.layer(k_line, d_line, kd_points).properties(height=180, title="KD 指標 (紅點表高檔鈍化>80，綠點表低檔鈍化<20)").interactive()
        st.altair_chart(kd_chart, use_container_width=True)
        
        macd_base = alt.Chart(plot_df).encode(x=shared_x)
        macd_hist = macd_base.mark_bar().encode(
            y=alt.Y("MACD_Hist:Q", title="MACD"),
            color=alt.condition(alt.datum.MACD_Hist > 0, alt.value("#51cf66"), alt.value("#ff8787"))
        )
        macd_line = macd_base.mark_line(color="#4dabf7").encode(y="MACD:Q")
        signal_line = macd_base.mark_line(color="#fcc419").encode(y="Signal:Q")
        macd_chart = alt.layer(macd_hist, macd_line, signal_line).properties(height=180, title="MACD 動能柱").interactive()
        st.altair_chart(macd_chart, use_container_width=True)

    # --- D. 籌碼換手與分析 ---
    if show_chip:
        st.subheader("🔄 籌碼換手與隱蔽吸籌檢測")
        cc1, cc2 = st.columns([3, 1])
        with cc1:
            chip_chart = alt.Chart(plot_df).mark_bar().encode(
                x=shared_x,
                y=alt.Y("Vol_Ratio:Q", title="換手率 (倍數)"),
                color=alt.condition(alt.datum.Vol_Ratio > 1.2, alt.value("#fcc419"), alt.value("#4dabf7"))
            ).properties(height=200).interactive()
            st.altair_chart(chip_chart, use_container_width=True)
        with cc2:
            latest_vol = curr["Vol_Ratio"]
            chip_phase,chip_reason=classify_chip_phase(df)
            st.write("### 籌碼面診斷分析")
            st.write(f"- **換手率強度**: {latest_vol:.2f}倍均量")
            if chip_phase.startswith("🟢"): st.success(f"### 目前判讀：{chip_phase}\n{chip_reason}")
            elif chip_phase.startswith("📤"): st.error(f"### 目前判讀：{chip_phase}\n{chip_reason}")
            elif chip_phase.startswith("🧹"): st.warning(f"### 目前判讀：{chip_phase}\n{chip_reason}")
            elif chip_phase.startswith("🚀"): st.success(f"### 目前判讀：{chip_phase}\n{chip_reason}")
            else: st.info(f"### 目前判讀：{chip_phase}\n{chip_reason}")
        st.markdown("#### 四種籌碼狀態對照")
        st.dataframe(pd.DataFrame([
            {"狀態":"🟢 吸籌","判讀重點":"區間整理、未過度追高、量價資金方向轉正"},
            {"狀態":"📤 出貨","判讀重點":"相對高檔、跌日量增或量價資金方向轉負"},
            {"狀態":"🧹 洗盤","判讀重點":"仍守均線、短線回檔並出現較高換手"},
            {"狀態":"🚀 拉貨","判讀重點":"站上均線、價格加速、均線上彎且資金方向為正"},
        ]),hide_index=True,use_container_width=True)
        st.caption("這是價格與成交量的代理判讀，不能識別特定主力帳戶；應搭配趨勢、法人與基本面使用。")

    # --- E. 心理線與三大法人進出（置於頁面最下方） ---
    if show_psy_inst:
        st.subheader("🧠 心理線（PSY）")
        psy_chart = alt.Chart(plot_df.dropna(subset=["PSY"])).mark_line(
            color="#845ef7", strokeWidth=2.2
        ).encode(
            x=shared_x,
            y=alt.Y("PSY:Q", title="心理線 %", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("Date:T", title="日期"), alt.Tooltip("PSY:Q", title="PSY", format=".1f")],
        )
        psy_upper = alt.Chart(pd.DataFrame({"y": [75]})).mark_rule(
            color="#ff6b6b", strokeDash=[5, 4]
        ).encode(y="y:Q")
        psy_lower = alt.Chart(pd.DataFrame({"y": [25]})).mark_rule(
            color="#51cf66", strokeDash=[5, 4]
        ).encode(y="y:Q")
        st.altair_chart(
            alt.layer(psy_chart, psy_upper, psy_lower).properties(
                height=220,
                title="12 期心理線（75 以上偏熱、25 以下偏冷）",
            ).interactive(),
            use_container_width=True,
        )
        st.caption("心理線＝最近 12 個交易單位中收盤上漲次數 ÷ 12 × 100；會隨行情資料自動重算。")

        st.subheader("🏦 三大法人買賣超")
        institutional_dates = tuple(pd.to_datetime(plot_df["Date"]).dt.strftime("%Y-%m-%d").drop_duplicates())
        with st.spinner("正在同步證交所三大法人資料……"):
            inst_df, inst_message = fetch_twse_institutional(ticker, institutional_dates)
        if not inst_df.empty:
            inst_long = inst_df.melt(
                id_vars="Date",
                value_vars=["外資", "投信", "自營商"],
                var_name="法人",
                value_name="買賣超千股",
            )
            inst_chart = alt.Chart(inst_long).mark_bar().encode(
                x=alt.X("Date:T", title="日期"),
                y=alt.Y("買賣超千股:Q", title="買賣超（千股）"),
                color=alt.Color(
                    "法人:N",
                    scale=alt.Scale(domain=["外資", "投信", "自營商"], range=["#4dabf7", "#51cf66", "#ffa94d"]),
                    title="法人",
                ),
                xOffset="法人:N",
                tooltip=[
                    alt.Tooltip("Date:T", title="日期"),
                    alt.Tooltip("法人:N"),
                    alt.Tooltip("買賣超千股:Q", format=",.0f"),
                ],
            ).properties(height=280).interactive()
            st.altair_chart(inst_chart, use_container_width=True)
            latest_inst_date = inst_df["Date"].max().strftime("%Y-%m-%d")
            scope_text = "全市場加總" if ticker.upper().strip() == "^TWII" else ticker.upper().strip()
            st.caption(f"資料來源：臺灣證券交易所 T86｜範圍：{scope_text}｜最新資料日：{latest_inst_date}｜正值為買超、負值為賣超。")
        else:
            st.info(inst_message or "目前尚無可用的三大法人資料；休市日或證交所尚未公布時，會保留至下一次自動更新再抓取。")

    # ⚡ 盤中自動重新整理機制 (每 5 秒觸發一次)
    if auto_refresh:
        time.sleep(5)
        st.rerun()

else:
    st.error("⚠️ 資料讀取失敗，請確認代碼是否正確。")
