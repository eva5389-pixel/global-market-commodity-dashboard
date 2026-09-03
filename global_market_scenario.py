"""全球主要市場情境分析 Streamlit 模板。

執行：streamlit run market_scenario_template.py
ETF 資金流是價量代理，不是實際申購／贖回金額；評分僅供研究。
"""
from __future__ import annotations

import re
import html
from datetime import date, datetime, timedelta
from io import BytesIO, StringIO
from urllib.parse import urlencode

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="全球市場情境評估", page_icon="🌏", layout="wide")
SCENARIO_VIEW = globals().get("scenario_view", "markets")

MARKETS = {
    "台灣": {"index": "^TWII", "etf": "EWT", "imf": "TWN", "sectors": {"大型權值": "0050.TW", "科技／半導體": "0052.TW", "金融": "0055.TW"}},
    "日本": {"index": "^N225", "etf": "EWJ", "imf": "JPN", "sectors": {"大型權值": "1306.T", "電子科技": "1625.T", "銀行": "1615.T"}},
    "韓國": {"index": "^KS11", "etf": "EWY", "imf": "KOR", "sectors": {"大型權值": "069500.KS", "半導體": "091160.KS", "金融": "091170.KS"}},
    "中國": {"index": "000001.SS", "etf": "MCHI", "imf": "CHN", "sectors": {"大型權值": "510300.SS", "半導體": "512480.SS", "銀行": "512800.SS"}},
    "香港": {"index": "^HSI", "etf": "EWH", "imf": "HKG", "sectors": {"大型權值": "2800.HK", "科技": "3033.HK", "金融": "2829.HK"}},
    "美國": {"index": "^GSPC", "etf": "SPY", "imf": "USA", "sectors": {"大型權值": "SPY", "科技": "QQQ", "金融": "XLF"}},
    "英國": {"index": "^FTSE", "etf": "EWU", "imf": "GBR", "sectors": {"大型權值": "ISF.L", "科技": "IITU.L", "金融": "IUFS.L"}},
    "法國": {"index": "^FCHI", "etf": "EWQ", "imf": "FRA", "sectors": {"大型權值": "CAC.PA", "科技": "TNO.PA", "金融": "BNK.PA"}},
    "德國": {"index": "^GDAXI", "etf": "EWG", "imf": "DEU", "sectors": {"大型權值": "EXS1.DE", "科技": "EXV3.DE", "金融": "EXV1.DE"}},
    "印度": {"index": "^NSEI", "etf": "INDA", "imf": "IND", "sectors": {"大型權值": "NIFTYBEES.NS", "科技": "ITBEES.NS", "金融": "BANKBEES.NS"}},
    "印尼": {"index": "^JKSE", "etf": "EIDO", "imf": "IDN", "sectors": {"大型權值": "XIIT.JK", "金融": "XIML.JK", "消費": "XIIC.JK"}},
    "澳洲": {"index": "^AXJO", "etf": "EWA", "imf": "AUS", "sectors": {"大型權值": "STW.AX", "科技": "TECH.AX", "金融": "MVB.AX"}},
    "巴西": {"index": "^BVSP", "etf": "EWZ", "imf": "BRA", "sectors": {"大型權值": "BOVA11.SA", "科技": "TECK11.SA", "金融": "FIND11.SA"}},
}
IMF_CODES = {"NGDP_RPCH": "實質GDP成長%", "PCPIPCH": "CPI年增率%", "TX_RPCH": "出口量成長%", "TM_RPCH": "進口量成長%"}
SCENARIOS = {
    "基準／軟著陸": {"rate": -25, "stock": 4, "bond": 2, "bias": {"台灣": 5, "日本": 5, "韓國": 4, "中國": 2, "香港": 2}, "sector": {"科技": 7, "半導體": 7, "金融": 1, "銀行": 1}},
    "降息／流動性寬鬆": {"rate": -100, "stock": 8, "bond": 6, "bias": {"台灣": 8, "日本": 4, "韓國": 7, "中國": 5, "香港": 7}, "sector": {"科技": 12, "半導體": 12, "金融": -3, "銀行": -4}},
    "升息／高利率更久": {"rate": 100, "stock": -7, "bond": -6, "bias": {"台灣": -6, "日本": 3, "韓國": -5, "中國": -3, "香港": -5}, "sector": {"科技": -10, "半導體": -9, "金融": 7, "銀行": 8}},
    "股債雙殺／通膨再起": {"rate": 125, "stock": -15, "bond": -10, "bias": {"台灣": -10, "日本": -3, "韓國": -9, "中國": -5, "香港": -7}, "sector": {"科技": -14, "半導體": -12, "金融": 3, "銀行": 4}},
    "景氣衰退／風險趨避": {"rate": -150, "stock": -18, "bond": 10, "bias": {"台灣": -9, "日本": -5, "韓國": -10, "中國": -6, "香港": -8}, "sector": {"科技": -10, "半導體": -12, "金融": -10, "銀行": -10}},
}

# 免費行情源無法穩定提供所有授權指數，因此個別項目保留明確代理標示。
# DAX「農金」依官方名稱解讀為 DAXglobal Agribusiness；若使用者原意為
# DAXglobal Gold Miners，可在這裡將 MOO 改成對應的授權資料源。
COMMODITY_ASSETS = {
    "黃金期貨指數": {"symbol": "GC=F", "group": "黃金／礦業", "type": "黃金期貨近月連續行情", "source": "Yahoo Finance"},
    "費城金銀指數": {"symbol": "^XAU", "group": "黃金／礦業", "type": "原指數", "source": "Nasdaq PHLX／Yahoo Finance"},
    "DAXglobal 農業企業指數": {"symbol": "MOO", "group": "黃金／礦業", "type": "MOO 農業企業ETF代理", "source": "STOXX 指數定義／Yahoo Finance"},
    "彭博世界礦業指數": {"symbol": "PICK", "group": "黃金／礦業", "type": "PICK 全球金屬礦業ETF代理", "source": "Bloomberg 指數名稱／Yahoo Finance"},
    "西德州原油期貨": {"symbol": "CL=F", "group": "石油／能源", "type": "WTI原油期貨近月連續行情", "source": "NYMEX／Yahoo Finance"},
    "NYSE Arca Oil 指數": {"symbol": "^XOI", "group": "石油／能源", "type": "原指數", "source": "ICE NYSE／Yahoo Finance"},
}

GLOBAL_FACTORS = {
    "美元指數": {"symbol": "DX-Y.NYB", "unit": "", "description": "美元走強通常壓抑亞洲資金與非美貨幣"},
    "WTI原油": {"symbol": "CL=F", "unit": "美元／桶", "description": "油價上升增加多數亞洲進口國成本"},
    "黃金": {"symbol": "GC=F", "unit": "美元／盎司", "description": "與VIX、美元合看，可觀察避險及通膨需求"},
    "比特幣": {"symbol": "BTC-USD", "unit": "美元", "description": "短期常反映流動性與高風險資產偏好，不視為純避險資產"},
    "美國10年債殖利率": {"symbol": "^TNX", "unit": "%", "description": "殖利率上升提高全球折現率與資金成本"},
    "VIX恐慌指數": {"symbol": "^VIX", "unit": "", "description": "風險趨避升高通常不利亞洲風險資產"},
}

# 模型敏感係數，不等同出口/GDP實際百分比；反映出口循環對企業獲利與股市結構的傳導。
EXPORT_SENSITIVITY = {"台灣":1.15,"韓國":1.05,"日本":.80,"中國":.85,"香港":1.10,"美國":.45,"英國":.55,"法國":.65,"德國":.95,"印度":.40,"印尼":.75,"澳洲":.85,"巴西":.80}
VOLATILITY_SOURCES = {
    "台灣":{"name":"TAIWAN VIX","url":"https://www.taifex.com.tw/cht/7/vixDaily3MNew"},
    "日本":{"name":"Nikkei 225 VI","url":"https://indexes.nikkei.co.jp/en/nkave/index/profile?idx=nk225vi"},
    "韓國":{"name":"US VIX（依指定）","url":"https://www.cboe.com/tradable_products/vix/"},
    "香港":{"name":"VHSI／HSI波動率代理","url":"https://www.hsi.com.hk/eng/indexes/all-indexes/volatilityindex"},
    "中國":{"name":"上證選擇權／上證指數波動率代理","url":"https://www.sse.com.cn/home/wechat/stockOption/stockOptionPage/"},
    "美國":{"name":"S&P 500 20日實現波動率","url":"https://www.cboe.com/tradable_products/vix/"},
    "英國":{"name":"FTSE 100 20日實現波動率","url":"https://www.lseg.com/en/ftse-russell/indices/uk"},
    "法國":{"name":"CAC 40 20日實現波動率","url":"https://live.euronext.com/en/product/indices/FR0003500008-XPAR"},
    "德國":{"name":"DAX 20日實現波動率","url":"https://www.stoxx.com/index-details?symbol=DAX"},
    "印度":{"name":"NIFTY 50 20日實現波動率","url":"https://www.niftyindices.com/indices/equity/broad-based-indices/NIFTY-50"},
    "印尼":{"name":"IDX Composite 20日實現波動率","url":"https://www.idx.co.id/en/market-data/statistical-reports/digital-statistic/market-indexes"},
    "澳洲":{"name":"S&P/ASX 200 20日實現波動率","url":"https://www.spglobal.com/spdji/en/indices/equity/sp-asx-200/"},
    "巴西":{"name":"Ibovespa 20日實現波動率","url":"https://www.b3.com.br/en_us/market-data-and-indices/indices/broad-indices/ibovespa.htm"},
}

OPTICAL_FUND_WATCHLIST = [
    {"基金":"野村台灣運籌基金","相關持股%":13.42,"實際持股":"旺矽 5.48%、聯亞 3.22%、華星光 2.92%、光聖 1.80%","持股期別":"2026/03","持股／績效連結":"https://tfccbank.moneydj.com/w/wr/wr04_ACKH03.djhtm"},
    {"基金":"富邦日盛基金","相關持股%":10.04,"實際持股":"光聖 4.11%、旺矽 3.95%、華星光 1.98%","持股期別":"2026/06","持股／績效連結":"https://invest2.hontai.com.tw/w/wr/wr04_ACJS01.djhtm"},
    {"基金":"群益店頭市場基金","相關持股%":10.27,"實際持股":"旺矽 7.99%、華星光 2.28%","持股期別":"公開說明書最新揭露","持股／績效連結":"https://www.moneydj.com/"},
    {"基金":"安聯台灣科技基金","相關持股%":7.61,"實際持股":"旺矽 7.61%","持股期別":"2026/07","持股／績效連結":"https://mmafund.sinopac.com/w/wr/wr04.djhtm?a=ACDD04-TT3"},
    {"基金":"第一金電子基金","相關持股%":5.60,"實際持股":"旺矽 5.60%","持股期別":"2026/07","持股／績效連結":"https://newfund.tw.dbs.com/mobile/a4.aspx?a=ACNC16"},
    {"基金":"玉山中小型股基金","相關持股%":4.56,"實際持股":"旺矽 2.04%、華星光 1.30%、聯亞 1.22%","持股期別":"2026/06","持股／績效連結":"https://sunnybank.moneydj.com/w/wr/wr04_ACML09-5810.djhtm"},
]


def num(value, default=np.nan):
    try:
        text = re.sub(r"[^0-9+\-.]", "", str(value))
        return float(text) if text not in {"", "+", "-", "."} else default
    except (TypeError, ValueError):
        return default


def twse_index_prices() -> pd.DataFrame:
    """Official TAIEX fallback when Yahoo throttles Streamlit Cloud."""
    rows = []
    month_start = pd.Timestamp.now().normalize().replace(day=1)
    for offset in range(15):
        month = month_start - pd.DateOffset(months=offset)
        try:
            response = requests.get(
                "https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK",
                params={"date": month.strftime("%Y%m01"), "response": "json"},
                headers={"User-Agent": "Mozilla/5.0 (market-dashboard/1.0)"},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            fields = payload.get("fields", [])
            for values in payload.get("data", []):
                record = dict(zip(fields, values))
                roc_year, roc_month, roc_day = map(int, record["日期"].split("/"))
                close = num(record.get("發行量加權股價指數"))
                if pd.notna(close):
                    rows.append({
                        "Date": datetime(roc_year + 1911, roc_month, roc_day),
                        "Open": close, "High": close, "Low": close, "Close": close,
                        "Volume": num(record.get("成交股數"), 0),
                    })
        except (KeyError, TypeError, ValueError, requests.RequestException):
            continue
    return pd.DataFrame(rows)


def regional_index_prices(symbol: str) -> pd.DataFrame:
    """Non-Yahoo fallbacks for the Asian benchmark indexes."""
    try:
        if symbol == "^N225":
            response = requests.get(
                "https://indexes.nikkei.co.jp/nkave/historical/nikkei_stock_average_daily_en.csv",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=20,
            )
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text)).rename(columns={"Date of Data": "Date"})
            df["Volume"] = 0
            return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        if symbol == "^KS11":
            end = pd.Timestamp.now().strftime("%Y%m%d")
            start = (pd.Timestamp.now() - pd.DateOffset(months=15)).strftime("%Y%m%d")
            response = requests.get(
                "https://api.finance.naver.com/siseJson.naver",
                params={"symbol": "KOSPI", "requestType": 1, "startTime": start, "endTime": end, "timeframe": "day"},
                headers={"User-Agent": "Mozilla/5.0"}, timeout=20,
            )
            response.raise_for_status()
            matches = re.findall(r'\["(\d{8})",\s*([-0-9.]+),\s*([-0-9.]+),\s*([-0-9.]+),\s*([-0-9.]+),\s*([-0-9.]+)', response.text)
            return pd.DataFrame(matches, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        if symbol in {"000001.SS", "^HSI"}:
            secid = "1.000001" if symbol == "000001.SS" else "100.HSI"
            response = requests.get(
                "https://push2his.eastmoney.com/api/qt/stock/kline/get",
                params={"secid": secid, "klt": 101, "fqt": 1, "beg": (pd.Timestamp.now()-pd.DateOffset(months=15)).strftime("%Y%m%d"),
                        "end": "20500101", "lmt": 400, "ut": "7eea3edcaed734bea9cbfc24409ed989",
                        "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"},
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}, timeout=25,
            )
            response.raise_for_status()
            klines = response.json()["data"]["klines"]
            rows = [line.split(",")[:6] for line in klines]
            return pd.DataFrame(rows, columns=["Date", "Open", "Close", "High", "Low", "Volume"])
    except (KeyError, TypeError, ValueError, requests.RequestException):
        pass
    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def prices(symbol: str, period="2y") -> pd.DataFrame:
    # Prefer the crumb-free chart endpoint. yfinance's cookie handshake is
    # frequently rate-limited on shared Streamlit Cloud IP addresses.
    df = pd.DataFrame()
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            encoded = requests.utils.quote(symbol, safe="")
            response = requests.get(
                f"https://{host}/v8/finance/chart/{encoded}",
                params={"range": period, "interval": "1d", "events": "history"},
                headers={"User-Agent": "Mozilla/5.0 (market-dashboard/1.0)"},
                timeout=20,
            )
            response.raise_for_status()
            result = response.json()["chart"]["result"][0]
            quote = result["indicators"]["quote"][0]
            df = pd.DataFrame({"Date": pd.to_datetime(result["timestamp"], unit="s", utc=True), **{
                name: quote.get(name.lower(), []) for name in ("Open", "High", "Low", "Close", "Volume")
            }})
            if not df.empty:
                break
        except (KeyError, IndexError, TypeError, ValueError, requests.RequestException):
            df = pd.DataFrame()
    # Yahoo can return a non-empty but one-session-stale TAIEX series.  Always
    # merge the official TWSE close into Taiwan data instead of treating TWSE
    # merely as an all-or-nothing fallback.  Official rows are appended last,
    # so they win when both sources contain the same trading date.
    if symbol == "^TWII":
        official = twse_index_prices()
        if not official.empty:
            if df.empty:
                df = official
            else:
                df = df.copy()
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
                official = official.copy()
                official["Date"] = pd.to_datetime(official["Date"], errors="coerce").dt.normalize()
                df = pd.concat([df, official], ignore_index=True)
                df = df.drop_duplicates(subset=["Date"], keep="last")
    elif df.empty:
        df = regional_index_prices(symbol)
    if df.empty:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"Datetime": "Date"})
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not set(cols).issubset(df.columns):
        return pd.DataFrame()
    df = df[cols].dropna(subset=["Close"]).copy()
    for column in ("Open", "High", "Low", "Close", "Volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    return df.dropna(subset=["Date", "Close"]).sort_values("Date")


@st.cache_data(ttl=21600, show_spinner=False)
def fred_series(series_id: str, value_name: str) -> pd.DataFrame:
    """Read a public FRED CSV and normalize it to Date/value columns."""
    url=f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response=requests.get(url,timeout=30,headers={"User-Agent":"market-research-dashboard/1.0"})
    response.raise_for_status()
    frame=pd.read_csv(StringIO(response.text))
    if len(frame.columns)<2: return pd.DataFrame()
    frame=frame.rename(columns={frame.columns[0]:"Date",frame.columns[1]:value_name})
    frame["Date"]=pd.to_datetime(frame["Date"],errors="coerce")
    frame[value_name]=pd.to_numeric(frame[value_name],errors="coerce")
    return frame.dropna().sort_values("Date")


@st.cache_data(ttl=21600, show_spinner=False)
def oil_inventory() -> pd.DataFrame:
    """EIA weekly U.S. commercial crude stocks excluding the SPR."""
    url="https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls"
    response=requests.get(url,timeout=35,headers={"User-Agent":"Mozilla/5.0"})
    response.raise_for_status()
    raw=pd.read_excel(BytesIO(response.content),sheet_name="Data 1",skiprows=2)
    raw=raw.iloc[:,:2].copy(); raw.columns=["Date","庫存量"]
    raw["Date"]=pd.to_datetime(raw["Date"],errors="coerce")
    raw["庫存量"]=pd.to_numeric(raw["庫存量"],errors="coerce")/1000
    return raw.dropna().sort_values("Date")


@st.cache_data(ttl=21600, show_spinner=False)
def wti_spot_price() -> pd.DataFrame:
    """EIA daily Cushing WTI spot price in dollars per barrel."""
    url="https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls"
    response=requests.get(url,timeout=35,headers={"User-Agent":"Mozilla/5.0"})
    response.raise_for_status()
    raw=pd.read_excel(BytesIO(response.content),sheet_name="Data 1",skiprows=2)
    raw=raw.iloc[:,:2].copy(); raw.columns=["Date","現貨"]
    raw["Date"]=pd.to_datetime(raw["Date"],errors="coerce")
    raw["現貨"]=pd.to_numeric(raw["現貨"],errors="coerce")
    return raw.dropna().sort_values("Date")


@st.cache_data(ttl=1800, show_spinner=False)
def gold_spot_proxy() -> tuple[pd.DataFrame, str]:
    """Scale GLD history to the latest live XAU/USD spot price for a transparent proxy."""
    response=requests.get("https://api.goldprice.dev/v1/prices?symbol=XAU-USD-SPOT",timeout=25)
    response.raise_for_status()
    row=response.json()["symbols"][0]
    live_spot=float(row["price"])
    gld=prices("GLD")
    if gld.empty: return pd.DataFrame(),"GLD 行情暫時無法取得"
    scale=live_spot/float(gld["Close"].iloc[-1])
    proxy=gld[["Date","Close"]].rename(columns={"Close":"現貨"}).copy()
    proxy["現貨"]=proxy["現貨"]*scale
    timestamp=str(row.get("computed_at",gld["Date"].iloc[-1]))
    return proxy,f"XAU/USD 即時價 {live_spot:,.2f} 美元／盎司校準於 {timestamp[:10]}"


@st.cache_data(ttl=21600, show_spinner=False)
def gold_inventory_snapshot() -> tuple[pd.DataFrame, str]:
    """Fetch CME's current COMEX depository report; never substitute ETF holdings."""
    url="https://www.cmegroup.com/delivery_reports/Gold_Stocks.xls"
    try:
        response=requests.get(url,timeout=30,headers={"User-Agent":"Mozilla/5.0","Referer":"https://www.cmegroup.com/"})
        response.raise_for_status()
        raw=pd.read_excel(BytesIO(response.content),header=None)
        text=raw.fillna("").astype(str)
        result=[]
        for category in ("REGISTERED","ELIGIBLE","PLEDGED"):
            hits=np.where(text.apply(lambda col: col.str.upper().eq(category)).to_numpy())
            if len(hits[0]):
                row=int(hits[0][0]); values=pd.to_numeric(raw.iloc[row],errors="coerce").dropna()
                if not values.empty: result.append({"分類":category.title(),"庫存量":float(values.iloc[-1])})
        if result: return pd.DataFrame(result),"CME COMEX Gold Stocks（當日快照，金衡盎司）"
        return pd.DataFrame(),"CME 檔案格式暫時無法辨識"
    except Exception as cme_exc:
        try:
            headers={"User-Agent":"Mozilla/5.0","X-Requested-With":"XMLHttpRequest"}
            token_response=requests.get("https://metalcharts.org/api/security/token",headers=headers,timeout=20)
            token_response.raise_for_status(); token=token_response.json()["token"]
            response=requests.get(
                "https://metalcharts.org/api/comex/inventory?symbol=XAU&range=1Y",
                headers={**headers,"X-MC-Token":token},timeout=25
            )
            response.raise_for_status(); payload=response.json()
            raw=pd.DataFrame(payload.get("data",[]))
            if raw.empty: raise ValueError("empty inventory mirror")
            raw=raw.rename(columns={"date":"Date","registered":"Registered","eligible":"Eligible","total":"Total"})
            raw["Date"]=pd.to_datetime(raw["Date"],errors="coerce")
            for column in ("Registered","Eligible","Total"): raw[column]=pd.to_numeric(raw[column],errors="coerce")/1_000_000
            raw=raw.dropna(subset=["Date","Total"]).sort_values("Date")
            note=f"CME 原始檔遭雲端 IP 阻擋（{type(cme_exc).__name__}）；改用 MetalCharts 每日同步 CME 報告的備援資料（百萬金衡盎司）"
            return raw,note
        except Exception as mirror_exc:
            return pd.DataFrame(),f"CME 官方檔與備援資料目前皆無法讀取（{type(cme_exc).__name__}／{type(mirror_exc).__name__}）"


def futures_spot_spread(futures: pd.DataFrame, spot: pd.DataFrame, spot_col: str) -> pd.DataFrame:
    left=futures[["Date","Close"]].rename(columns={"Close":"期貨價"}).sort_values("Date")
    right=spot[["Date",spot_col]].rename(columns={spot_col:"現貨價"}).sort_values("Date")
    left["Date"]=pd.to_datetime(left["Date"]).astype("datetime64[ns]")
    right["Date"]=pd.to_datetime(right["Date"]).astype("datetime64[ns]")
    merged=pd.merge_asof(left,right,on="Date",direction="backward",tolerance=pd.Timedelta(days=4)).dropna()
    merged["價差"]=merged["期貨價"]-merged["現貨價"]
    merged["價差率%"]=merged["價差"]/merged["現貨價"]*100
    return merged


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); close = out["Close"]
    for n in (20, 60, 200): out[f"MA{n}"] = close.rolling(n).mean()
    delta = close.diff(); gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    out["RSI14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean(); out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]
    low9=out["Low"].rolling(9).min(); high9=out["High"].rolling(9).max(); rsv=(close-low9)/(high9-low9).replace(0,np.nan)*100
    out["K"] = rsv.ewm(alpha=1/3,adjust=False).mean(); out["D"] = out["K"].ewm(alpha=1/3,adjust=False).mean()
    prev = close.shift(1); out["ATR14"] = pd.concat([out["High"]-out["Low"], (out["High"]-prev).abs(), (out["Low"]-prev).abs()], axis=1).max(axis=1).rolling(14).mean()
    out["VOL_MA20"] = out["Volume"].rolling(20).mean()
    return out


def technical_diagnosis(df: pd.DataFrame) -> dict:
    last=df.iloc[-1]; prev=df.iloc[-2] if len(df)>1 else last
    day=(last.Close/prev.Close-1)*100 if prev.Close else 0; m5=(last.Close/df.iloc[-6].Close-1)*100 if len(df)>5 else day
    vol_ratio=last.Volume/last.VOL_MA20 if pd.notna(last.VOL_MA20) and last.VOL_MA20 else np.nan
    high_volume=pd.notna(vol_ratio) and vol_ratio>=1.5; low_volume=pd.notna(vol_ratio) and vol_ratio<=.8
    if day>=0 and high_volume: pv="🔵 價漲爆量"
    elif day>=0 and low_volume: pv="🟢 價漲量縮"
    elif day<0 and high_volume: pv="🔴 價跌爆量"
    elif day<0 and low_volume: pv="🟡 價跌量縮"
    else: pv="⚪ 價量中性"
    kd_cross="黃金交叉" if last.K>last.D and prev.K<=prev.D else "死亡交叉" if last.K<last.D and prev.K>=prev.D else "多方排列" if last.K>=last.D else "空方排列"
    kd_zone="高檔鈍化" if last.K>=80 else "低檔鈍化" if last.K<=20 else "中性區"
    macd="多方擴張" if last.MACD_HIST>0 and last.MACD_HIST>prev.MACD_HIST else "多方收斂" if last.MACD_HIST>0 else "空方擴張" if last.MACD_HIST<prev.MACD_HIST else "空方收斂"
    recent=df.tail(20); range_pct=(recent.High.max()-recent.Low.min())/last.Close*100 if last.Close else np.nan
    above_ma=last.Close>last.MA20 if pd.notna(last.MA20) else False
    if pd.notna(range_pct) and range_pct<=7 and abs(m5)<=2 and (pd.isna(vol_ratio) or vol_ratio<=1.15): phase="↔️ 橫盤整理"
    elif m5>=3 and above_ma and pd.notna(vol_ratio) and vol_ratio>=1.2: phase="🚀 拉升"
    elif day<0 and high_volume and (last.K>=70 or (pd.notna(last.MA20) and last.Close<last.MA20)): phase="📤 出貨警訊"
    elif day<0 and high_volume and above_ma and m5>-5: phase="🧹 洗盤觀察"
    elif day<0 and low_volume and above_ma: phase="🧹 洗盤／健康回檔"
    else: phase="📈 多頭趨勢" if above_ma else "📉 空頭整理"
    returns=df["Close"].pct_change(); signed_turnover=(returns*df["Volume"]).tail(20)
    turnover_base=df["Volume"].tail(20).sum() or np.nan
    turnover_flow=signed_turnover.sum()/turnover_base*100 if pd.notna(turnover_base) else np.nan
    m20=(last.Close/df.iloc[-21].Close-1)*100 if len(df)>20 else m5
    window60=df.tail(60); low60=window60.Low.min(); high60=window60.High.max()
    price_position=(last.Close-low60)/(high60-low60) if high60>low60 else .5
    ma20_slope=(last.MA20/df.iloc[-6].MA20-1)*100 if len(df)>5 and pd.notna(last.MA20) and pd.notna(df.iloc[-6].MA20) else 0
    down_volume=df.loc[returns<0,"Volume"].tail(10).mean(); up_volume=df.loc[returns>=0,"Volume"].tail(10).mean()
    down_up_ratio=down_volume/up_volume if pd.notna(down_volume) and pd.notna(up_volume) and up_volume else 1.0
    if price_position>=.72 and ((pd.notna(turnover_flow) and turnover_flow<-.02) or (day<0 and high_volume) or down_up_ratio>=1.25):
        chip_phase="📤 出貨"
        chip_reason=f"位於60日區間{price_position:.0%}位置，跌日／漲日量比{down_up_ratio:.2f}倍，量價資金方向{turnover_flow:+.3f}%"
    elif pd.notna(last.MA60) and last.Close>=last.MA60 and -8<=m5<0 and ((pd.notna(vol_ratio) and vol_ratio>=1.15) or down_up_ratio>=1.15):
        chip_phase="🧹 洗盤"
        chip_reason=f"價格仍守60日均線，5日回檔{m5:.1f}%，換手{vol_ratio:.2f}倍"
    elif above_ma and m5>=2.5 and ma20_slope>0 and pd.notna(turnover_flow) and turnover_flow>0:
        chip_phase="🚀 拉貨"
        chip_reason=f"站上20日均線且5日上漲{m5:.1f}%，20日均線斜率{ma20_slope:+.1f}%，資金方向為正"
    elif pd.notna(range_pct) and range_pct<=12 and -4<=m20<=8 and pd.notna(turnover_flow) and turnover_flow>=0 and price_position<=.72:
        chip_phase="🟢 吸籌"
        chip_reason=f"20日漲跌{m20:+.1f}%、區間振幅{range_pct:.1f}%，價格未過度追高且量價資金方向為正"
    else:
        chip_phase="⚪ 籌碼結構穩定"
        chip_reason=f"換手{vol_ratio:.2f}倍、20日漲跌{m20:+.1f}%；四種典型訊號尚未同時成立"
    return {"價量判讀":pv,"KD判讀":f"{kd_cross}／{kd_zone}","MACD判讀":macd,"階段判讀":phase,"籌碼判讀":chip_phase,"籌碼理由":chip_reason,"換手強度":vol_ratio,"換手資金方向%":turnover_flow,"20日振幅%":range_pct}


def ret(df, periods):
    return (df["Close"].iloc[-1] / df["Close"].iloc[-periods-1] - 1) * 100 if len(df) > periods else np.nan


def analyze(symbol: str) -> dict:
    raw = prices(symbol)
    if raw.empty: return {"error": "暫無行情", "symbol": symbol}
    df = indicators(raw); last = df.iloc[-1]; recent = df.tail(60)
    flow = (df["Close"].pct_change() * df["Volume"]).tail(20).sum() / (df["Volume"].tail(20).sum() or np.nan) * 100
    score = 50 + (7 if last.Close > last.MA20 else -7) + (9 if last.Close > last.MA60 else -9) + (12 if last.Close > last.MA200 else -12)
    score += 7 if last.MACD_HIST > 0 else -7; score += 5 if 45 <= last.RSI14 <= 70 else -5 if last.RSI14 >= 78 else 0
    score += np.clip(ret(df, 21), -10, 10) + np.clip(flow * 10, -8, 8)
    diagnosis=technical_diagnosis(df)
    volume_ratio=last.Volume/last.VOL_MA20 if pd.notna(last.VOL_MA20) and last.VOL_MA20 else np.nan
    return {"symbol": symbol, "df": df, "date": last.Date.date().isoformat(), "close": last.Close, "day": ret(df, 1), "m1": ret(df, 21), "m3": ret(df, 63),
            "volume_ratio": volume_ratio, "rsi": last.RSI14, "atr": last.ATR14, "support": recent.Low.quantile(.1), "resistance": recent.High.quantile(.9),
            "flow": flow, "technical": float(np.clip(score, 0, 100)),"k":last.K,"d":last.D,"macd_hist":last.MACD_HIST,**diagnosis}


def _vol_result(series: pd.Series, name: str, source: str, proxy: bool=False) -> dict:
    series=pd.to_numeric(series,errors="coerce").dropna()
    if series.empty: return {"error":"無波動率資料","name":name,"source":source,"proxy":proxy}
    m1=(series.iloc[-1]/series.iloc[-22]-1)*100 if len(series)>=22 else (series.iloc[-1]/series.iloc[0]-1)*100 if len(series)>1 else np.nan
    return {"close":float(series.iloc[-1]),"m1":float(m1),"name":name,"source":source,"proxy":proxy}


@st.cache_data(ttl=1800,show_spinner=False)
def official_market_volatility() -> dict:
    result={}
    # 台灣：期交所最近三個月份的每日收盤檔。
    try:
        frames=[]; today=date.today()
        for offset in range(3):
            month=(today.replace(day=1)-pd.DateOffset(months=offset)).strftime("%Y%m")
            raw=requests.get(f"https://www.taifex.com.tw/file/taifex/Dailydownload/vix/log2data/{month}new.txt",timeout=15).content.decode("big5",errors="ignore")
            for line in raw.splitlines():
                parts=[p.strip() for p in line.split("\t") if p.strip()]
                if parts and re.fullmatch(r"\d{8}",parts[0]) and len(parts)>=3: frames.append((parts[0],num(parts[-1])))
        tw=pd.Series({d:v for d,v in frames}).sort_index(); result["台灣"]=_vol_result(tw,"TAIWAN VIX","TAIFEX官方日收盤")
    except Exception as exc: result["台灣"]={"error":type(exc).__name__}
    # 日本：Nikkei官方每日CSV。
    try:
        url="https://indexes.nikkei.co.jp/nkave/historical/nikkei_stock_average_vi_daily_en.csv"
        jp=pd.read_csv(StringIO(requests.get(url,timeout=20).text))["Close"]
        result["日本"]=_vol_result(jp,"Nikkei 225 VI","Nikkei Indexes官方CSV")
    except Exception as exc: result["日本"]={"error":type(exc).__name__}
    return result


def realized_volatility_proxy(analysis: dict, name: str, source: str) -> dict:
    if "error" in analysis: return {"error":analysis["error"],"name":name,"source":source,"proxy":True}
    close=analysis["df"]["Close"]; rv=close.pct_change().rolling(20).std()*np.sqrt(252)*100
    return _vol_result(rv,name,source,True)


def factors_for_market(global_factors: dict, market_volatility: dict, market: str) -> dict:
    factors=dict(global_factors)
    local=market_volatility.get(market,{})
    if "error" not in local: factors["VIX恐慌指數"]=local
    return factors


@st.cache_data(ttl=21600, show_spinner=False)
def imf_data() -> tuple[pd.DataFrame, str]:
    country_path = "/".join(v["imf"] for v in MARKETS.values()); rows=[]; errors=[]
    for code, label in IMF_CODES.items():
        try:
            r=requests.get(f"https://www.imf.org/external/datamapper/api/v1/{code}/{country_path}", timeout=25, headers={"User-Agent":"asia-market-template/1.0"}); r.raise_for_status()
            values=r.json().get("values",{}).get(code,{})
        except Exception as exc: values={}; errors.append(f"{code}:{type(exc).__name__}")
        for market,cfg in MARKETS.items():
            series=values.get(cfg["imf"],{}); candidates=[(int(y),v) for y,v in series.items() if str(y).isdigit() and int(y)<=date.today().year+1 and v is not None]
            if candidates:
                year,value=max(candidates); rows.append({"市場":market,"指標":label,"數值":float(value),"年度":year,"資料來源":"IMF","來源代碼":code})
    return pd.DataFrame(rows), "；".join(errors)


@st.cache_data(ttl=21600, show_spinner=False)
def stockq_data() -> tuple[pd.DataFrame, str]:
    """Public-table fallback for GDP/CPI when IMF DataMapper is unavailable."""
    url="https://www.stockq.org/economy/worldstats.php"
    country_map={"台灣":"台灣","日本":"日本","韓國":"韓國","中國":"中國","香港":"香港"}
    try:
        response=requests.get(url,timeout=20,headers={"User-Agent":"Mozilla/5.0 (compatible; market-research-dashboard/1.0)"}); response.raise_for_status()
        html=response.content.decode("utf-8",errors="replace")
        tables=pd.read_html(StringIO(html))
        table=next(t for t in tables if {"國家","GDP 成長率","通貨 膨脹率"}.issubset(t.columns) and len(t)>=5)
        updated=re.search(r"更新日期[：:]?\s*([^<\n]+)",html)
        update_text=updated.group(1).strip()[:30] if updated else date.today().isoformat()
        rows=[]
        for market,country in country_map.items():
            match=table[table["國家"].astype(str).str.strip().eq(country)]
            if match.empty: continue
            row=match.iloc[0]
            for column,label in (("GDP 成長率","實質GDP成長%"),("通貨 膨脹率","CPI年增率%")):
                value=pd.to_numeric(row.get(column),errors="coerce")
                if pd.notna(value): rows.append({"市場":market,"指標":label,"數值":float(value),"年度":update_text,"資料來源":"StockQ備援","來源代碼":column})
        return pd.DataFrame(rows),""
    except Exception as exc:
        return pd.DataFrame(),f"StockQ:{type(exc).__name__}"


@st.cache_data(ttl=21600, show_spinner=False)
def macromicro_export_data() -> tuple[pd.DataFrame, str]:
    """Latest public MacroMicro export YoY observations verified from its cross-country table."""
    # The public page currently rejects non-browser requests (HTTP 403). These values are the
    # latest visible observations verified on 2026-09-01; IMF remains the automatic fallback.
    observations={
        "台灣":(32.9,"2026-07"),
        "日本":(23.2,"2026-07"),
        "韓國":(63.0,"2026-07"),
        "中國":(23.9,"2026-07"),
        "香港":(23.9,"2026-07"),
        "美國":(12.0,"2026-07"),
        "英國":(6.1,"2026-07"),
        "法國":(6.8,"2026-07"),
        "德國":(6.6,"2026-07"),
        "印度":(18.7,"2026-07"),
        "印尼":(8.8,"2026-07"),
        "澳洲":(6.6,"2026-07"),
        "巴西":(6.2,"2026-07"),
    }
    rows=[{"市場":market,"指標":"出口值年增率%","數值":value,"年度":period,"資料來源":"財經M平方（中國出口代理）" if market=="香港" else "財經M平方（公開頁面）","來源代碼":"exports-yoy-cn-proxy" if market=="香港" else "exports-yoy"} for market,(value,period) in observations.items()]
    return pd.DataFrame(rows),"公開頁面驗證日：2026-09-01；香港依指定採中國出口值年增率作為代理"


def combine_macro_sources(imf: pd.DataFrame, stockq: pd.DataFrame, macromicro: pd.DataFrame | None=None) -> pd.DataFrame:
    """IMF has priority; StockQ only fills missing market/indicator pairs."""
    frames=[]
    if imf is not None and not imf.empty: frames.append(imf.copy())
    if stockq is not None and not stockq.empty:
        fallback=stockq.copy()
        if frames:
            keys=set(zip(frames[0]["市場"],frames[0]["指標"]))
            fallback=fallback[[ (m,i) not in keys for m,i in zip(fallback["市場"],fallback["指標"]) ]]
        if not fallback.empty: frames.append(fallback)
    if macromicro is not None and not macromicro.empty: frames.append(macromicro.copy())
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=["市場","指標","數值","年度","資料來源","來源代碼"])


@st.cache_data(ttl=1800, show_spinner=False)
def twse_flow():
    for offset in range(10):
        d=date.today()-timedelta(days=offset); url="https://www.twse.com.tw/rwd/zh/fund/BFI82U?"+urlencode({"date":d.strftime("%Y%m%d"),"response":"json"})
        try:
            p=requests.get(url,timeout=15,headers={"User-Agent":"Mozilla/5.0"}).json(); frame=pd.DataFrame(p.get("data",[]),columns=p.get("fields",[]))
            if frame.empty: continue
            identity=next(c for c in frame if "單位名稱" in c); net=next(c for c in frame if "買賣差額" in c)
            return pd.DataFrame({"法人":frame[identity],"買賣超億元":frame[net].map(num)/1e8}),d.isoformat()
        except Exception: continue
    return pd.DataFrame(),"官方資料暫無回應"


@st.cache_data(ttl=1800, show_spinner=False)
def taifex_positions():
    url="https://openapi.taifex.com.tw/v1/MarketDataOfMajorInstitutionalTradersDetailsOfFuturesContractsBytheDate"
    try:
        frame=pd.DataFrame(requests.get(url,timeout=20,headers={"User-Agent":"asia-market-template/1.0"}).json())
        product=next((c for c in frame if "商品" in c or c.lower() in {"commodity","product","contractcode"}),None)
        identity=next((c for c in frame if "身份" in c or "身分" in c or c.lower() in {"identity","item"}),None)
        oi=next((c for c in frame if "多空未平倉口數淨額" in c or c.lower().replace(" ","") in {"openinterest(net)","openinterestnet"}),None)
        date_col=next((c for c in frame if "日期" in c or c.lower()=="date"),None)
        if not all((product,identity,oi)):
            return pd.DataFrame(),"期交所欄位格式已更新，暫時無法解析"
        tx=frame[frame[product].astype(str).str.contains("臺股期貨|台股期貨",regex=True,na=False)]
        return pd.DataFrame({"法人":tx[identity],"淨未平倉口數":tx[oi].map(num)}),str(tx[date_col].iloc[0]) if date_col and not tx.empty else "最新交易日"
    except Exception as exc: return pd.DataFrame(),f"期交所暫無回應：{type(exc).__name__}"


def macro_score(market, macro):
    if macro is None or macro.empty or not {"市場","指標","數值"}.issubset(macro.columns):
        return 50.0
    s=macro[macro["市場"].eq(market)]; v=dict(zip(s["指標"],s["數值"]));
    if not v: return 50.0
    score=50+np.clip(v.get("實質GDP成長%",0),-5,8)*3+np.clip(v.get("進口量成長%",0),-10,15)*.4-abs(v.get("CPI年增率%",2)-2)*2.5
    return float(np.clip(score,0,100))


def verdict(score): return "🟢 可分批布局" if score>=70 else "🟡 等回檔／小量觀察" if score>=58 else "⚪ 中性觀望" if score>=45 else "🔴 防守／暫緩"


def global_factor_adjustment(market: str, factors: dict, weights: dict | None=None) -> tuple[float, list[str]]:
    """Translate observable cross-asset moves into a small, capped market adjustment."""
    weights=weights or {}; weight=lambda name:float(weights.get(name,1.0))
    score=0.0; notes=[]
    dxy=factors.get("美元指數",{}).get("m1",np.nan)
    oil=factors.get("WTI原油",{}).get("m1",np.nan)
    us10=factors.get("美國10年債殖利率",{}).get("m1",np.nan)
    vix=factors.get("VIX恐慌指數",{}).get("m1",np.nan)
    gold=factors.get("黃金",{}).get("m1",np.nan)
    bitcoin=factors.get("比特幣",{}).get("m1",np.nan)
    if pd.notna(dxy):
        impact=np.clip(-dxy*.55*weight("美元指數"),-7,7); score+=impact
        notes.append(f"美元1M {dxy:+.1f}%（{impact:+.1f}分）")
    if pd.notna(oil):
        # 五個市場皆偏能源進口；日本、韓國對能源進口成本更敏感。
        sensitivity={"日本":.42,"韓國":.45,"台灣":.34,"中國":.25,"香港":.18}.get(market,.25)
        impact=np.clip(-oil*sensitivity*weight("WTI原油"),-7,7); score+=impact
        notes.append(f"油價1M {oil:+.1f}%（{impact:+.1f}分）")
    if pd.notna(us10):
        impact=np.clip(-us10*.22*weight("美國10年債殖利率"),-6,6); score+=impact
        notes.append(f"美債殖利率指數1M {us10:+.1f}%（{impact:+.1f}分）")
    if pd.notna(vix):
        impact=np.clip(-vix*.18*weight("VIX恐慌指數"),-6,6); score+=impact
        notes.append(f"VIX 1M {vix:+.1f}%（{impact:+.1f}分）")
    if pd.notna(gold):
        impact=np.clip(-gold*.10*weight("黃金"),-4,4); score+=impact
        notes.append(f"黃金1M {gold:+.1f}%（{impact:+.1f}分）")
    if pd.notna(bitcoin):
        impact=np.clip(bitcoin*.06*weight("比特幣"),-4,4); score+=impact
        notes.append(f"比特幣1M {bitcoin:+.1f}%（{impact:+.1f}分）")
    return float(np.clip(score,-12,12)),notes


def export_factor_score(market: str, macro: pd.DataFrame, weight: float=1.0) -> tuple[float, float]:
    if macro is None or macro.empty or not {"市場","指標","數值"}.issubset(macro.columns): return 0.0,np.nan
    market_rows=macro[macro["市場"].eq(market)]
    match=market_rows[market_rows["指標"].eq("出口值年增率%")]
    if match.empty: match=market_rows[market_rows["指標"].eq("出口量成長%")]
    if match.empty: return 0.0,np.nan
    growth=float(match.iloc[-1]["數值"]); sensitivity=EXPORT_SENSITIVITY.get(market,.7)
    return float(np.clip(growth*.80*sensitivity*weight,-12,12)),growth


def factor_breakdown(market: str, factors: dict, weights: dict, macro: pd.DataFrame, export_weight: float, rate_bps: float) -> pd.DataFrame:
    w=lambda name:float(weights.get(name,1.0)); value=lambda name:factors.get(name,{}).get("m1",np.nan)
    oil_sensitivity={"日本":.42,"韓國":.45,"台灣":.34,"中國":.25,"香港":.18}.get(market,.25)
    formulas={
        "美元指數":lambda x:np.clip(-x*.55*w("美元指數"),-7,7),
        "WTI原油":lambda x:np.clip(-x*oil_sensitivity*w("WTI原油"),-7,7),
        "黃金":lambda x:np.clip(-x*.10*w("黃金"),-4,4),
        "比特幣":lambda x:np.clip(x*.06*w("比特幣"),-4,4),
        "美國10年債殖利率":lambda x:np.clip(-x*.22*w("美國10年債殖利率"),-6,6),
        "VIX恐慌指數":lambda x:np.clip(-x*.18*w("VIX恐慌指數"),-6,6),
    }
    rows=[]
    for name,formula in formulas.items():
        move=value(name); rows.append({"市場":market,"因子":name,"1M變動%":move,"權重":w(name),"分數貢獻":float(formula(move)) if pd.notna(move) else 0.0})
    export_score,export_growth=export_factor_score(market,macro,export_weight); rows.append({"市場":market,"因子":"出口成長","1M變動%":export_growth,"權重":export_weight,"分數貢獻":export_score})
    semi_score,_=semiconductor_derating(market,factors,rate_bps,weights); rows.append({"市場":market,"因子":"半導體去估值","1M變動%":np.nan,"權重":1.0,"分數貢獻":semi_score})
    return pd.DataFrame(rows)


def risk_sentiment(factors: dict) -> tuple[float, str]:
    """0–100 risk-off gauge. Uses 1M moves and is intentionally transparent."""
    moves={name:factors.get(name,{}).get("m1",np.nan) for name in ("VIX恐慌指數","黃金","美元指數","比特幣")}
    parts=[]
    if pd.notna(moves["VIX恐慌指數"]): parts.append(np.clip(moves["VIX恐慌指數"]*1.1,-25,25))
    if pd.notna(moves["黃金"]): parts.append(np.clip(moves["黃金"]*.7,-12,12))
    if pd.notna(moves["美元指數"]): parts.append(np.clip(moves["美元指數"]*1.2,-10,10))
    if pd.notna(moves["比特幣"]): parts.append(np.clip(-moves["比特幣"]*.35,-15,15))
    gauge=float(np.clip(50+(sum(parts) if parts else 0),0,100))
    label="🔴 高度避險" if gauge>=70 else "🟠 偏避險" if gauge>=58 else "⚪ 中性" if gauge>=42 else "🟢 偏風險偏好"
    return gauge,label


def semiconductor_derating(market: str, factors: dict, rate_bps: float, weights: dict | None=None) -> tuple[float, str]:
    """以折現率與風險偏好估算半導體去估值壓力；負值代表扣分。"""
    exposure={"台灣":1.00,"韓國":.85,"日本":.55,"中國":.35,"香港":.15}.get(market,.25)
    def move(name):
        value=factors.get(name,{}).get("m1",0)
        return 0 if pd.isna(value) else value
    weights=weights or {}; w=lambda name:float(weights.get(name,1.0))
    raw=max(rate_bps,0)/45+max(move("美國10年債殖利率"),0)*.18*w("美國10年債殖利率")+max(move("美元指數"),0)*.28*w("美元指數")+max(move("VIX恐慌指數"),0)*.10*w("VIX恐慌指數")
    relief=max(-rate_bps,0)/130+max(-move("美國10年債殖利率"),0)*.08*w("美國10年債殖利率")
    adjustment=float(np.clip((relief-raw)*exposure,-10,3))
    label="高" if adjustment<=-6 else "中" if adjustment<=-3 else "低" if adjustment<0 else "緩解"
    return adjustment,label


def flow_table(etf_data: dict) -> pd.DataFrame:
    valid={m:s for m,s in etf_data.items() if "error" not in s and pd.notna(s.get("flow"))}
    denominator=sum(abs(s["flow"]) for s in valid.values()) or np.nan
    rows=[]
    for market,s in etf_data.items():
        value=s.get("flow",np.nan)
        if pd.isna(value): direction="資料不足"; strength=np.nan; signed=np.nan
        else:
            direction="🟢 流入" if value>0.01 else "🔴 流出" if value<-0.01 else "⚪ 中性"
            strength=abs(value)/denominator*100 if pd.notna(denominator) else 0
            signed=np.sign(value)*strength
        rows.append({"市場":market,"ETF":MARKETS[market]["etf"],"判讀":direction,"資金強度占比%":strength,"淨方向比例%":signed,"量價流向值%":value,"1M報酬%":s.get("m1"),"日期":s.get("date")})
    return pd.DataFrame(rows)


def line_chart(df, show_fibonacci: bool=False):
    long=df.tail(260).melt(id_vars="Date",value_vars=["Close","MA20","MA60","MA200"],var_name="線型",value_name="價格")
    lines=alt.Chart(long).mark_line().encode(x="Date:T",y=alt.Y("價格:Q",scale=alt.Scale(zero=False)),color="線型:N",tooltip=["Date:T","線型:N",alt.Tooltip("價格:Q",format=",.2f")])
    latest=df.tail(1).assign(線型="最新日線",價格=lambda x:x["Close"])
    point=alt.Chart(latest).mark_point(size=115,filled=True,color="#ff4b4b").encode(x="Date:T",y="價格:Q",tooltip=[alt.Tooltip("Date:T",title="資料日期"),alt.Tooltip("價格:Q",title="最新日線價",format=",.2f")])
    layers=[lines,point]
    fib_toggle=None
    if show_fibonacci:
        fib_toggle=alt.selection_point(name="fib_toggle",fields=["控制"],on="click",toggle=True,empty=False)
        window=df.tail(260); high=float(window["High"].max()); low=float(window["Low"].min()); span=high-low
        fib=pd.DataFrame([
            {"Date":window["Date"].max(),"比例":label,"價格":high-ratio*span,"控制":"費波那契"}
            for ratio,label in ((0.0,"0% 壓力"),(.382,"38.2%"),(.5,"50%"),(.618,"61.8%"),(1.0,"100% 支撐"))
        ])
        fib["右側標籤"]=fib.apply(lambda row:f"{row['比例']}｜{row['價格']:,.2f}",axis=1)
        rules=alt.Chart(fib).mark_rule(color="#f59e0b",strokeDash=[6,4]).encode(
            y="價格:Q",opacity=alt.condition(fib_toggle,alt.value(.9),alt.value(0)),tooltip=["比例:N",alt.Tooltip("價格:Q",format=",.2f")]
        )
        labels=alt.Chart(fib).mark_text(align="right",dx=-6,dy=-5,color="#f59e0b",fontSize=11,fontWeight="bold").encode(
            x="Date:T",y="價格:Q",text=alt.Text("右側標籤:N"),opacity=alt.condition(fib_toggle,alt.value(1),alt.value(0))
        )
        control=alt.Chart(pd.DataFrame({"控制":["費波那契"]})).mark_text(
            align="left",baseline="top",fontSize=14,fontWeight="bold",cursor="pointer"
        ).encode(
            x=alt.value(8),y=alt.value(8),
            text=alt.condition(fib_toggle,alt.value("☑ 顯示費波那契支撐壓力"),alt.value("☐ 顯示費波那契支撐壓力")),
            color=alt.condition(fib_toggle,alt.value("#f59e0b"),alt.value("#9ca3af")),
            tooltip=alt.value("點一下顯示或隱藏費波那契線")
        ).add_params(fib_toggle)
        layers.extend([rules,labels,control])
    chart=alt.layer(*layers).properties(height=380,title="價格與均線").interactive()
    return chart


def price_volume_chart(df):
    display=df.tail(260).copy()
    display["漲跌方向"]=np.where(display["Close"]>=display["Open"],"上漲","下跌")
    return alt.Chart(display).mark_bar(opacity=.75).encode(
        x=alt.X("Date:T",title=None),y=alt.Y("Volume:Q",title="成交量"),
        color=alt.Color("漲跌方向:N",scale=alt.Scale(domain=["上漲","下跌"],range=["#ef5350","#26a69a"]),legend=None),
        tooltip=[alt.Tooltip("Date:T",title="日期"),alt.Tooltip("Volume:Q",title="成交量",format=",")]
    ).properties(height=150,title="成交量").interactive()


def spread_chart(frame: pd.DataFrame, title: str):
    base=alt.Chart(frame.tail(520)).encode(x=alt.X("Date:T",title="日期"))
    area=base.mark_area(opacity=.28).encode(
        y=alt.Y("價差:Q",title="期貨－現貨"),
        color=alt.condition(alt.datum["價差"]>=0,alt.value("#ef5350"),alt.value("#26a69a")),
        tooltip=[alt.Tooltip("Date:T",title="日期"),alt.Tooltip("期貨價:Q",format=",.2f"),alt.Tooltip("現貨價:Q",format=",.2f"),alt.Tooltip("價差:Q",format="+.2f"),alt.Tooltip("價差率%:Q",format="+.2f")]
    )
    zero=alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(color="#888",strokeDash=[4,4]).encode(y="y:Q")
    return alt.layer(area,zero).properties(height=280,title=title).interactive()


def inventory_chart(frame: pd.DataFrame, title: str, categorical: bool=False):
    if categorical:
        return alt.Chart(frame).mark_bar().encode(
            x=alt.X("分類:N",title=None),y=alt.Y("庫存量:Q",title="金衡盎司"),
            color=alt.Color("分類:N",legend=None),tooltip=["分類:N",alt.Tooltip("庫存量:Q",format=",")]
        ).properties(height=280,title=title)
    return alt.Chart(frame.tail(260)).mark_line(color="#ff9f1c",strokeWidth=2).encode(
        x=alt.X("Date:T",title="日期"),y=alt.Y("庫存量:Q",title="百萬桶",scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Date:T",title="週別"),alt.Tooltip("庫存量:Q",title="百萬桶",format=",.1f")]
    ).properties(height=280,title=title).interactive()


def gold_inventory_history_chart(frame: pd.DataFrame):
    long=frame.tail(260).melt(id_vars="Date",value_vars=["Registered","Eligible","Total"],var_name="庫存分類",value_name="百萬盎司")
    return alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("Date:T",title="日期"),y=alt.Y("百萬盎司:Q",title="百萬金衡盎司",scale=alt.Scale(zero=False)),
        color=alt.Color("庫存分類:N",scale=alt.Scale(domain=["Registered","Eligible","Total"],range=["#f59e0b","#60a5fa","#ef4444"])),
        tooltip=[alt.Tooltip("Date:T",title="日期"),"庫存分類:N",alt.Tooltip("百萬盎司:Q",format=",.2f")]
    ).properties(height=280,title="COMEX 黃金庫存（Registered／Eligible／Total）").interactive()


def oscillator_charts(df):
    kd=df.tail(180).melt(id_vars="Date",value_vars=["K","D"],var_name="指標",value_name="數值")
    kd_chart=alt.Chart(kd).mark_line().encode(x="Date:T",y=alt.Y("數值:Q",scale=alt.Scale(domain=[0,100])),color="指標:N",tooltip=["Date:T","指標:N",alt.Tooltip("數值:Q",format=".1f")]).properties(height=190,title="KD（80以上偏熱、20以下偏冷）").interactive()
    macd=df.tail(180); macd_chart=alt.Chart(macd).mark_bar().encode(x="Date:T",y="MACD_HIST:Q",color=alt.condition(alt.datum.MACD_HIST>=0,alt.value("#2ca02c"),alt.value("#d62728")),tooltip=["Date:T",alt.Tooltip("MACD_HIST:Q",format=".3f")]).properties(height=190,title="MACD柱狀動能").interactive()
    return kd_chart,macd_chart


def turnover_chart(df):
    data=df.tail(60).copy(); data["換手率倍數"]=data["Volume"]/data["VOL_MA20"]
    return alt.Chart(data).mark_bar().encode(
        x=alt.X("Date:T",title="時間"),y=alt.Y("換手率倍數:Q",title="換手率（倍數）"),
        color=alt.condition(alt.datum["換手率倍數"]>=1.5,alt.value("#fdbb18"),alt.value("#49a6e9")),
        tooltip=["Date:T",alt.Tooltip("換手率倍數:Q",format=".2f")]
    ).properties(height=280,title="🔄 籌碼換手與隱藏吸籌檢測").interactive()


def render_metric_grid(items):
    """Render responsive cards without Streamlit metric text truncation."""
    cards=[]
    for label,value,delta in items:
        delta_html=f'<div class="metric-delta">{html.escape(str(delta))}</div>' if delta not in (None,"") else ""
        cards.append(
            '<div class="metric-card">'
            f'<div class="metric-label">{html.escape(str(label))}</div>'
            f'<div class="metric-value">{html.escape(str(value))}</div>'
            f'{delta_html}</div>'
        )
    st.markdown("""
    <style>
    .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:8px 0 20px}
    .metric-card{border:1px solid rgba(128,128,128,.28);border-radius:12px;padding:16px 18px;min-width:0;background:rgba(128,128,128,.04)}
    .metric-label{font-size:1rem;font-weight:650;opacity:.78;margin-bottom:8px;white-space:normal}
    .metric-value{font-size:clamp(1.45rem,2.4vw,2.35rem);font-weight:700;line-height:1.18;white-space:normal;overflow:visible;text-overflow:clip;overflow-wrap:anywhere}
    .metric-delta{display:inline-block;margin-top:9px;padding:3px 9px;border-radius:999px;font-size:.95rem;background:rgba(128,128,128,.16)}
    @media(max-width:700px){.metric-grid{grid-template-columns:repeat(auto-fit,minmax(155px,1fr))}.metric-value{font-size:1.4rem}}
    </style>
    <div class="metric-grid">"""+"".join(cards)+"</div>",unsafe_allow_html=True)


def commodity_scenario_score(name: str, technical: float, rate_bps: float, stock_pct: float, bond_pct: float) -> tuple[float, str]:
    """Combine technical strength with transparent macro-scenario sensitivities."""
    if name == "黃金期貨指數": adjustment=-rate_bps*.035-stock_pct*.35+bond_pct*.25
    elif name in {"費城金銀指數","彭博世界礦業指數"}: adjustment=-rate_bps*.025+stock_pct*.20+bond_pct*.12
    elif name == "DAXglobal 農業企業指數": adjustment=-rate_bps*.012+stock_pct*.35-bond_pct*.08
    elif name == "西德州原油期貨": adjustment=-rate_bps*.012+stock_pct*.80-bond_pct*.10
    else: adjustment=-rate_bps*.010+stock_pct*.65-bond_pct*.08
    score=float(np.clip(technical*.70+15+adjustment,0,100))
    explanation=f"技術分×70%＋中性基準15分＋情境調整{adjustment:+.1f}分"
    return score,explanation


def render_commodity_section(commodity_data: dict, scenario_name: str, rate: int, stock: int, bond: int) -> None:
    st.subheader(f"情境：{scenario_name}｜利率 {rate:+d}bps｜股票 {stock:+d}%｜債券 {bond:+d}%")
    st.caption("情境分會隨左側情境與自訂衝擊即時重算；原指數無穩定免費歷史行情時，以ETF代理並清楚標示。")
    commodity_rows=[]
    for name,cfg in COMMODITY_ASSETS.items():
        s=commodity_data[name]
        row={"分類":cfg["group"],"指標":name,"代碼":cfg["symbol"],"資料屬性":cfg["type"],"資料來源":cfg["source"]}
        if "error" in s:
            row.update({"日期":None,"目前值":np.nan,"日漲跌%":np.nan,"1M%":np.nan,"3M%":np.nan,"技術分":np.nan,"情境分":np.nan,"結論":"暫無行情","情境計算":""})
        else:
            scenario_score,scenario_note=commodity_scenario_score(name,s["technical"],rate,stock,bond)
            row.update({"日期":s["date"],"目前值":s["close"],"日漲跌%":s["day"],"1M%":s["m1"],"3M%":s["m3"],"技術分":s["technical"],"情境分":scenario_score,"結論":verdict(scenario_score),"情境計算":scenario_note})
        commodity_rows.append(row)
    commodity_frame=pd.DataFrame(commodity_rows).sort_values("情境分",ascending=False,na_position="last")
    available=commodity_frame.dropna(subset=["情境分"])
    if not available.empty: st.metric("目前相對優先商品",available.iloc[0]["指標"],available.iloc[0]["結論"])
    st.dataframe(commodity_frame,hide_index=True,width="stretch")
    selected_asset=st.selectbox("選擇指標查看技術線",list(COMMODITY_ASSETS),key=f"commodity_asset_{SCENARIO_VIEW}")
    selected_data=commodity_data[selected_asset]
    if "error" in selected_data:
        st.warning(f"{selected_asset}目前無法取得行情；請稍後清除快取重試。")
        return
    cfg=COMMODITY_ASSETS[selected_asset]
    scenario_score,scenario_note=commodity_scenario_score(selected_asset,selected_data["technical"],rate,stock,bond)
    st.markdown(f"### {selected_asset}｜{cfg['symbol']}｜資料日 {selected_data['date']}")
    render_metric_grid([
        ("目前值",f"{selected_data['close']:,.2f}",f"{selected_data['day']:+.2f}%"),
        ("1個月",f"{selected_data['m1']:+.2f}%",None),("3個月",f"{selected_data['m3']:+.2f}%",None),
        ("情境分",f"{scenario_score:.1f}",verdict(scenario_score)),("技術階段",selected_data["階段判讀"],None),
    ])
    st.altair_chart(line_chart(selected_data["df"],show_fibonacci=True),width="stretch")
    st.altair_chart(price_volume_chart(selected_data["df"]),width="stretch")
    st.caption(f"{scenario_note}。資料屬性：{cfg['type']}。紅柱為收高、綠柱為收低；指數若未提供成交量會顯示空值。DAX『農金』依 DAXglobal Agribusiness（農業企業）解讀。")

    st.markdown("### 期現貨價差")
    spread_choice=st.radio("商品",["WTI 原油","黃金"],horizontal=True,key=f"spread_choice_{SCENARIO_VIEW}")
    if spread_choice=="WTI 原油":
        try:
            spot=wti_spot_price()
            spread=futures_spot_spread(commodity_data["西德州原油期貨"]["df"],spot,"現貨")
            if spread.empty: st.info("WTI 期貨與現貨日期目前無法對齊。")
            else:
                st.altair_chart(spread_chart(spread,"WTI 近月期貨－Cushing 現貨價差（美元／桶）"),width="stretch")
                latest=spread.iloc[-1]; st.caption(f"最新價差 {latest['價差']:+.2f} 美元／桶（{latest['價差率%']:+.2f}%）。正值通常為期貨溢價，負值通常為現貨溢價；近月連續合約換月時可能出現跳動。資料：Yahoo Finance、EIA。")
        except Exception as exc: st.info(f"WTI 期現貨價差暫時無法更新（{type(exc).__name__}）。")
    else:
        try:
            spot,spot_note=gold_spot_proxy()
            spread=futures_spot_spread(commodity_data["黃金期貨指數"]["df"],spot,"現貨")
            if spread.empty: st.info("黃金期貨與現貨代理日期目前無法對齊。")
            else:
                st.altair_chart(spread_chart(spread,"黃金近月期貨－XAU/USD 現貨代理價差（美元／盎司）"),width="stretch")
                latest=spread.iloc[-1]
                st.caption(f"最新代理價差 {latest['價差']:+.2f} 美元／盎司（{latest['價差率%']:+.2f}%）。{spot_note}；歷史現貨以 GLD 報酬比例換算，適合觀察方向，不等同正式 LBMA 歷史定盤價。資料：Yahoo Finance、goldprice.dev。")
        except Exception as exc: st.info(f"黃金期現貨代理價差暫時無法更新（{type(exc).__name__}）。")

    st.markdown("### 庫存量")
    inv_gold,inv_note=gold_inventory_snapshot()
    inv_col1,inv_col2=st.columns(2)
    with inv_col1:
        try:
            oil_inv=oil_inventory()
            st.altair_chart(inventory_chart(oil_inv,"美國商業原油庫存（不含 SPR）"),width="stretch")
            if not oil_inv.empty:
                last=oil_inv.iloc[-1]; change=last["庫存量"]-oil_inv.iloc[-2]["庫存量"] if len(oil_inv)>1 else np.nan
                st.caption(f"{last['Date']:%Y-%m-%d}：{last['庫存量']:,.1f} 百萬桶，週變動 {change:+,.1f} 百萬桶。資料：EIA。")
        except Exception as exc: st.info(f"EIA 原油庫存暫時無法更新（{type(exc).__name__}）。")
    with inv_col2:
        if inv_gold.empty:
            st.info(inv_note)
            st.link_button("查看 CME 金屬庫存官方報告","https://www.cmegroup.com/solutions/clearing/operations-and-deliveries/nymex-delivery-notices.html")
        elif "Date" in inv_gold.columns:
            st.altair_chart(gold_inventory_history_chart(inv_gold),width="stretch")
            latest=inv_gold.iloc[-1]
            st.caption(f"{latest['Date']:%Y-%m-%d}：Registered {latest['Registered']:,.2f}、Eligible {latest['Eligible']:,.2f}、Total {latest['Total']:,.2f} 百萬盎司。{inv_note}。")
            st.link_button("下載 CME Gold Stocks 官方檔","https://www.cmegroup.com/delivery_reports/Gold_Stocks.xls")
        else:
            st.altair_chart(inventory_chart(inv_gold,"COMEX 黃金庫存結構（最新快照）",categorical=True),width="stretch")
            st.caption(inv_note)


if SCENARIO_VIEW == "commodities":
    st.title("🪙 黃金／石油｜情境模擬與進場評估")
    st.caption("黃金、礦業、農業企業與石油｜價量技術 × 利率 × 全球股債情境。")
else:
    st.title("🌏 全球主要市場｜情境模擬與市場進場評估")
    st.caption("台、美、日、韓、中、港、英、法、德、印度、印尼、澳洲、巴西｜價量技術 × ETF資金流代理 × IMF總經。")
with st.sidebar:
    scenario_name=st.selectbox("總體情境",list(SCENARIOS)); preset=SCENARIOS[scenario_name]; custom=st.checkbox("自行調整衝擊假設")
    rate=st.slider("政策利率變動（bps）",-300,300,preset["rate"],25,disabled=not custom); stock=st.slider("全球股票衝擊（%）",-40,30,preset["stock"],1,disabled=not custom); bond=st.slider("全球債券衝擊（%）",-25,25,preset["bond"],1,disabled=not custom)
    with st.expander("⚙️ 動態因子權重",expanded=False):
        st.caption("0＝不納入，1＝標準權重，2＝加倍影響；調整後立即重算。")
        factor_weights={name:st.slider(f"{name}權重",0.0,2.0,.4 if name=="VIX恐慌指數" else 1.0,.1,key=f"weight_{cfg['symbol']}") for name,cfg in GLOBAL_FACTORS.items()}
        export_weight=st.slider("出口成長權重",0.0,2.0,1.0,.1)
    if st.button("🔄 清除快取並更新",width="stretch"): st.cache_data.clear(); st.rerun()

if SCENARIO_VIEW == "commodities":
    with st.spinner("同步黃金與石油行情……"):
        commodity_data={name:analyze(cfg["symbol"]) for name,cfg in COMMODITY_ASSETS.items()}
    render_commodity_section(commodity_data,scenario_name,rate,stock,bond)
    st.caption(f"產生時間：{datetime.now():%Y-%m-%d %H:%M:%S}｜行情快取30分鐘。本頁僅供研究，不構成投資建議。")
    st.stop()

with st.spinner("同步行情與總經資料……"):
    imf_macro,macro_error=imf_data(); stockq_macro,stockq_error=stockq_data(); mm_export,mm_export_note=macromicro_export_data(); macro=combine_macro_sources(imf_macro,stockq_macro,mm_export); index_data={m:analyze(c["index"]) for m,c in MARKETS.items()}; etf_data={m:analyze(c["etf"]) for m,c in MARKETS.items()}; factor_data={name:analyze(cfg["symbol"]) for name,cfg in GLOBAL_FACTORS.items()}; commodity_data={name:analyze(cfg["symbol"]) for name,cfg in COMMODITY_ASSETS.items()}
official_vol=official_market_volatility()
market_volatility={
    market:realized_volatility_proxy(index_data[market],f"{market}指數20日實現波動率","Yahoo Finance指數行情代理")
    for market in MARKETS
}
market_volatility["台灣"]=official_vol.get("台灣") if "error" not in official_vol.get("台灣",{"error":1}) else market_volatility["台灣"]
market_volatility["日本"]=official_vol.get("日本") if "error" not in official_vol.get("日本",{"error":1}) else market_volatility["日本"]
market_volatility["韓國"]={**factor_data["VIX恐慌指數"],"name":"US VIX（依指定）","source":"CBOE行情／Yahoo Finance","proxy":False}
cash,cash_date=twse_flow(); futures,futures_date=taifex_positions()
tabs=st.tabs(["🏁 市場結論","📈 價量技術","🌍 全球因子","💧 資金流／法人","🌐 IMF總經","🏭 產業評估","🪙 黃金／石油","🧮 方法"])

with tabs[0]:
    rows=[]; cash_total=cash["買賣超億元"].sum() if not cash.empty else 0; foreign_oi=futures.loc[futures["法人"].astype(str).str.contains("外資"),"淨未平倉口數"].sum() if not futures.empty else 0
    for market in MARKETS:
        s=index_data[market]; e=etf_data[market]
        if "error" in s: rows.append({"市場":market,"結論":"資料不足"}); continue
        chip=np.clip(cash_total/100,-5,5)+np.clip(foreign_oi/15000,-5,5) if market=="台灣" else 0
        local_factors=factors_for_market(factor_data,market_volatility,market)
        factor_score,factor_notes=global_factor_adjustment(market,local_factors,factor_weights)
        semi_score,semi_pressure=semiconductor_derating(market,local_factors,rate,factor_weights)
        export_score,export_growth=export_factor_score(market,macro,export_weight)
        total=np.clip(s["technical"]*.45+macro_score(market,macro)*.30+12.5+preset["bias"].get(market,0)+stock*.35-max(rate,0)/50+min(rate,0)/-100+np.clip(e.get("flow",0),-5,5)+chip+factor_score+semi_score+export_score,0,100)
        rows.append({"市場":market,"日期":s["date"],"指數":s["close"],"日漲跌%":s["day"],"1M%":s["m1"],"3M%":s["m3"],"價量":s["價量判讀"],"技術階段":s["階段判讀"],"技術分":s["technical"],"總經分":macro_score(market,macro),"最新出口成長%":export_growth,"出口敏感係數":EXPORT_SENSITIVITY.get(market),"出口因子分":export_score,"全球因子分":factor_score,"半導體去估值分":semi_score,"去估值壓力":semi_pressure,"ETF流向代理":e.get("flow"),"情境總分":total,"結論":verdict(total),"因子解讀":"；".join(factor_notes)})
    ranking=pd.DataFrame(rows)
    if "情境總分" in ranking.columns: ranking=ranking.sort_values("情境總分",ascending=False,na_position="last")
    st.subheader(f"情境：{scenario_name}｜利率 {rate:+d}bps｜股票 {stock:+d}%｜債券 {bond:+d}%")
    ranked=ranking.dropna(subset=["情境總分"]) if "情境總分" in ranking.columns else pd.DataFrame()
    if not ranked.empty: st.metric("目前相對優先市場",ranked.iloc[0]["市場"],ranked.iloc[0]["結論"])
    if not ranking.empty: st.dataframe(ranking,hide_index=True,width="stretch")
    else: st.info("行情來源暫時無回應，請稍後按清除快取並更新。")
    st.subheader("📊 各市場因子圖表")
    home_market=st.selectbox("市場因子長條圖",list(MARKETS),key="home_factor_market")
    home_breakdown=factor_breakdown(home_market,factors_for_market(factor_data,market_volatility,home_market),factor_weights,macro,export_weight,rate)
    home_breakdown["方向"]=np.where(home_breakdown["分數貢獻"]>=0,"正向支持","負向壓力")
    home_bar=alt.Chart(home_breakdown).mark_bar().encode(
        x=alt.X("分數貢獻:Q",title="對市場評分影響（分）"),y=alt.Y("因子:N",sort="-x"),
        color=alt.Color("方向:N",scale=alt.Scale(domain=["正向支持","負向壓力"],range=["#2ca02c","#d62728"])),
        tooltip=["市場:N","因子:N",alt.Tooltip("1M變動%:Q",format=".2f"),alt.Tooltip("權重:Q",format=".1f"),alt.Tooltip("分數貢獻:Q",format="+.2f")]
    ).properties(height=330,title=f"{home_market}｜因子貢獻長條圖")
    st.altair_chart(home_bar,width="stretch")
    home_all=pd.concat([factor_breakdown(m,factors_for_market(factor_data,market_volatility,m),factor_weights,macro,export_weight,rate) for m in MARKETS],ignore_index=True)
    home_heat=alt.Chart(home_all).mark_rect().encode(
        x=alt.X("因子:N",title=None),y=alt.Y("市場:N",title=None),
        color=alt.Color("分數貢獻:Q",scale=alt.Scale(domain=[-10,0,10],range=["#d62728","#f2f2f2","#2ca02c"]),title="分數"),
        tooltip=["市場:N","因子:N",alt.Tooltip("分數貢獻:Q",format="+.2f"),alt.Tooltip("權重:Q",format=".1f")]
    ).properties(height=360,title="全球主要市場｜因子貢獻熱力圖")
    st.altair_chart(home_heat,width="stretch")
    st.warning("這是相對排序，不是保證進場訊號；極端事件、匯率與政策可能迅速改變結果。")

with tabs[1]:
    st.subheader("選擇國家查看技術線")
    technical_market=st.selectbox("市場",list(MARKETS),key="technical_market",label_visibility="collapsed")
    s=index_data[technical_market]
    if "error" in s: st.error(s["error"])
    else:
        st.markdown(f"### {technical_market}｜{MARKETS[technical_market]['index']} 技術分析｜資料日 {s['date']}")
        render_metric_grid([
            ("指數",f"{s['close']:,.2f}",f"{s['day']:+.2f}%"),
            ("量／20日均量",f"{s['volume_ratio']:.2f}x",None),
            ("RSI14",f"{s['rsi']:.1f}",None),
            ("支撐",f"{s['support']:,.2f}",None),
            ("壓力",f"{s['resistance']:,.2f}",None),
        ])
        render_metric_grid([
            ("價量狀態",s["價量判讀"],None),
            ("KD",s["KD判讀"],None),
            ("MACD",s["MACD判讀"],None),
            ("型態階段",s["階段判讀"],None),
            ("籌碼階段",s["籌碼判讀"],None),
        ])
        st.altair_chart(line_chart(s["df"]),width="stretch"); kd_chart,macd_chart=oscillator_charts(s["df"]); c1,c2=st.columns(2); c1.altair_chart(kd_chart,width="stretch"); c2.altair_chart(macd_chart,width="stretch")
        tc1,tc2=st.columns([2,1]); tc1.altair_chart(turnover_chart(s["df"]),width="stretch")
        with tc2:
            st.subheader("籌碼面診斷分析")
            st.markdown(f"### {s['籌碼判讀']}")
            st.write(s["籌碼理由"])
            st.caption("此為指數成交量／換手代理判讀，不代表可識別特定主力帳戶。")
        st.caption(f"支撐／壓力採近60日低高價10%／90%分位；ATR14={s['atr']:,.2f}。換手強度={s['換手強度']:.2f}倍、20日振幅={s['20日振幅%']:.2f}%。")
        overview=pd.DataFrame([{"市場":m,"價量判讀":v.get("價量判讀"),"KD判讀":v.get("KD判讀"),"MACD判讀":v.get("MACD判讀"),"階段判讀":v.get("階段判讀"),"籌碼判讀":v.get("籌碼判讀"),"換手強度":v.get("換手強度"),"技術分":v.get("technical")} for m,v in index_data.items() if "error" not in v])
        st.subheader("全球主要市場技術線判讀"); st.dataframe(overview,hide_index=True,width="stretch")

with tabs[2]:
    st.subheader("美元、原油、利率與風險情緒")
    risk_score,risk_label=risk_sentiment(factors_for_market(factor_data,market_volatility,technical_market))
    r1,r2=st.columns(2); r1.metric("避險情緒指標（0–100）",f"{risk_score:.1f}"); r2.metric("目前狀態",risk_label)
    st.progress(int(risk_score)); st.caption(f"目前顯示{technical_market}避險情緒；由當地波動率、黃金、美元與比特幣近1個月變動合成。VIX預設權重已降為0.4倍。")
    st.subheader("各市場波動率來源與變動")
    vol_rows=[]
    for market,v in market_volatility.items():
        vol_rows.append({"市場":market,"波動率指標":v.get("name"),"目前值":v.get("close"),"1M變動%":v.get("m1"),"資料來源":v.get("source"),"是否代理":"是" if v.get("proxy") else "否","官方連結":VOLATILITY_SOURCES[market]["url"]})
    st.dataframe(pd.DataFrame(vol_rows),hide_index=True,width="stretch",column_config={"官方連結":st.column_config.LinkColumn("官方連結")})
    factor_rows=[]
    for name,cfg in GLOBAL_FACTORS.items():
        s=factor_data[name]
        if "error" in s: factor_rows.append({"因子":name,"代碼":cfg["symbol"],"狀態":s["error"]}); continue
        display_name="美國VIX（全球參考）" if name=="VIX恐慌指數" else name
        factor_rows.append({"因子":display_name,"代碼":cfg["symbol"],"日期":s["date"],"目前值":s["close"],"日變動%":s["day"],"1M%":s["m1"],"3M%":s["m3"],"判讀":cfg["description"]})
    st.dataframe(pd.DataFrame(factor_rows),hide_index=True,width="stretch")
    st.subheader("各市場因子分數圖")
    factor_market=st.selectbox("選擇市場查看因子貢獻",list(MARKETS),key="factor_chart_market")
    breakdown=factor_breakdown(factor_market,factors_for_market(factor_data,market_volatility,factor_market),factor_weights,macro,export_weight,rate)
    breakdown["方向"]=np.where(breakdown["分數貢獻"]>=0,"正向支持","負向壓力")
    contribution_chart=alt.Chart(breakdown).mark_bar().encode(
        x=alt.X("分數貢獻:Q",title="對市場評分的影響（分）"),y=alt.Y("因子:N",sort="-x"),
        color=alt.Color("方向:N",scale=alt.Scale(domain=["正向支持","負向壓力"],range=["#2ca02c","#d62728"])),
        tooltip=["市場:N","因子:N",alt.Tooltip("1M變動%:Q",format=".2f"),alt.Tooltip("權重:Q",format=".1f"),alt.Tooltip("分數貢獻:Q",format="+.2f")]
    ).properties(height=330,title=f"{factor_market}｜因子正負貢獻").interactive()
    st.altair_chart(contribution_chart,width="stretch")
    all_breakdowns=pd.concat([factor_breakdown(m,factors_for_market(factor_data,market_volatility,m),factor_weights,macro,export_weight,rate) for m in MARKETS],ignore_index=True)
    heatmap=alt.Chart(all_breakdowns).mark_rect().encode(
        x=alt.X("因子:N",title=None),y=alt.Y("市場:N",title=None),
        color=alt.Color("分數貢獻:Q",scale=alt.Scale(domain=[-10,0,10],range=["#d62728","#f2f2f2","#2ca02c"]),title="分數"),
        tooltip=["市場:N","因子:N",alt.Tooltip("分數貢獻:Q",format="+.2f"),alt.Tooltip("權重:Q",format=".1f")]
    ).properties(height=360,title="全球主要市場｜因子貢獻熱力圖")
    st.altair_chart(heatmap,width="stretch")
    st.dataframe(all_breakdowns,hide_index=True,width="stretch")
    available=[name for name,s in factor_data.items() if "error" not in s]
    if available:
        chosen_factor=st.selectbox("因子趨勢圖",available)
        st.altair_chart(line_chart(factor_data[chosen_factor]["df"]),width="stretch")
    st.caption("^TNX 是殖利率報價指數；其百分比報酬不等於殖利率上升的基點數。黃金與比特幣的屬性會隨市場環境改變，須與VIX、美元及利率交叉判讀。全球因子對市場總分的合計調整限制在 ±12 分。")

with tabs[3]:
    st.subheader("ETF價量資金流代理"); flows=flow_table(etf_data); st.dataframe(flows,hide_index=True,width="stretch")
    inflow=flows.loc[flows["判讀"].eq("🟢 流入"),"資金強度占比%"].sum(); outflow=flows.loc[flows["判讀"].eq("🔴 流出"),"資金強度占比%"].sum()
    f1,f2=st.columns(2); f1.metric("流入訊號占比",f"{inflow:.1f}%"); f2.metric("流出訊號占比",f"{outflow:.1f}%")
    st.info("量價流向＝近20日每日報酬×成交量的加權方向。資金強度占比是各市場絕對訊號的相對比例；正值標示流入、負值標示流出，不等同ETF實際申購／贖回金額。")
    st.subheader("台灣現貨三大法人｜TWSE"); st.dataframe(cash,hide_index=True,width="stretch") if not cash.empty else st.info(cash_date); st.caption(f"資料日：{cash_date}")
    st.subheader("台股期貨法人淨未平倉｜TAIFEX"); st.dataframe(futures,hide_index=True,width="stretch") if not futures.empty else st.info(futures_date); st.caption(f"資料日：{futures_date}；正值偏多、負值偏空，但不代表單一法人策略。")

with tabs[4]:
    if macro.empty: st.error("IMF與StockQ資料暫時無法取得。"+macro_error+"；"+stockq_error)
    else:
        pivot=macro.pivot_table(index="市場",columns="指標",values="數值",aggfunc="last").reset_index(); sources=macro.groupby("市場")["資料來源"].agg(lambda x:"／".join(dict.fromkeys(x))).rename("資料來源").reset_index(); st.dataframe(pivot.merge(sources,on="市場"),hide_index=True,width="stretch")
        st.subheader("出口循環敏感係數"); st.dataframe(pd.DataFrame([{"市場":m,"出口敏感係數":v,"說明":"模型傳導係數，非出口/GDP百分比"} for m,v in EXPORT_SENSITIVITY.items()]),hide_index=True,width="stretch")
        st.dataframe(macro[["市場","指標","數值","年度","資料來源"]],hide_index=True,width="stretch")
        st.caption("出口因子優先採財經M平方公開頁面的最新出口值年增率；香港依指定採中國出口值年增率作為代理，但仍使用香港自己的出口敏感係數。GDP、CPI與進口仍以IMF優先，缺值才由StockQ補充。")
        st.link_button("🔎 MacroMicro 各國出口年增率交叉驗證","https://www.macromicro.me/cross-country-database/exports-yoy")
        st.caption(f"M平方資料狀態：{mm_export_note}。該頁有反自動存取保護；若無法更新，系統不會中斷，而會保留已驗證值並以IMF資料備援。")

with tabs[5]:
    rows=[]
    for market,cfg in MARKETS.items():
        for sector,symbol in cfg["sectors"].items():
            s=analyze(symbol)
            if "error" in s: continue
            bias=sum(v for k,v in preset["sector"].items() if k in sector)
            semi_adj,semi_pressure=semiconductor_derating(market,factors_for_market(factor_data,market_volatility,market),rate,factor_weights) if any(k in sector for k in ("半導體","科技","電子")) else (0.0,"不適用")
            market_export,_=export_factor_score(market,macro,export_weight); export_sensitivity=1.0 if any(k in sector for k in ("半導體","科技","電子")) else .25 if any(k in sector for k in ("金融","銀行")) else .6; sector_export=market_export*export_sensitivity
            score=np.clip(s["technical"]+bias+stock*.25+semi_adj+sector_export,0,100)
            rows.append({"市場":market,"產業":sector,"代理ETF":symbol,"1M%":s["m1"],"3M%":s["m3"],"技術分":s["technical"],"出口因子分":sector_export,"去估值調整分":semi_adj,"去估值壓力":semi_pressure,"情境分":score,"結論":verdict(score)})
    sector_frame=pd.DataFrame(rows); st.dataframe(sector_frame.sort_values("情境分",ascending=False),hide_index=True,width="stretch") if not sector_frame.empty else st.info("產業ETF行情暫無資料。")
    st.subheader("光通訊｜基金觀察名單")
    st.dataframe(
        pd.DataFrame(OPTICAL_FUND_WATCHLIST).sort_values("相關持股%",ascending=False),
        hide_index=True,
        width="stretch",
        column_config={
            "相關持股%":st.column_config.NumberColumn("相關持股%",format="%.2f%%"),
            "持股／績效連結":st.column_config.LinkColumn("持股／績效連結",display_text="查看 MoneyDJ ↗"),
        },
    )
    st.caption("僅加總光_export.csv 內的光通訊股票；同一基金、同一股票只計一次，不混合不同月份，因此相關持股不會因重複分類超過100%。")

with tabs[6]:
    st.subheader("黃金、礦業、農業與石油市場")
    render_commodity_section(commodity_data,scenario_name,rate,stock,bond)

with tabs[7]:
    st.markdown("""### 評分框架
- 技術面45%：MA20／60／200、RSI、MACD、1M動能與量價代理。
- 技術型態：成交量達20日均量1.5倍視為爆量、低於0.8倍視為量縮；再結合5日報酬、MA20、KD與20日振幅判讀橫盤、拉升、洗盤及出貨警訊。
- 總經面30%：IMF實質GDP、CPI與進口量成長；出口另列為可調權重因子，優先採財經M平方最新出口值年增率，缺值時回退IMF出口量成長率，依台日韓中港出口循環敏感係數放大，最高正負12分。
- 總經備援：IMF缺少GDP或CPI時，從StockQ全球經濟數據公開表格補值並標示來源；不以網頁值取代IMF進出口序列。
- 中性基準25%，再加入情境、市場偏好與台灣法人籌碼調整。
- 全球因子：美元指數、WTI原油、黃金、比特幣、美國10年債殖利率與VIX，合計最多調整 ±12 分。
- 波動率採市場別資料：台灣TAIFEX VIX、日本Nikkei 225 VI、韓國依指定採美國VIX；香港與中國在官方歷史API不可用時採當地指數20日實現波動率並標示代理。VIX預設權重0.4倍。
- 半導體去估值：升息、美債殖利率、美元與VIX上升會增加折現率壓力；台灣與韓國曝險權重較高，最多扣10分。
- ETF資金流：以20日價量訊號標示流入／流出，並計算各市場絕對訊號的相對強度占比；不是實際申購贖回金額。
- 動態權重：全球六項因子與出口因子均可在側邊欄設為0至2倍，頁面會即時重算。
- 籌碼階段：以價格在60日區間的位置、5日／20日報酬、20日均線斜率、成交量相對20日均量、跌日／漲日量比及量價資金方向，區分吸籌、出貨、洗盤、拉貨；無明確共振時顯示籌碼結構穩定。
- 避險情緒指標：以近1個月 VIX、黃金、美元與比特幣變動合成；0偏風險偏好、100偏高度避險。
- 台灣現貨採TWSE BFI82U；台股期貨採TAIFEX OpenAPI。

本模板僅供研究，不構成投資建議。ETF量價代理不是實際基金流量。""")
    st.markdown("[IMF API](https://www.imf.org/external/datamapper/api/)｜[StockQ全球經濟數據](https://www.stockq.org/economy/worldstats.php)｜[TWSE](https://www.twse.com.tw/zh/trading/foreign/t86.html)｜[TAIFEX](https://www.taifex.com.tw/cht/3/futContractsDate)｜[TAIFEX OpenAPI](https://openapi.taifex.com.tw/)")

st.caption(f"產生時間：{datetime.now():%Y-%m-%d %H:%M:%S}｜行情與法人快取30分鐘、IMF快取6小時")
