import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os
import time

# --- Page setup ---
st.set_page_config(page_title="全方位戰情室 AI (v70.0 終極版)", layout="wide", page_icon="🏦")
st.markdown("### 🏦 全方位戰情室 AI (v70.0 終極多週期版)")

# --- [核心] NpEncoder (解決存檔崩潰) ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

# --- Persistence System ---
DATA_FILE = "trade_data_v70.json"

def save_data():
    data = {
        "balance": st.session_state.balance,
        "positions": st.session_state.positions,
        "pending_orders": st.session_state.pending_orders,
        "history": st.session_state.history
    }
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=NpEncoder, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"存檔失敗: {e}")

def load_data():
    if 'init_done' not in st.session_state:
        st.session_state.balance = 10000.0
        st.session_state.positions = []
        st.session_state.pending_orders = []
        st.session_state.history = []
        st.session_state.trade_amt_box = 1000.0
        st.session_state.chart_symbol = "BTC-USD"
        st.session_state.market = "加密貨幣"
        st.session_state.ai_entry = 0.0
        st.session_state.ai_tp = 0.0
        st.session_state.ai_sl = 0.0
        st.session_state.init_done = True

    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                st.session_state.balance = float(data.get("balance", 10000.0))
                st.session_state.positions = data.get("positions", [])
                st.session_state.pending_orders = data.get("pending_orders", [])
                st.session_state.history = data.get("history", [])
        except:
            st.session_state.balance = 10000.0

load_data()

# --- Helpers ---
def fmt_price(val):
    if val is None: return "N/A"
    try:
        valf = float(val)
        if valf < 0.01: return f"${valf:.6f}"
        elif valf < 20: return f"${valf:.4f}"
        else: return f"${valf:,.2f}"
    except: return str(val)

def get_current_price(sym):
    try:
        ticker = yf.Ticker(sym)
        # 嘗試從 fast_info 獲取，若失敗則抓 1 分鐘 K 線
        fi = getattr(ticker, 'fast_info', None)
        if fi and getattr(fi, 'last_price', None):
            return float(fi.last_price)
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except: pass
    return None

# --- [核心邏輯] 技術指標計算 (通用) ---
def calculate_indicators(df):
    if df is None or df.empty: return df
    df = df.copy()
    # EMA
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA60'] = df['Close'].ewm(span=60).mean()
    df['EMA120'] = df['Close'].ewm(span=120).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean().replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(14).mean()

    # ZigZag (簡易版)
    df['max_roll'] = df['High'].rolling(10, center=True).max()
    df['min_roll'] = df['Low'].rolling(10, center=True).min()
    
    # MACD
    exp12 = df['Close'].ewm(span=12).mean()
    exp26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    return df

# --- [超級核心] 多週期交叉分析 (Cross-Reference) ---
@st.cache_data(ttl=300) # 緩存 5 分鐘，避免頻繁請求
def get_mtf_analysis(symbol):
    intervals = {"M": "1mo", "W": "1wk", "D": "1d"}
    periods = {"M": "5y", "W": "2y", "D": "1y"}
    data_store = {}
    
    # 1. 抓取數據
    for tf, interval in intervals.items():
        try:
            df = yf.Ticker(symbol).history(period=periods[tf], interval=interval)
            if not df.empty:
                data_store[tf] = calculate_indicators(df)
        except:
            pass
            
    if not data_store: return None

    # 2. 分析各週期狀態
    scores = {"M": 0, "W": 0, "D": 0}
    trends = {}
    
    for tf, df in data_store.items():
        last = df.iloc[-1]
        score = 0
        trend_str = "震盪"
        
        # 均線排列
        if last['Close'] > last['EMA20'] > last['EMA60']:
            score += 2
            trend_str = "多頭排列"
        elif last['Close'] < last['EMA20'] < last['EMA60']:
            score -= 2
            trend_str = "空頭排列"
        else:
            # 判斷是回調還是反彈
            if last['Close'] > last['EMA60']: trend_str = "多頭回調"
            elif last['Close'] < last['EMA60']: trend_str = "空頭反彈"
        
        # RSI 過濾
        if last['RSI'] > 70: score -= 0.5 # 超買
        if last['RSI'] < 30: score += 0.5 # 超賣
        
        scores[tf] = score
        trends[tf] = trend_str

    # 3. 交叉比對 (Cross-Reference Logic)
    # 權重: 月(30%) + 週(30%) + 日(40%)
    total_score = (scores.get("M",0) * 0.3) + (scores.get("W",0) * 0.3) + (scores.get("D",0) * 0.4)
    
    # 產生建議
    direction = "觀望"
    if total_score >= 1.5: direction = "強力做多 (Strong Long)"
    elif total_score >= 0.5: direction = "嘗試做多 (Long)"
    elif total_score <= -1.5: direction = "強力做空 (Strong Short)"
    elif total_score <= -0.5: direction = "嘗試做空 (Short)"
    
    # 找入場點 (基於日線 ATR)
    last_d = data_store.get("D", data_store.get("W")).iloc[-1]
    curr_price = last_d['Close']
    atr = last_d.get('ATR', curr_price*0.02)
    
    if total_score > 0:
        entry = curr_price if trends.get("D") == "多頭回調" else curr_price # 如果正在回調就市價，否則追高
        # 若日線 EMA20 在下方，掛在 EMA20 附近
        if last_d['EMA20'] < curr_price:
            entry = (curr_price + last_d['EMA20']) / 2
        tp = entry + (atr * 3)
        sl = entry - (atr * 1.5)
    else:
        entry = curr_price
        if last_d['EMA20'] > curr_price:
            entry = (curr_price + last_d['EMA20']) / 2
        tp = entry - (atr * 3)
        sl = entry + (atr * 1.5)

    return {
        "score": total_score,
        "direction": direction,
        "trends": trends,
        "scores": scores,
        "entry": float(entry),
        "tp": float(tp),
        "sl": float(sl),
        "last_close": float(curr_price),
        "df_d": data_store.get("D") # 回傳日線給圖表用
    }

# --- Dialogs (取代 Modal) ---
@st.dialog("⚡ 倉位管理")
def manage_position_dialog(i, pos, current_price):
    st.markdown(f"**{pos.get('symbol','--')}** ({pos.get('type','--')} x{float(pos.get('lev',1)):.0f})")
    
    # 計算盈虧
    try:
        entry = float(pos.get('entry', 0))
        lev = float(pos.get('lev', 1))
        margin = float(pos.get('margin', 0))
        d = 1 if pos.get('type') == 'Long' else -1
        u_pnl = margin * (((current_price - entry) / entry) * lev * d)
        color = "green" if u_pnl >= 0 else "red"
        st.markdown(f"未結盈虧: <span style='color:{color}; font-weight:bold'>${u_pnl:+.2f}</span>", unsafe_allow_html=True)
    except: pass

    tab_close, tab_tpsl = st.tabs(["平倉", "止盈止損"])
    
    with tab_close:
        ratio = st.radio("平倉比例", [25,50,75,100], 3, horizontal=True, key=f"dr_{i}", format_func=lambda x:f"{x}%")
        if st.button("確認平倉", key=f"btn_c_{i}", type="primary", use_container_width=True):
            close_position(i, ratio, "手動", current_price)
            st.rerun()

    with tab_tpsl:
        c1, c2 = st.columns(2)
        cur_tp = float(pos.get('tp', 0))
        cur_sl = float(pos.get('sl', 0))
        new_tp = c1.number_input("TP 價格", value=cur_tp, key=f"ntp_{i}")
        new_sl = c2.number_input("SL 價格", value=cur_sl, key=f"nsl_{i}")
        if st.button("更新設定", key=f"btn_u_{i}", use_container_width=True):
            st.session_state.positions[i]['tp'] = new_tp
            st.session_state.positions[i]['sl'] = new_sl
            save_data()
            st.toast("✅ 已更新止盈止損")
            st.rerun()

def close_position(pos_index, percentage, reason, exit_price):
    if pos_index >= len(st.session_state.positions): return
    pos = st.session_state.positions[pos_index]
    
    # 計算部分平倉
    close_ratio = percentage / 100.0
    margin = float(pos.get('margin', 0))
    close_margin = margin * close_ratio
    
    direction = 1 if pos.get('type') == 'Long' else -1
    entry = float(pos.get('entry', 1))
    lev = float(pos.get('lev', 1))
    
    pnl = close_margin * (((exit_price - entry) / entry) * lev * direction)
    return_amount = close_margin + pnl
    
    st.session_state.balance += return_amount
    
    # 寫入歷史
    st.session_state.history.append({
        "時間": datetime.now().strftime("%m-%d %H:%M"),
        "幣種": pos.get('symbol'),
        "動作": f"{'全平' if percentage==100 else f'平{percentage}%'}",
        "價格": exit_price,
        "盈虧": round(pnl, 2),
        "原因": reason
    })

    if percentage == 100:
        st.session_state.positions.pop(pos_index)
    else:
        st.session_state.positions[pos_index]['margin'] -= close_margin
    
    save_data()

# --- Sidebar ---
st.sidebar.header("🎯 戰情室設定")
market = st.sidebar.radio("市場", ["加密貨幣", "美股", "台股"], index=0)
st.session_state.market = market

# 預設清單
if market == "加密貨幣":
    targets = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD", "BNB-USD"]
elif market == "美股":
    targets = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "COIN"]
else:
    targets = ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "0050.TW"]

col_search, col_select = st.sidebar.columns([1,2])
user_input = st.sidebar.text_input("輸入代碼 (例如 2330)", "")
selection = st.sidebar.selectbox("快速選擇", targets)

final_symbol = user_input.upper() if user_input.strip() else selection
if market == "台股" and final_symbol.isdigit(): final_symbol += ".TW"
if market == "加密貨幣" and "-" not in final_symbol and "USD" not in final_symbol: final_symbol += "-USD"

if final_symbol != st.session_state.chart_symbol:
    st.session_state.chart_symbol = final_symbol
    st.rerun()

symbol = st.session_state.chart_symbol

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ 清空所有數據 (重置)"):
    if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
    st.session_state.clear()
    st.rerun()

# --- Main Logic ---

# 1. 執行 AI 分析 (多週期)
with st.spinner(f"正在連線 {symbol} 進行多週期戰略分析..."):
    ai_data = get_mtf_analysis(symbol)

if ai_data:
    curr_price = ai_data['last_price']
    
    # 把 AI 建議存入 session 供下單區使用
    st.session_state.ai_entry = ai_data['entry']
    st.session_state.ai_tp = ai_data['tp']
    st.session_state.ai_sl = ai_data['sl']

    # --- Header ---
    c1, c2, c3 = st.columns([2, 1, 1])
    p_color = "#00C853" if ai_data['df_d'].iloc[-1]['Close'] >= ai_data['df_d'].iloc[-1]['Open'] else "#FF3D00"
    c1.markdown(f"<h1 style='margin:0'>{symbol} <span style='color:{p_color}'>${curr_price:,.2f}</span></h1>", unsafe_allow_html=True)
    c2.metric("可用餘額", f"${st.session_state.balance:,.2f}")
    
    # 計算總未結盈虧
    total_u_pnl = 0
    for p in st.session_state.positions:
        try:
            cur = get_current_price(p['symbol'])
            if cur:
                d = 1 if p['type']=='Long' else -1
                total_u_pnl += p['margin'] * (((cur - p['entry'])/p['entry']) * p['lev'] * d)
        except: pass
    c3.metric("總未結盈虧", f"${total_u_pnl:+.2f}", delta_color="normal")

    # --- AI Dashboard ---
    st.markdown("### 🧠 戰情室分析報告")
    
    # 顯示三個週期的狀態
    k1, k2, k3, k4 = st.columns(4)
    
    def get_arrow(trend):
        if "多頭" in trend: return "🟢"
        if "空頭" in trend: return "🔴"
        return "⚪"

    k1.info(f"**月線 (長期)**\n\n{get_arrow(ai_data['trends']['M'])} {ai_data['trends']['M']}")
    k2.info(f"**週線 (中期)**\n\n{get_arrow(ai_data['trends']['W'])} {ai_data['trends']['W']}")
    k3.info(f"**日線 (短期)**\n\n{get_arrow(ai_data['trends']['D'])} {ai_data['trends']['D']}")
    
    dir_color = "green" if "多" in ai_data['direction'] else ("red" if "空" in ai_data['direction'] else "gray")
    k4.markdown(f"""
    <div style='background-color:#262730; padding:10px; border-radius:5px; border: 1px solid {dir_color}; text-align:center'>
        <div style='font-size:12px; color:#aaa'>綜合戰略建議</div>
        <div style='font-size:18px; font-weight:bold; color:{dir_color}'>{ai_data['direction']}</div>
        <div style='font-size:12px'>信心分數: {ai_data['score']:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("查看詳細點位建議", expanded=True):
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("建議入場 (Entry)", fmt_price(ai_data['entry']))
        ec2.metric("目標止盈 (TP)", fmt_price(ai_data['tp']))
        ec3.metric("防守止損 (SL)", fmt_price(ai_data['sl']))

    # --- Chart Area ---
    df = ai_data['df_d'] # 使用日線繪圖
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # K線
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    # 均線
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='yellow', width=1), name='EMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA60'], line=dict(color='cyan', width=1), name='EMA60'), row=1, col=1)
    
    # 標註持倉線
    for pos in st.session_state.positions:
        if pos['symbol'] == symbol:
            fig.add_hline(y=pos['entry'], line_dash="dash", line_color="orange", annotation_text=f"持倉 {pos['type']}")
    
    # 指標
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='violet', width=2), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), dragmode='pan')
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Trading Area ---
    st.markdown("### ⚡ 交易控制台")
    
    tab1, tab2 = st.tabs(["下單", "持倉管理"])
    
    with tab1:
        c_t1, c_t2, c_t3 = st.columns(3)
        trade_type = c_t1.selectbox("方向", ["做多 (Long)", "做空 (Short)"], index=0 if "多" in ai_data['direction'] else 1)
        lev = c_t2.slider("槓桿倍數", 1, 125, 20)
        amt = c_t3.number_input("本金 (U)", min_value=10.0, value=float(st.session_state.trade_amt_box))
        st.session_state.trade_amt_box = amt
        
        # 自動填入 AI 建議
        with st.expander("進階設定 (止盈止損)", expanded=False):
            t_tp = st.number_input("止盈價格", value=st.session_state.ai_tp)
            t_sl = st.number_input("止損價格", value=st.session_state.ai_sl)
            t_entry = st.number_input("掛單價格 (0為市價)", value=0.0)

        if st.button("🚀 下單執行", type="primary", use_container_width=True):
            if amt > st.session_state.balance:
                st.error("餘額不足！")
            else:
                entry_price = curr_price if t_entry == 0 else t_entry
                new_pos = {
                    "symbol": symbol,
                    "type": "Long" if "多" in trade_type else "Short",
                    "entry": entry_price,
                    "lev": lev,
                    "margin": amt,
                    "tp": t_tp,
                    "sl": t_sl,
                    "time": datetime.now().strftime("%m-%d %H:%M")
                }
                
                if t_entry == 0:
                    st.session_state.positions.append(new_pos)
                    st.session_state.balance -= amt
                    st.toast(f"✅ 市價單成交！ {symbol}")
                else:
                    st.session_state.pending_orders.append(new_pos)
                    st.session_state.balance -= amt
                    st.toast(f"⏳ 掛單已提交！ {symbol}")
                
                save_data()
                st.rerun()

    with tab2:
        if not st.session_state.positions:
            st.info("目前無持倉")
        else:
            for i, pos in enumerate(st.session_state.positions):
                p_sym = pos['symbol']
                p_cur = get_current_price(p_sym)
                if p_cur:
                    # 顯示卡片
                    d = 1 if pos['type']=='Long' else -1
                    pnl = pos['margin'] * (((p_cur - pos['entry'])/pos['entry']) * pos['lev'] * d)
                    clr = "#00C853" if pnl >= 0 else "#FF3D00"
                    
                    st.markdown(f"""
                    <div style='border-left: 5px solid {clr}; padding: 10px; background: #262730; margin-bottom: 5px;'>
                        <div style='display:flex; justify-content:space-between'>
                            <strong>{p_sym} {pos['type']} x{pos['lev']}</strong>
                            <span style='color:{clr}'>${pnl:+.2f}</span>
                        </div>
                        <div style='font-size:12px; color:#aaa'>
                            開倉: {fmt_price(pos['entry'])} | 現價: {fmt_price(p_cur)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"⚙️ 管理 {p_sym}", key=f"mng_{i}", use_container_width=True):
                        manage_position_dialog(i, pos, p_cur)

else:
    st.error(f"無法獲取 {symbol} 的數據，請檢查代碼是否正確。")
