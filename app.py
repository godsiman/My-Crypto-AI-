import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from datetime import datetime
import json
import os

# --- Page setup ---
st.set_page_config(page_title="全方位戰情室 AI", layout="wide")
st.markdown("### 🏦 全方位戰情室 AI (v55.1 最終修復版)")

# --- Persistence System (語法修正) ---
DATA_FILE = "trade_data.json"

def save_data():
    data = {
        "balance": st.session_state.balance,
        "positions": st.session_state.positions,
        "pending_orders": st.session_state.pending_orders,
        "history": st.session_state.history
    }
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        st.error(f"存檔失敗: {e}")

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                st.session_state.balance = data.get("balance", 10000.0)
                st.session_state.positions = data.get("positions", [])
                st.session_state.pending_orders = data.get("pending_orders", [])
                st.session_state.history = data.get("history", [])
        except Exception as e:
            pass # 忽略讀取錯誤，使用預設值
    else:
        if 'balance' not in st.session_state: st.session_state.balance = 10000.0
        if 'positions' not in st.session_state: st.session_state.positions = []
        if 'pending_orders' not in st.session_state: st.session_state.pending_orders = []
        if 'history' not in st.session_state: st.session_state.history = []

if 'data_loaded' not in st.session_state:
    load_data()
    st.session_state.data_loaded = True

# Global State Init
if 'trade_amt_box' not in st.session_state: st.session_state.trade_amt_box = 1000.0
if 'ai_entry' not in st.session_state: st.session_state.ai_entry = 0.0
if 'ai_tp' not in st.session_state: st.session_state.ai_tp = 0.0
if 'ai_sl' not in st.session_state: st.session_state.ai_sl = 0.0

if 'chart_symbol' not in st.session_state: st.session_state.chart_symbol = "BTC-USD"
if 'market' not in st.session_state: st.session_state.market = "加密貨幣"

# --- Callbacks ---
def set_amt(ratio):
    st.session_state.trade_amt_box = float(st.session_state.balance * ratio)

# --- Helpers ---
def fmt_price(val):
    if val is None: return "N/A"
    try: valf = float(val)
    except: return str(val)
    if valf < 0.01: return f"${valf:.6f}"
    elif valf < 20: return f"${valf:.4f}"
    else: return f"${valf:,.2f}"

def get_current_price(sym):
    try:
        ticker = yf.Ticker(sym)
        if hasattr(ticker, 'fast_info') and getattr(ticker.fast_info, 'last_price', None): return float(ticker.fast_info.last_price)
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty: return float(hist['Close'].iloc[-1])
    except: return None
    return None

def calc_price_from_roe(entry, leverage, direction_str, roe_pct):
    if entry == 0: return 0.0
    direction = 1 if "Long" in direction_str or "做多" in direction_str else -1
    try: return float(entry * (1 + (roe_pct / 100) / (leverage * direction)))
    except: return 0.0

def calc_roe_from_price(entry, leverage, direction_str, target_price):
    if entry == 0: return 0.0
    direction = 1 if "Long" in direction_str or "做多" in direction_str else -1
    try: return float(((target_price - entry) / entry) * leverage * direction * 100)
    except: return 0.0

# --- Dialog ---
@st.dialog("⚡ 倉位管理", width="small")
def manage_position_dialog(i, pos, current_price):
    st.markdown(f"**{pos['symbol']}** ({pos['type']} x{pos['lev']})")
    st.caption(f"本金: {pos['margin']} U | 開倉: {fmt_price(pos['entry'])}")
    
    tab_close, tab_tpsl = st.tabs(["平倉", "止盈止損"])
    with tab_close:
        ratio = st.radio("Ratio", [25,50,75,100], 3, horizontal=True, key=f"d_r_{i}", format_func=lambda x:f"{x}%")
        if st.button("確認平倉", key=f"d_btn_close_{i}", type="primary", use_container_width=True):
            close_position(i, ratio, "手動", current_price); st.rerun()
    with tab_tpsl:
        current_tp = float(pos.get('tp', 0)); current_sl = float(pos.get('sl', 0))
        input_mode = st.radio("輸入單位", ["價格", "ROE %"], horizontal=True, key=f"d_mode_{i}")
        c_t, c_s = st.columns(2)
        if input_mode == "價格":
            t_val = c_t.number_input("TP", value=current_tp, key=f"d_t_p_{i}")
            s_val = c_s.number_input("SL", value=current_sl, key=f"d_s_p_{i}")
        else:
            def get_roe(p, d): return calc_roe_from_price(pos['entry'], pos['lev'], pos['type'], p) if p>0 else d
            t_roe = st.slider("止盈 %", 0.0, 500.0, float(f"{max(0.0, get_roe(current_tp, 30.0)):.2f}"), 5.0, key=f"d_t_s_{i}")
            s_roe = st.slider("止損 %", -100.0, 0.0, float(f"{min(0.0, get_roe(current_sl, -20.0)):.2f}"), 5.0, key=f"d_s_s_{i}")
            t_val = calc_price_from_roe(pos['entry'], pos['lev'], pos['type'], t_roe)
            s_val = calc_price_from_roe(pos['entry'], pos['lev'], pos['type'], s_roe)
            if t_val>0: st.success(f"TP: {fmt_price(t_val)}")
            if s_val>0: st.error(f"SL: {fmt_price(s_val)}")
        if st.button("更新", key=f"d_u_{i}", use_container_width=True):
            st.session_state.positions[i]['tp'] = t_val
            st.session_state.positions[i]['sl'] = s_val
            st.toast("已更新"); save_data(); st.rerun()

# --- Sidebar ---
st.sidebar.header("🎯 設定")
market = st.sidebar.radio("市場", ["加密貨幣", "美股", "台股"], index=0, key="market_radio")
st.session_state.market = market

crypto_list = ["BTC", "ETH", "SOL", "BNB", "DOGE", "XRP", "ADA", "AVAX"]
us_stock_list = ["AAPL", "NVDA", "TSLA", "MSFT", "META", "AMZN", "GOOGL", "AMD"]
tw_stock_dict = {"2330 台積電":"2330", "2454 聯發科":"2454", "2317 鴻海":"2317", "2603 長榮":"2603", "0050 元大台灣50":"0050"}

raw_symbol = "" 
if market == "加密貨幣": raw_symbol = st.sidebar.selectbox("快速選擇", crypto_list)
elif market == "美股": raw_symbol = st.sidebar.selectbox("快速選擇", us_stock_list)
else: raw_symbol = st.sidebar.selectbox("快速選擇", list(tw_stock_dict.keys()))

search_input = st.sidebar.text_input("代碼搜尋", placeholder="例如: 2330")
if search_input.strip(): raw_symbol = search_input.strip().upper()

final_symbol = raw_symbol
if market == "加密貨幣":
    if "USD" not in final_symbol and "-" not in final_symbol: final_symbol += "-USD"
elif market == "台股":
    if final_symbol.isdigit() or (len(final_symbol) == 4 and final_symbol.isdigit()): final_symbol += ".TW"
    elif not final_symbol.endswith(".TW") and not final_symbol.endswith(".TWO"): final_symbol += ".TW"

if 'chart_symbol' not in st.session_state: st.session_state.chart_symbol = final_symbol
if st.sidebar.button("🚀 載入 K 線"): st.session_state.chart_symbol = final_symbol; st.rerun()

symbol = st.session_state.chart_symbol 
interval_ui = st.sidebar.radio("週期", ["15分鐘", "1小時", "4小時", "日線"], index=3)

show_six = st.sidebar.checkbox("EMA 均線", True)
show_bb = st.sidebar.checkbox("布林通道", False) 
show_zigzag = st.sidebar.checkbox("SMC 結構", True)
show_fvg = st.sidebar.checkbox("SMC 缺口", True)
show_fib = st.sidebar.checkbox("Fib 止盈", True)
show_orders = st.sidebar.checkbox("圖表掛單", True)

st.sidebar.markdown("---")
with st.sidebar.expander("💰 錢包管理"):
    st.caption(f"餘額: ${st.session_state.balance:,.2f}")
    if st.button("🔄 重置為 1W U"): st.session_state.balance = 10000.0; st.session_state.positions = []; st.session_state.pending_orders = []; save_data(); st.rerun()
    if st.button("➕ 補血 +1W U"): st.session_state.balance += 10000.0; save_data(); st.rerun()

def get_params(ui_selection):
    if "15分鐘" in ui_selection: return "5d", "15m"
    elif "1小時" in ui_selection: return "1mo", "1h"
    elif "4小時" in ui_selection: return "6mo", "1h"
    else: return "2y", "1d"
period, interval = get_params(interval_ui)

@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df is None or df.empty: return None
        if interval == "1h" and "6mo" in period:
            logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df = df.resample('4h').apply(logic).dropna()
        df['Delta'] = df['Close'].diff()
        delta = df['Delta']
        gain = (delta.where(delta > 0, 0)).fillna(0); loss = (-delta.where(delta < 0, 0)).fillna(0)
        rs = gain.rolling(14).mean() / (loss.rolling(14).mean().replace(0, np.nan))
        df['RSI'] = 100 - (100 / (1 + rs))
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA60'] = df['Close'].ewm(span=60, adjust=False).mean()
        df['EMA120'] = df['Close'].ewm(span=120, adjust=False).mean()
        df['MA20'] = df['Close'].rolling(20).mean(); df['STD20'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['MA20'] + (df['STD20'] * 2); df['BB_Lower'] = df['MA20'] - (df['STD20'] * 2)
        exp12 = df['Close'].ewm(span=12).mean(); exp26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp12 - exp26; df['Signal'] = df['MACD'].ewm(span=9).mean(); df['Hist'] = df['MACD'] - df['Signal']
        prev_macd = df['MACD'].shift(1); prev_sig = df['Signal'].shift(1)
        df['MACD_Cross'] = 0
        df.loc[(prev_macd < prev_sig) & (df['MACD'] > df['Signal']), 'MACD_Cross'] = 1
        df.loc[(prev_macd > prev_sig) & (df['MACD'] < df['Signal']), 'MACD_Cross'] = -1
        df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(14).mean()
        return df.dropna(how='all')
    except: return None

# --- AI & Indicators Logic ---
def calculate_zigzag(df, depth=12):
    try:
        df = df.copy(); df['max_roll'] = df['High'].rolling(depth, center=True).max(); df['min_roll'] = df['Low'].rolling(depth, center=True).min()
        pivots = []
        for i in range(len(df)):
            if not np.isnan(df['max_roll'].iloc[i]) and df['High'].iloc[i] == df['max_roll'].iloc[i]:
                pivots.append({'idx': df.index[i], 'val': float(df['High'].iloc[i]), 'type': 'high'})
            elif not np.isnan(df['min_roll'].iloc[i]) and df['Low'].iloc[i] == df['min_roll'].iloc[i]:
                pivots.append({'idx': df.index[i], 'val': float(df['Low'].iloc[i]), 'type': 'low'})
        if len(pivots) >= 2:
            for i in range(2, len(pivots)):
                curr = pivots[i]; prev = pivots[i-2]
                if curr['type'] == 'high': curr['label'] = "HH" if curr['val'] > prev['val'] else "LH"
                else: curr['label'] = "LL" if curr['val'] < prev['val'] else "HL"
        return pivots
    except: return []

def calculate_fvg(df):
    try:
        bull, bear = [], []
        h, l, c = df['High'].values, df['Low'].values, df['Close'].values
        for i in range(max(2, len(df)-300), len(df)):
            if l[i] > h[i-2] and c[i-1] > h[i-2]: bull.append({'start': df.index[i-2], 'top': float(l[i]), 'bottom': float(h[i-2]), 'active': True})
            if h[i] < l[i-2] and c[i-1] < l[i-2]: bear.append({'start': df.index[i-2], 'top': float(l[i-2]), 'bottom': float(h[i]), 'active': True})
        return bull, bear
    except: return [], []

# --- AI Analysis ---
def run_ai_analysis(df, pivots, fvg_bull, fvg_bear):
    last = df.iloc[-1]
    close = last['Close']
    atr = last.get('ATR', close * 0.02)
    score = 0
    reasons = []
    
    # 1. Trend Analysis
    ema20, ema60, ema120 = last.get('EMA20'), last.get('EMA60'), last.get('EMA120')
    if close > ema20 > ema60 > ema120: score += 3; reasons.append("均線多頭")
    elif close < ema20 < ema60 < ema120: score -= 3; reasons.append("均線空頭")
    
    # 2. Structure Analysis
    if len(pivots) >= 2:
        last_p = pivots[-1]
        if last_p['type'] == 'high':
            if 'label' in last_p and 'H' in last_p['label']: score += 2; reasons.append("結構創高")
            else: score -= 1
        else:
            if 'label' in last_p and 'L' in last_p['label']: score -= 2; reasons.append("結構破底")
            else: score += 1

    # 3. Momentum
    if last['RSI'] < 30: score += 2; reasons.append("RSI 超賣")
    elif last['RSI'] > 70: score -= 2; reasons.append("RSI 超買")
    
    # 4. Support/Resistance
    if fvg_bull:
        latest_bull = fvg_bull[-1]
        if close > latest_bull['top'] and (close - latest_bull['top'])/close < 0.01: score += 2; reasons.append("FVG 支撐")
    if fvg_bear:
        latest_bear = fvg_bear[-1]
        if close < latest_bear['bottom'] and (latest_bear['bottom'] - close)/close < 0.01: score -= 2; reasons.append("FVG 壓力")

    # Final
    direction = "做多 (Long)" if score > 0 else "做空 (Short)"
    confidence = min(abs(score), 10)
    
    # Calc
    if score > 0:
        entry_price = ema20 if ema20 < close else (close - atr)
        sl_price = entry_price - (1.5 * atr)
        tp_price = entry_price + (2.5 * atr) 
    else:
        entry_price = ema20 if ema20 > close else (close + atr)
        sl_price = entry_price + (1.5 * atr)
        tp_price = entry_price - (2.5 * atr)
        
    return {"score": score, "dir": direction, "conf": confidence, "entry": entry_price, "tp": tp_price, "sl": sl_price, "reasons": ", ".join(reasons)}

def close_position(pos_index, percentage=100, reason="手動平倉", exit_price=None):
    if pos_index >= len(st.session_state.positions): return
    pos = st.session_state.positions[pos_index]
    if exit_price is None: exit_price = get_current_price(pos['symbol']) or pos['entry']
    close_margin = pos['margin'] * (percentage / 100)
    direction = 1 if pos['type'] == 'Long' else -1
    try: pnl_pct = ((exit_price - pos['entry']) / pos['entry']) * pos['lev'] * direction * 100
    except: pnl_pct = 0
    pnl_usdt = close_margin * (pnl_pct / 100)
    st.session_state.balance += (close_margin + pnl_usdt)
    st.session_state.history.append({"時間": datetime.now().strftime("%m-%d %H:%M"), "幣種": pos['symbol'], "動作": f"平倉 {percentage}%", "入場": pos['entry'], "出場": exit_price, "損益(U)": round(pnl_usdt, 2), "獲利%": round(pnl_pct, 2), "原因": reason})
    if percentage == 100: st.session_state.positions.pop(pos_index); st.toast(f"✅ 全平 {pos['symbol']}")
    else: st.session_state.positions[pos_index]['margin'] -= close_margin; st.toast(f"✅ 部分平倉 {pos['symbol']}")
    save_data()

def cancel_pending_order(idx):
    if idx < len(st.session_state.pending_orders):
        ord = st.session_state.pending_orders.pop(idx); st.session_state.balance += ord['margin']; st.toast(f"🗑️ 已撤銷"); save_data(); st.rerun()

# --- Main Page ---
df = get_data(symbol, period, interval)

if df is not None and not df.empty:
    last = df.iloc[-1]; curr_price = float(last['Close'])

    # Pending Orders Check
    pending_updated = False
    if st.session_state.pending_orders:
        for i in reversed(range(len(st.session_state.pending_orders))):
            ord = st.session_state.pending_orders[i]
            is_filled = False
            if ord['type'] == 'Long' and curr_price <= ord['entry']: is_filled = True
            elif ord['type'] == 'Short' and curr_price >= ord['entry']: is_filled = True
            if is_filled:
                new_pos = st.session_state.pending_orders.pop(i); new_pos['time'] = datetime.now().strftime('%m-%d %H:%M')
                st.session_state.positions.append(new_pos); st.toast(f"🔔 成交！{new_pos['symbol']}"); pending_updated = True
    if pending_updated: save_data()

    # --- AI Analysis ---
    pivots = calculate_zigzag(df); bull_fvg, bear_fvg = calculate_fvg(df)
    ai_res = run_ai_analysis(df, pivots, bull_fvg, bear_fvg)
    
    # Store AI values
    st.session_state.ai_entry = ai_res['entry']
    st.session_state.ai_tp = ai_res['tp']
    st.session_state.ai_sl = ai_res['sl']

    # --- AI Dashboard ---
    st.markdown("### 🧠 AI 戰略指揮中心")
    c_ai1, c_ai2, c_ai3 = st.columns([1.5, 1, 1.5])
    
    with c_ai1:
        color = "green" if ai_res['score'] > 0 else "red"
        st.markdown(f"#### 建議: :{color}[{ai_res['dir']}]")
        st.caption(f"評分: {ai_res['conf']}/10")
        st.write(f"💡 理由: {ai_res['reasons']}")
    
    with c_ai2:
        st.metric("建議入場", fmt_price(ai_res['entry']))
        if st.button("📋 一鍵帶入"): st.toast("已填入下單區")
    
    with c_ai3:
        c_tp, c_sl = st.columns(2)
        c_tp.metric("目標止盈", fmt_price(ai_res['tp']))
        c_sl.metric("防守止損", fmt_price(ai_res['sl']))

    st.divider()

    # --- Chart ---
    indicator_mode = st.radio("副圖", ["RSI", "MACD"], horizontal=True, label_visibility="collapsed")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25], subplot_titles=("價格", "成交量", indicator_mode))
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    if show_six:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20', line=dict(width=1, color='yellow')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA60'], name='EMA60', line=dict(width=1, color='cyan')), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB', line=dict(width=1, color='rgba(255,255,255,0.3)')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB', line=dict(width=1, color='rgba(255,255,255,0.3)'), fill='tonexty', fillcolor='rgba(255,255,255,0.05)'), row=1, col=1)
    if show_fvg:
        for f in bull_fvg: fig.add_shape(type="rect", x0=f['start'], x1=df.index[-1], y0=f['bottom'], y1=f['top'], fillcolor="rgba(0,255,0,0.2)", line_width=0, xref='x', yref='y', row=1, col=1)
        for f in bear_fvg: fig.add_shape(type="rect", x0=f['start'], x1=df.index[-1], y0=f['bottom'], y1=f['top'], fillcolor="rgba(255,0,0,0.15)", line_width=0, xref='x', yref='y', row=1, col=1)
    if show_zigzag and pivots:
        px = [p['idx'] for p in pivots]; py = [p['val'] for p in pivots]
        fig.add_trace(go.Scatter(x=px, y=py, mode='lines+markers', name='ZigZag', line=dict(color='orange', width=2), marker_size=4), row=1, col=1)
        for p in pivots[-10:]:
            if 'label' in p:
                label_clr = '#00FF00' if 'H' in p['label'] and p['type'] == 'high' else 'red'
                fig.add_annotation(x=p['idx'], y=p['val'], text=p['label'], showarrow=False, font=dict(color=label_clr, size=10), yshift=15 if p['type']=='high' else -15, row=1, col=1)
    
    if st.session_state.ai_entry > 0:
        fig.add_hline(y=st.session_state.ai_entry, line_dash="dot", line_color="white", annotation_text="AI 進場", row=1, col=1)
        fig.add_hline(y=st.session_state.ai_tp, line_dash="dot", line_color="#00FF00", annotation_text="AI 止盈", row=1, col=1)
        fig.add_hline(y=st.session_state.ai_sl, line_dash="dot", line_color="red", annotation_text="AI 止損", row=1, col=1)

    colors = ['#00C853' if c >= o else '#FF3D00' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Vol', marker_color=colors), row=2, col=1)
    if indicator_mode == "RSI":
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(width=2, color='violet')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1); fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    else: 
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(width=1, color='cyan')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(width=1, color='orange')), row=3, col=1)
        hist_colors = ['#00C853' if h >= 0 else '#FF3D00' for h in df['Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['Hist'], name='Hist', marker_color=hist_colors), row=3, col=1)
    fig.update_layout(template="plotly_dark", height=700, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, dragmode='pan')
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

    # --- Wallet & Panel ---
    st.divider()
    total_unrealized = 0; total_margin = 0
    if st.session_state.positions:
        for pos in st.session_state.positions:
            lp = get_current_price(pos['symbol'])
            if lp:
                d = 1 if pos['type'] == 'Long' else -1
                total_unrealized += pos['margin'] * (((lp - pos['entry']) / pos['entry']) * pos['lev'] * d)
                total_margin += pos['margin']
    equity = st.session_state.balance + total_margin + total_unrealized
    if equity <= 0: st.error("💀 帳戶爆倉！"); st.session_state.positions=[]; st.session_state.pending_orders=[]; st.session_state.balance=0; save_data(); st.rerun()

    c_w1, c_w2, c_w3 = st.columns(3)
    c_w1.metric("💰 權益", f"${equity:,.2f}")
    c_w2.metric("💵 餘額", f"${st.session_state.balance:,.2f}")
    c_w3.metric("🔥 盈虧", f"${total_unrealized:+.2f} U", delta_color="normal")

    tab_trade, tab_ord, tab_hist = st.tabs(["🚀 下單", "📋 委託", "📜 歷史"])
    
    with tab_trade:
        order_type = st.radio("類型", ["⚡ 市價", "⏱️ 掛單"], horizontal=True, label_visibility="collapsed")
        c1, c2 = st.columns(2)
        side = c1.selectbox("方向", ["🟢 做多", "🔴 做空"], index=0 if ai_res['dir']=="做多 (Long)" else 1)
        lev = c2.number_input("槓桿", 1, 125, 20)
        
        def_p = curr_price
        if "掛單" in order_type and st.session_state.ai_entry > 0: def_p = st.session_state.ai_entry
        entry_p = st.number_input("掛單價格", value=float(def_p), format="%.6f") if "掛單" in order_type else st.caption(f"市價約: {fmt_price(curr_price)}") or curr_price
        
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        if c_p1.button("25%", use_container_width=True, on_click=set_amt, args=(0.25,)): pass
        if c_p2.button("50%", use_container_width=True, on_click=set_amt, args=(0.50,)): pass
        if c_p3.button("75%", use_container_width=True, on_click=set_amt, args=(0.75,)): pass
        if c_p4.button("Max", use_container_width=True, on_click=set_amt, args=(1.00,)): pass
        amt = st.number_input("本金 (U)", value=float(st.session_state.trade_amt_box), min_value=1.0, key="trade_amt_box")
        
        with st.expander("止盈止損 (預設 AI 建議)", expanded=True):
            def_tp = st.session_state.ai_tp if st.session_state.ai_tp > 0 else 0.0
            def_sl = st.session_state.ai_sl if st.session_state.ai_sl > 0 else 0.0
            new_tp = st.number_input("止盈", value=float(def_tp))
            new_sl = st.number_input("止損", value=float(def_sl))
            
        btn_txt = "買入/賣出 (市價)" if "市價" in order_type else "提交掛單"
        if st.button(btn_txt, type="primary", use_container_width=True):
            if amt > st.session_state.balance: st.error("餘額不足")
            else:
                new_ord = {"symbol": symbol, "type": "Long" if "做多" in side else "Short", "entry": entry_p, "lev": lev, "margin": amt, "tp": new_tp, "sl": new_sl, "time": datetime.now().strftime('%m-%d %H:%M')}
                if "市價" in order_type: st.session_state.positions.append(new_ord); st.session_state.balance -= amt; st.toast("✅ 成交！")
                else: st.session_state.pending_orders.append(new_ord); st.session_state.balance -= amt; st.toast("⏳ 掛單已提交")
                save_data(); st.rerun()

    with tab_ord:
        if st.session_state.pending_orders:
            for i, ord in enumerate(st.session_state.pending_orders):
                c1, c2 = st.columns([3, 1])
                c1.write(f"**{ord['symbol']}** {ord['type']} @ {fmt_price(ord['entry'])}")
                if c2.button("撤銷", key=f"cx_{i}"): cancel_pending_order(i)
        else: st.info("無掛單")
    
    with tab_hist:
        if st.session_state.history: st.dataframe(pd.DataFrame(st.session_state.history[::-1]), hide_index=True)
        else: st.info("無紀錄")

    st.markdown("### 🔥 持倉列表")
    if not st.session_state.positions: st.info("目前無持倉")
    else:
        for i, pos in enumerate(st.session_state.positions):
            live = curr_price if pos['symbol'] == symbol else get_current_price(pos['symbol'])
            if live:
                d = 1 if pos['type'] == 'Long' else -1
                u_pnl = pos['margin'] * (((live - pos['entry']) / pos['entry']) * pos['lev'] * d)
                pnl_pct = (((live - pos['entry']) / pos['entry']) * pos['lev'] * d) * 100
                liq = pos['entry']*(1 - 1/pos['lev']) if pos['type']=='Long' else pos['entry']*(1 + 1/pos['lev'])
                if (pos['type']=='Long' and live<=liq) or (pos['type']=='Short' and live>=liq): close_position(i, 100, "💀 爆倉", live); st.rerun()
                elif pos.get('tp',0)>0 and ((pos['type']=='Long' and live>=pos['tp']) or (pos['type']=='Short' and live<=pos['tp'])): close_position(i, 100, "🎯 止盈", live); st.rerun()
                elif pos.get('sl',0)>0 and ((pos['type']=='Long' and live<=pos['sl']) or (pos['type']=='Short' and live>=pos['sl'])): close_position(i, 100, "🛡️ 止損", live); st.rerun()

                col_h1, col_h2 = st.columns([4, 1])
                col_h1.markdown(f"**#{i+1} {pos['symbol']}**")
                if col_h2.button(f"🔍", key=f"jump_{i}"): st.session_state.chart_symbol = pos['symbol']; st.rerun()

                clr = "#00C853" if u_pnl >= 0 else "#FF3D00"
                icon = "🟢" if pos['type'] == 'Long' else "🔴"
                st.markdown(f"""<div style="background-color: #262730; padding: 12px; border-radius: 8px; border-left: 5px solid {clr}; margin-bottom: 8px;"><div style="display: flex; justify-content: space-between; font-size: 13px; color: #ccc;"><span>{icon} {pos['type']} x{pos['lev']} <span style="color:#888;">(本金: {pos['margin']:.0f} U)</span></span><span>🕒 {pos.get('time','--')}</span></div><div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 5px;"><div><div style="font-size: 12px; color: #aaa;">未結盈虧 (U)</div><div style="font-size: 18px; font-weight: bold; color: {clr};">{u_pnl:+.2f} U</div></div><div style="text-align: right;"><div style="font-size: 12px; color: #aaa;">回報率 (%)</div><div style="font-size: 18px; font-weight: bold; color: {clr};">{pnl_pct:+.2f}%</div></div></div><div style="margin-top: 8px; font-size: 12px; color: #888; display: flex; justify-content: space-between;"><span>開: {fmt_price(pos['entry'])}</span><span>現: {fmt_price(live)}</span></div></div>""", unsafe_allow_html=True)
                if st.button("⚙️ 管理 / 平倉", key=f"m_{i}", use_container_width=True): manage_position_dialog(i, pos, live)
                st.markdown("---")

else: st.error(f"❌ 無法讀取 {symbol}")
