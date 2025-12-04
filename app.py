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
st.markdown("### 🏦 全方位戰情室 AI (v59.0 終極穩定版)")

# --- State Initialization ---
KEYS_TO_INIT = {
    'balance': 10000.0,
    'positions': [],
    'pending_orders': [],
    'history': [],
    'trade_amt_input_val': 1000.0,
    'ai_entry': 0.0,
    'ai_tp': 0.0,
    'ai_sl': 0.0,
    'chart_symbol': 'BTC-USD',
    'market': '加密貨幣'
}

for k, v in KEYS_TO_INIT.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Persistence System (With Auto-Repair) ---
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
                st.session_state.balance = float(data.get("balance", 10000.0))
                
                # [自動修復] 檢查並修復壞掉的倉位數據
                raw_positions = data.get("positions", [])
                valid_positions = []
                for pos in raw_positions:
                    try:
                        # 確保關鍵欄位存在且為正確類型
                        if 'symbol' in pos and 'entry' in pos and 'margin' in pos:
                            pos['entry'] = float(pos['entry'])
                            pos['margin'] = float(pos['margin'])
                            pos['lev'] = float(pos.get('lev', 1))
                            pos['tp'] = float(pos.get('tp', 0))
                            pos['sl'] = float(pos.get('sl', 0))
                            valid_positions.append(pos)
                    except:
                        continue # 捨棄無法修復的壞資料
                
                st.session_state.positions = valid_positions
                st.session_state.pending_orders = data.get("pending_orders", [])
                st.session_state.history = data.get("history", [])
        except:
            pass 

if 'has_loaded_data' not in st.session_state:
    load_data()
    st.session_state.has_loaded_data = True

# --- Callbacks ---
def set_amt(ratio):
    val = float(st.session_state.balance * ratio)
    if val < 0: val = 0.0
    st.session_state.trade_amt_input_val = val

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
        if hasattr(ticker, 'fast_info') and getattr(ticker.fast_info, 'last_price', None):
            return float(ticker.fast_info.last_price)
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

# --- Dialog Functions ---
@st.dialog("⚡ 倉位管理", width="small")
def manage_position_dialog(i, pos, current_price):
    st.markdown(f"**{pos['symbol']}** ({pos['type']} x{pos['lev']})")
    st.caption(f"本金: {pos['margin']} U | 開倉: {fmt_price(pos['entry'])}")
    
    tab_close, tab_tpsl = st.tabs(["平倉", "止盈止損"])
    
    with tab_close:
        st.write("選擇平倉比例:")
        ratio = st.radio("Ratio", [25,50,75,100], 3, horizontal=True, key=f"d_r_{i}", format_func=lambda x:f"{x}%")
        if st.button("確認平倉", key=f"d_btn_close_{i}", type="primary", use_container_width=True):
            close_position(i, ratio, "手動", current_price)
            st.rerun()

    with tab_tpsl:
        current_tp = float(pos.get('tp', 0))
        current_sl = float(pos.get('sl', 0))
        input_mode = st.radio("輸入單位", ["價格", "盈虧 % (ROE)"], horizontal=True, key=f"d_mode_{i}")
        
        c_t, c_s = st.columns(2)
        
        if input_mode == "價格":
            t_val = c_t.number_input("TP", value=current_tp, key=f"d_t_p_{i}")
            s_val = c_s.number_input("SL", value=current_sl, key=f"d_s_p_{i}")
        else:
            def get_roe_val(price, default):
                if price > 0: return calc_roe_from_price(pos['entry'], pos['lev'], pos['type'], price)
                return default

            tp_roe_init = get_roe_val(current_tp, 30.0)
            sl_roe_init = get_roe_val(current_sl, -20.0)
            
            t_roe = st.slider("止盈 %", 0.0, 500.0, float(f"{max(0.0, tp_roe_init):.2f}"), 5.0, key=f"d_t_s_{i}")
            s_roe = st.slider("止損 %", -100.0, 0.0, float(f"{min(0.0, sl_roe_init):.2f}"), 5.0, key=f"d_s_s_{i}")
            
            t_val = calc_price_from_roe(pos['entry'], pos['lev'], pos['type'], t_roe)
            s_val = calc_price_from_roe(pos['entry'], pos['lev'], pos['type'], s_roe)
            
            if t_val > 0: c_t.caption(f"≈ {fmt_price(t_val)}")
            if s_val > 0: c_s.caption(f"≈ {fmt_price(s_val)}")

        if st.button("更新策略", key=f"d_u_{i}", use_container_width=True):
            st.session_state.positions[i]['tp'] = t_val
            st.session_state.positions[i]['sl'] = s_val
            st.toast("策略已更新")
            save_data()
            st.rerun()

# --- Sidebar ---
st.sidebar.header("🎯 設定")
market = st.sidebar.radio("市場", ["加密貨幣", "美股", "台股"], index=0, key="market_radio")
st.session_state.market = market

crypto_list = ["BTC", "ETH", "SOL", "BNB", "DOGE", "XRP", "ADA", "AVAX"]
us_stock_list = ["AAPL", "NVDA", "TSLA", "MSFT", "META", "AMZN", "GOOGL", "AMD"]
tw_stock_dict = {
    "2330 台積電": "2330", "2454 聯發科": "2454", "2317 鴻海": "2317", "2303 聯電": "2303",
    "2603 長榮": "2603", "2609 陽明": "2609", "2615 萬海": "2615", "0050 元大台灣50": "0050",
    "00878 國泰永續高股息": "00878"
}

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
if st.sidebar.button("🚀 載入 K 線"):
    st.session_state.chart_symbol = final_symbol
    st.rerun()

symbol = st.session_state.chart_symbol 
interval_ui = st.sidebar.radio("週期", ["15分鐘", "1小時", "4小時", "日線"], index=3)

# 視覺化開關
show_six = st.sidebar.checkbox("EMA 均線", True)
show_bb = st.sidebar.checkbox("布林通道", False) 
show_zigzag = st.sidebar.checkbox("SMC 結構 (ZigZag)", True)
show_fvg = st.sidebar.checkbox("SMC 缺口 (FVG)", True)
show_fib = st.sidebar.checkbox("Fib (止盈)", True)
show_orders = st.sidebar.checkbox("圖表掛單", True)

# --- 錢包管理 ---
st.sidebar.markdown("---")
with st.sidebar.expander("💰 錢包管理 (修復工具)"):
    st.caption(f"可用餘額: ${st.session_state.balance:,.2f}")
    if st.button("🔄 重置為 1W U"):
        st.session_state.balance = 10000.0
        st.session_state.positions = []
        st.session_state.pending_orders = []
        st.session_state.history = []
        save_data()
        st.rerun()
    if st.button("➕ 補血 +1W U"):
        st.session_state.balance += 10000.0
        save_data()
        st.rerun()
    if st.button("🧨 強制清空數據 (救命用)", type="primary"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        st.session_state.clear()
        st.rerun()

# --- Data Params ---
def get_params(ui_selection):
    if "15分鐘" in ui_selection: return "5d", "15m"
    elif "1小時" in ui_selection: return "1mo", "1h"
    elif "4小時" in ui_selection: return "6mo", "1h"
    else: return "2y", "1d"
period, interval = get_params(interval_ui)

@st.cache_data(ttl=60)
def get_mtf_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        df_daily = ticker.history(period="2y", interval="1d")
        df_hourly = ticker.history(period="1mo", interval="1h")
        
        if df_daily.empty: return None, None, None, None
        
        agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        df_weekly = df_daily.resample('W-MON').agg(agg_dict).dropna()
        df_monthly = df_daily.resample('ME').agg(agg_dict).dropna()
        
        return df_monthly, df_weekly, df_daily, df_hourly
    except: return None, None, None, None

def add_indicators(df):
    if df is None or df.empty: return df
    df = df.copy()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA60'] = df['Close'].ewm(span=60, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(lambda x: x>0, 0).rolling(14).mean() / df['Close'].diff().where(lambda x: x<0, 0).abs().rolling(14).mean().replace(0, np.nan))))
    return df

def analyze_trend(df):
    if df is None or len(df) < 20: return 0
    last = df.iloc[-1]
    if last['Close'] > last['EMA20']: return 1
    if last['Close'] < last['EMA20']: return -1
    return 0

def run_mtf_analysis(df_m, df_w, df_d, df_h):
    df_m = add_indicators(df_m); df_w = add_indicators(df_w)
    df_d = add_indicators(df_d); df_h = add_indicators(df_h)
    
    t_m = analyze_trend(df_m); t_w = analyze_trend(df_w)
    t_d = analyze_trend(df_d); t_h = analyze_trend(df_h)
    
    score = (t_m * 4) + (t_w * 3) + (t_d * 2) + (t_h * 1)
    
    direction = "觀望"
    if score >= 5: direction = "強力做多 (Strong Buy)"
    elif score >= 2: direction = "偏多 (Buy)"
    elif score <= -5: direction = "強力做空 (Strong Sell)"
    elif score <= -2: direction = "偏空 (Sell)"
    
    last_h = df_h.iloc[-1]
    atr = last_h['Close'] * 0.01
    entry = last_h['Close']
    tp = 0.0; sl = 0.0
    
    if score > 0:
        sl = entry - 2*atr; tp = entry + 3*atr
    elif score < 0:
        sl = entry + 2*atr; tp = entry - 3*atr
        
    return {"score": score, "dir": direction, "trends": [t_m, t_w, t_d, t_h], "entry": entry, "tp": tp, "sl": sl}

# --- Indicators ---
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
df_m, df_w, df_d, df_h = get_mtf_data(symbol)
df_chart = df_d
if "日" in interval_ui: df_chart = df_d
elif "4小時" in interval_ui or "1小時" in interval_ui: df_chart = df_h
else: df_chart = df_d

if df_chart is not None and not df_chart.empty:
    last = df_chart.iloc[-1]; curr_price = float(last['Close'])
    
    # [新增] 頂部行情看板
    st.markdown(f"""
    <h1 style='text-align: center; font-size: 40px;'>
        {symbol} <span style='color: {"#00C853" if df_chart['Close'].iloc[-1] >= df_chart['Open'].iloc[-1] else "#FF3D00"}'>
        ${curr_price:,.2f}</span>
    </h1>
    """, unsafe_allow_html=True)

    # Pending Orders Logic
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

    # Analysis
    mtf_res = run_mtf_analysis(df_m, df_w, df_d, df_h)
    st.session_state.ai_entry = mtf_res['entry']
    st.session_state.ai_tp = mtf_res['tp']
    st.session_state.ai_sl = mtf_res['sl']

    # --- Dashboard ---
    c_rad1, c_rad2, c_rad3 = st.columns([1.5, 1, 1.5])
    with c_rad1:
        color = "green" if mtf_res['score'] > 0 else "red" if mtf_res['score'] < 0 else "gray"
        st.markdown(f"#### 穩健建議: :{color}[{mtf_res['dir']}]")
        dots = "".join(["🟢 " if t==1 else "🔴 " if t==-1 else "⚪ " for t in mtf_res['trends']])
        st.write(f"趨勢: {dots} (月|週|日|時)")
    with c_rad2:
        st.metric("建議入場", fmt_price(mtf_res['entry']))
        if st.button("📋 帶入策略"): st.toast("已填入下單區")
    with c_rad3:
        c_tp, c_sl = st.columns(2)
        c_tp.metric("目標止盈", fmt_price(mtf_res['tp']))
        c_sl.metric("防守止損", fmt_price(mtf_res['sl']))

    st.divider()

    # --- Chart ---
    df_chart = add_indicators(df_chart)
    pivots = calculate_zigzag(df_chart); bull_fvg, bear_fvg = calculate_fvg(df_chart)
    indicator_mode = st.radio("副圖", ["RSI", "MACD"], horizontal=True, label_visibility="collapsed")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25], subplot_titles=("價格", "成交量", indicator_mode))
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='K線'), row=1, col=1)
    
    if show_six:
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA20'], name='EMA20', line=dict(width=1, color='yellow')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA60'], name='EMA60', line=dict(width=1, color='cyan')), row=1, col=1)
    if show_fvg:
        for f in bull_fvg: fig.add_shape(type="rect", x0=f['start'], x1=df_chart.index[-1], y0=f['bottom'], y1=f['top'], fillcolor="rgba(0,255,0,0.2)", line_width=0, xref='x', yref='y', row=1, col=1)
        for f in bear_fvg: fig.add_shape(type="rect", x0=f['start'], x1=df_chart.index[-1], y0=f['bottom'], y1=f['top'], fillcolor="rgba(255,0,0,0.15)", line_width=0, xref='x', yref='y', row=1, col=1)
    if show_zigzag and pivots:
        px = [p['idx'] for p in pivots]; py = [p['val'] for p in pivots]
        fig.add_trace(go.Scatter(x=px, y=py, mode='lines+markers', name='ZigZag', line=dict(color='orange', width=2), marker_size=4), row=1, col=1)
        for p in pivots[-10:]:
            if 'label' in p:
                label_clr = '#00FF00' if 'H' in p['label'] and p['type'] == 'high' else 'red'
                fig.add_annotation(x=p['idx'], y=p['val'], text=p['label'], showarrow=False, font=dict(color=label_clr, size=10), yshift=15 if p['type']=='high' else -15, row=1, col=1)
    
    if show_orders:
        if st.session_state.positions:
            for pos in st.session_state.positions:
                if pos['symbol'] == symbol:
                    if pos.get('tp', 0) > 0: fig.add_hline(y=pos['tp'], line_dash="dashdot", line_color="#00FF00", annotation_text=f"止盈")
                    if pos.get('sl', 0) > 0: fig.add_hline(y=pos['sl'], line_dash="dashdot", line_color="#FF0000", annotation_text=f"止損")
        if st.session_state.pending_orders:
            for ord in st.session_state.pending_orders:
                if ord['symbol'] == symbol: fig.add_hline(y=ord['entry'], line_dash="dash", line_color="orange", annotation_text=f"掛單")

    colors = ['#00C853' if c >= o else '#FF3D00' for c, o in zip(df_chart['Close'], df_chart['Open'])]
    fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], name='Vol', marker_color=colors), row=2, col=1)

    if indicator_mode == "RSI":
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], name='RSI', line=dict(width=2, color='violet')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1); fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    else: 
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MACD'], name='MACD', line=dict(width=1, color='cyan')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Signal'], name='Signal', line=dict(width=1, color='orange')), row=3, col=1)
        hist_colors = ['#00C853' if h >= 0 else '#FF3D00' for h in df_chart['Hist']]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Hist'], name='Hist', marker_color=hist_colors), row=3, col=1)

    fig.update_layout(template="plotly_dark", height=700, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, dragmode='pan')
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

    # --- Wallet & Panel ---
    st.divider()
    total_unrealized = 0; total_margin = 0
    if st.session_state.positions:
        for pos in st.session_state.positions:
            try:
                lp = get_current_price(pos['symbol'])
                if lp:
                    d = 1 if pos['type'] == 'Long' else -1
                    # Safe Float Conversion
                    margin = float(pos['margin'])
                    entry = float(pos['entry'])
                    lev = float(pos['lev'])
                    
                    u_pnl = margin * (((lp - entry) / entry) * lev * d)
                    total_unrealized += u_pnl
                    total_margin += margin
            except: continue

    equity = st.session_state.balance + total_margin + total_unrealized
    
    # Liquidation Check
    if equity <= 0 and st.session_state.positions:
        st.error("💀 帳戶爆倉！所有倉位已強制平倉。")
        st.session_state.positions = []
        st.session_state.pending_orders = []
        st.session_state.balance = 0
        save_data()
        st.rerun()

    c_w1, c_w2, c_w3 = st.columns(3)
    c_w1.metric("💰 權益 (Equity)", f"${equity:,.2f}")
    c_w2.metric("💵 可用餘額", f"${st.session_state.balance:,.2f}")
    c_w3.metric("🔥 盈虧 (PnL)", f"${total_unrealized:+.2f} U", delta_color="normal")

    tab_trade, tab_ord, tab_hist = st.tabs(["🚀 下單", "📋 委託", "📜 歷史"])
    
    with tab_trade:
        order_type = st.radio("類型", ["⚡ 市價", "⏱️ 掛單"], horizontal=True, label_visibility="collapsed")
        c1, c2 = st.columns(2)
        side = c1.selectbox("方向", ["🟢 做多", "🔴 做空"], index=0 if mtf_res['score']>0 else 1)
        lev = c2.number_input("槓桿", min_value=1, max_value=200, value=20)
        
        def_p = curr_price
        if "掛單" in order_type and st.session_state.ai_entry > 0: def_p = st.session_state.ai_entry
        entry_p = st.number_input("掛單價格", value=float(def_p), format="%.6f") if "掛單" in order_type else st.caption(f"市價約: {fmt_price(curr_price)}") or curr_price
        
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        if c_p1.button("25%", use_container_width=True, on_click=set_amt, args=(0.25,)): pass
        if c_p2.button("50%", use_container_width=True, on_click=set_amt, args=(0.50,)): pass
        if c_p3.button("75%", use_container_width=True, on_click=set_amt, args=(0.75,)): pass
        if c_p4.button("Max", use_container_width=True, on_click=set_amt, args=(1.00,)): pass
        
        amt = st.number_input("本金 (U)", value=float(st.session_state.trade_amt_input_val), min_value=1.0, key="trade_amt_input_val_widget", on_change=lambda: st.session_state.update({"trade_amt_input_val": st.session_state.trade_amt_input_val_widget}))
        
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
            try:
                live = curr_price if pos['symbol'] == symbol else get_current_price(pos['symbol'])
                if live:
                    d = 1 if pos['type'] == 'Long' else -1
                    # 強制轉型
                    margin = float(pos['margin'])
                    entry = float(pos['entry'])
                    lev = float(pos['lev'])
                    
                    u_pnl = margin * (((live - entry) / entry) * lev * d)
                    pnl_pct = (((live - entry) / entry) * lev * d) * 100
                    
                    liq = entry*(1 - 1/lev) if pos['type']=='Long' else entry*(1 + 1/lev)
                    if (pos['type']=='Long' and live<=liq) or (pos['type']=='Short' and live>=liq): close_position(i, 100, "💀 爆倉", live); st.rerun()
                    elif pos.get('tp',0)>0 and ((pos['type']=='Long' and live>=pos['tp']) or (pos['type']=='Short' and live<=pos['tp'])): close_position(i, 100, "🎯 止盈", live); st.rerun()
                    elif pos.get('sl',0)>0 and ((pos['type']=='Long' and live<=pos['sl']) or (pos['type']=='Short' and live>=pos['sl'])): close_position(i, 100, "🛡️ 止損", live); st.rerun()

                    col_h1, col_h2 = st.columns([4, 1])
                    col_h1.markdown(f"**#{i+1} {pos['symbol']}**")
                    if col_h2.button(f"🔍", key=f"jump_{i}"): st.session_state.chart_symbol = pos['symbol']; st.rerun()

                    clr = "#00C853" if u_pnl >= 0 else "#FF3D00"
                    icon = "🟢" if pos['type'] == 'Long' else "🔴"
                    st.markdown(f"""<div style="background-color: #262730; padding: 12px; border-radius: 8px; border-left: 5px solid {clr}; margin-bottom: 8px;"><div style="display: flex; justify-content: space-between; font-size: 13px; color: #ccc;"><span>{icon} {pos['type']} x{lev:.0f} <span style="color:#888;">(本金: {margin:.0f} U)</span></span><span>🕒 {pos.get('time','--')}</span></div><div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 5px;"><div><div style="font-size: 12px; color: #aaa;">未結盈虧 (U)</div><div style="font-size: 18px; font-weight: bold; color: {clr};">{u_pnl:+.2f} U</div></div><div style="text-align: right;"><div style="font-size: 12px; color: #aaa;">回報率 (%)</div><div style="font-size: 18px; font-weight: bold; color: {clr};">{pnl_pct:+.2f}%</div></div></div><div style="margin-top: 8px; font-size: 12px; color: #888; display: flex; justify-content: space-between;"><span>開: {fmt_price(entry)}</span><span>現: {fmt_price(live)}</span></div></div>""", unsafe_allow_html=True)
                    if st.button("⚙️ 管理 / 平倉", key=f"m_{i}", use_container_width=True): manage_position_dialog(i, pos, live)
                    st.markdown("---")
            except Exception: continue

else: st.error(f"❌ 無法讀取 {symbol}")
