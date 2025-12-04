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
st.markdown("### 🏦 全方位戰情室 AI (v60.0 終極架構版)")

# --- [核心修復] JSON 序列化編碼器 (解決存檔失敗) ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# --- Persistence System ---
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
            # 使用 NpEncoder 解決 numpy 格式問題
            json.dump(data, f, cls=NpEncoder) 
    except Exception as e:
        st.error(f"存檔失敗: {e}")

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                st.session_state.balance = float(data.get("balance", 10000.0))
                
                # 強制清洗數據，轉為 float，防止報錯
                clean_pos = []
                for p in data.get("positions", []):
                    try:
                        p['entry'] = float(p['entry'])
                        p['margin'] = float(p['margin'])
                        p['lev'] = float(p['lev'])
                        p['tp'] = float(p.get('tp', 0))
                        p['sl'] = float(p.get('sl', 0))
                        clean_pos.append(p)
                    except: continue
                st.session_state.positions = clean_pos
                
                # 清洗掛單
                clean_ords = []
                for o in data.get("pending_orders", []):
                    try:
                        o['entry'] = float(o['entry'])
                        o['margin'] = float(o['margin'])
                        clean_ords.append(o)
                    except: continue
                st.session_state.pending_orders = clean_ords
                
                st.session_state.history = data.get("history", [])
        except: pass

# --- State Init (保證變數存在) ---
if 'init_done' not in st.session_state:
    st.session_state.balance = 10000.0
    st.session_state.positions = []
    st.session_state.pending_orders = []
    st.session_state.history = []
    st.session_state.trade_amt_box = 1000.0
    st.session_state.chart_symbol = "BTC-USD"
    st.session_state.market = "加密貨幣"
    load_data()
    st.session_state.init_done = True

# --- Callbacks ---
def set_amt(ratio):
    # 直接修改輸入框綁定的 key
    val = float(st.session_state.balance * ratio)
    if val < 0: val = 0.0
    st.session_state.trade_amt_box = val

def update_symbol():
    # 當側邊欄選擇改變時觸發
    pass

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
    st.markdown(f"**{pos['symbol']}** ({pos['type']} x{pos['lev']:.0f})")
    st.caption(f"本金: {pos['margin']:.2f} U | 開倉: {fmt_price(pos['entry'])}")
    
    tab_close, tab_tpsl = st.tabs(["平倉", "止盈止損"])
    with tab_close:
        ratio = st.radio("Ratio", [25,50,75,100], 3, horizontal=True, key=f"d_r_{i}", format_func=lambda x:f"{x}%")
        if st.button("確認平倉", key=f"d_btn_close_{i}", type="primary", use_container_width=True):
            close_position(i, ratio, "手動", current_price); st.rerun()
    with tab_tpsl:
        current_tp = float(pos.get('tp', 0)); current_sl = float(pos.get('sl', 0))
        input_mode = st.radio("單位", ["價格", "ROE %"], horizontal=True, key=f"d_mode_{i}")
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

# 選擇列表
target_list = crypto_list if market == "加密貨幣" else (us_stock_list if market == "美股" else list(tw_stock_dict.keys()))

# [修復] 使用 temp_symbol 加上 button，避免 session 衝突
select_val = st.sidebar.selectbox("快速選擇", target_list)
search_val = st.sidebar.text_input("代碼搜尋 (例如 2330)")

# 決定最終標的字串
raw_symbol = search_val.strip().upper() if search_val.strip() else select_val
if market == "台股" and raw_symbol in tw_stock_dict: raw_symbol = tw_stock_dict[raw_symbol]

final_symbol = raw_symbol
if market == "加密貨幣":
    if "USD" not in final_symbol and "-" not in final_symbol: final_symbol += "-USD"
elif market == "台股":
    if final_symbol.isdigit(): final_symbol += ".TW"
    elif not final_symbol.endswith(".TW"): final_symbol += ".TW"

# 側邊欄更新機制
if st.sidebar.button("🚀 載入 K 線"):
    st.session_state.chart_symbol = final_symbol
    st.rerun()

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
    # 安全讀取 balance
    bal = st.session_state.get('balance', 0.0)
    st.caption(f"可用餘額: ${bal:,.2f}")
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
    if st.button("🧨 強制清空數據"): 
        if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
        st.session_state.positions = []
        st.rerun()

def get_params(ui_selection):
    if "15分鐘" in ui_selection: return "5d", "15m"
    elif "1小時" in ui_selection: return "1mo", "1h"
    elif "4小時" in ui_selection: return "6mo", "1h"
    else: return "2y", "1d"
period, interval = get_params(interval_ui)

@st.cache_data(ttl=60)
def get_data_safe(symbol, period, interval):
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
        df['EMA20'] = df['Close'].ewm(span=20).mean(); df['EMA60'] = df['Close'].ewm(span=60).mean(); df['EMA120'] = df['Close'].ewm(span=120).mean()
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

# --- AI Analysis ---
def run_ai_analysis(df, pivots, fvg_bull, fvg_bear):
    last = df.iloc[-1]
    close = last['Close']
    atr = last.get('ATR', close * 0.02)
    score = 0; reasons = []
    
    ema20, ema60 = last.get('EMA20'), last.get('EMA60')
    if close > ema20 > ema60: score += 3; reasons.append("均線多頭")
    elif close < ema20 < ema60: score -= 3; reasons.append("均線空頭")
    
    if pivots and len(pivots)>=2:
        if pivots[-1]['type'] == 'high' and 'H' in pivots[-1].get('label',''): score+=2; reasons.append("結構創高")
    
    if last['RSI'] < 30: score += 2; reasons.append("RSI 超賣")
    elif last['RSI'] > 70: score -= 2; reasons.append("RSI 超買")

    direction = "做多 (Long)" if score > 0 else "做空 (Short)"
    conf = min(abs(score), 10)
    
    if score > 0:
        entry = ema20 if ema20 < close else close-atr
        sl = entry - 1.5*atr; tp = entry + 2.5*atr
    else:
        entry = ema20 if ema20 > close else close+atr
        sl = entry + 1.5*atr; tp = entry - 2.5*atr
    return {"score": score, "dir": direction, "conf": conf, "entry": entry, "tp": tp, "sl": sl, "reasons": ", ".join(reasons)}

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
df = get_data_safe(symbol, period, interval)

if df is not None and not df.empty:
    last = df.iloc[-1]; curr_price = float(last['Close'])

    # [新增] 頂部行情
    st.markdown(f"<h1 style='text-align:center'>{symbol} <span style='color:{'#00C853' if df['Close'].iloc[-1]>=df['Open'].iloc[-1] else '#FF3D00'}'>${curr_price:,.2f}</span></h1>", unsafe_allow_html=True)

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

    # Analysis
    pivots = calculate_zigzag(df); bull_fvg, bear_fvg = calculate_fvg(df)
    ai_res = run_ai_analysis(df, pivots, bull_fvg, bear_fvg)

    c1, c2, c3 = st.columns([1.5, 1, 1.5])
    with c1:
        clr = "green" if ai_res['score']>0 else "red"
        st.markdown(f"#### AI 建議: :{clr}[{ai_res['dir']}]")
        st.caption(f"理由: {ai_res['reasons']}")
    with c2: st.metric("建議入場", fmt_price(ai_res['entry']))
    with c3: st.metric("目標止盈", fmt_price(ai_res['tp'])); st.metric("防守止損", fmt_price(ai_res['sl']))

    # Chart
    indicator_mode = st.radio("副圖", ["RSI", "MACD"], horizontal=True, label_visibility="collapsed")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25], subplot_titles=("價格", "成交量", indicator_mode))
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    if show_six:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20', line=dict(width=1, color='yellow')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA60'], name='EMA60', line=dict(width=1, color='cyan')), row=1, col=1)
    if show_fvg:
        for f in bull_fvg: fig.add_shape(type="rect", x0=f['start'], x1=df.index[-1], y0=f['bottom'], y1=f['top'], fillcolor="rgba(0,255,0,0.2)", line_width=0, xref='x', yref='y', row=1, col=1)
        for f in bear_fvg: fig.add_shape(type="rect", x0=f['start'], x1=df.index[-1], y0=f['bottom'], y1=f['top'], fillcolor="rgba(255,0,0,0.15)", line_width=0, xref='x', yref='y', row=1, col=1)
    if show_zigzag and pivots:
        px = [p['idx'] for p in pivots]; py = [p['val'] for p in pivots]
        fig.add_trace(go.Scatter(x=px, y=py, mode='lines+markers', name='ZigZag', line=dict(color='orange', width=2), marker_size=4), row=1, col=1)
        for p in pivots[-10:]:
            if 'label' in p:
                l_clr = '#00FF00' if 'H' in p['label'] and p['type'] == 'high' else 'red'
                fig.add_annotation(x=p['idx'], y=p['val'], text=p['label'], showarrow=False, font=dict(color=l_clr, size=10), yshift=15 if p['type']=='high' else -15, row=1, col=1)
    
    if show_orders:
        if st.session_state.positions:
            for pos in st.session_state.positions:
                if pos['symbol'] == symbol:
                    if pos.get('tp', 0) > 0: fig.add_hline(y=pos['tp'], line_dash="dashdot", line_color="#00FF00", annotation_text=f"止盈")
                    if pos.get('sl', 0) > 0: fig.add_hline(y=pos['sl'], line_dash="dashdot", line_color="#FF0000", annotation_text=f"止損")
        if st.session_state.pending_orders:
            for ord in st.session_state.pending_orders:
                if ord['symbol'] == symbol: fig.add_hline(y=ord['entry'], line_dash="dash", line_color="orange", annotation_text=f"掛單")

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

    # --- Wallet ---
    st.divider()
    total_unrealized = 0; total_margin = 0
    for pos in st.session_state.positions:
        try:
            lp = get_current_price(pos['symbol'])
            if lp:
                d = 1 if pos['type'] == 'Long' else -1
                u_pnl = float(pos['margin']) * (((lp - float(pos['entry'])) / float(pos['entry'])) * float(pos['lev']) * d)
                total_unrealized += u_pnl
                total_margin += float(pos['margin'])
        except: continue

    equity = st.session_state.balance + total_margin + total_unrealized
    if equity <= 0 and st.session_state.positions:
        st.error("💀 帳戶爆倉！")
        st.session_state.positions = []; st.session_state.pending_orders = []; st.session_state.balance = 0
        save_data(); st.rerun()

    c_w1, c_w2, c_w3 = st.columns(3)
    c_w1.metric("💰 權益", f"${equity:,.2f}")
    c_w2.metric("💵 餘額", f"${st.session_state.balance:,.2f}")
    c_w3.metric("🔥 盈虧", f"${total_unrealized:+.2f} U", delta_color="normal")

    # --- Trade ---
    tab_trade, tab_ord, tab_hist = st.tabs(["🚀 下單", "📋 委託", "📜 歷史"])
    
    with tab_trade:
        order_type = st.radio("類型", ["⚡ 市價", "⏱️ 掛單"], horizontal=True, label_visibility="collapsed")
        c1, c2 = st.columns(2)
        side = c1.selectbox("方向", ["🟢 做多", "🔴 做空"], index=0 if ai_res['dir']=="做多 (Long)" else 1)
        lev = c2.number_input("槓桿", 1, 200, 20)
        
        def_p = curr_price
        if "掛單" in order_type and st.session_state.ai_entry > 0: def_p = st.session_state.ai_entry
        entry_p = st.number_input("掛單價格", value=float(def_p), format="%.6f") if "掛單" in order_type else st.caption(f"市價約: {fmt_price(curr_price)}") or curr_price
        
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        if c_p1.button("25%", use_container_width=True, on_click=set_amt, args=(0.25,)): pass
        if c_p2.button("50%", use_container_width=True, on_click=set_amt, args=(0.50,)): pass
        if c_p3.button("75%", use_container_width=True, on_click=set_amt, args=(0.75,)): pass
        if c_p4.button("Max", use_container_width=True, on_click=set_amt, args=(1.00,)): pass
        
        amt = st.number_input("本金 (U)", value=float(st.session_state.trade_amt_box), min_value=1.0, key="trade_amt_box")
        
        with st.expander("止盈止損"):
            new_tp = st.number_input("止盈", value=float(st.session_state.ai_tp))
            new_sl = st.number_input("止損", value=float(st.session_state.ai_sl))
            
        if st.button("確認下單", type="primary", use_container_width=True):
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
                    margin = float(pos['margin']); entry = float(pos['entry']); lev = float(pos['lev'])
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
            except: continue

else: st.error(f"❌ 無法讀取 {symbol}")
