import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from datetime import datetime

# --- Page setup ---
st.set_page_config(page_title="全方位戰情室 AI (Cloud版)", layout="wide")
st.title("🏦 全方位戰情室 AI (v36.0 全功能交易所版)")

# --- Session init ---
if 'balance' not in st.session_state: st.session_state.balance = 10000.0
if 'positions' not in st.session_state: st.session_state.positions = []
if 'pending_orders' not in st.session_state: st.session_state.pending_orders = [] # 新增掛單儲存
if 'history' not in st.session_state: st.session_state.history = []
if 'chart_symbol' not in st.session_state: st.session_state.chart_symbol = "BTC-USD"
if 'market' not in st.session_state: st.session_state.market = "加密貨幣"

# --- Helpers ---
def fmt_price(val):
    if val is None: return "N/A"
    try:
        valf = float(val)
    except:
        return str(val)
    if valf < 0.01: return f"${valf:.6f}"
    elif valf < 20: return f"${valf:.4f}"
    else: return f"${valf:,.2f}"

def get_current_price(sym):
    try:
        ticker = yf.Ticker(sym)
        if hasattr(ticker, 'fast_info') and getattr(ticker.fast_info, 'last_price', None):
            return float(ticker.fast_info.last_price)
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        return None
    return None

def calc_price_from_roe(entry, leverage, direction_str, roe_pct):
    if entry == 0: return 0.0
    direction = 1 if "Long" in direction_str or "做多" in direction_str else -1
    try:
        price = entry * (1 + (roe_pct / 100) / (leverage * direction))
        return float(price)
    except:
        return 0.0

def calc_roe_from_price(entry, leverage, direction_str, target_price):
    if entry == 0: return 0.0
    direction = 1 if "Long" in direction_str or "做多" in direction_str else -1
    try:
        roe = ((target_price - entry) / entry) * leverage * direction * 100
        return float(roe)
    except:
        return 0.0

# --- Sidebar UI: market + symbol selection ---
st.sidebar.header("🎯 市場與標的")

market = st.sidebar.radio("選擇市場", ["加密貨幣", "美股", "台股"], index=0, key="market_radio")
st.session_state.market = market

crypto_list = ["BTC", "ETH", "SOL", "BNB", "DOGE", "XRP", "ADA", "AVAX"]
us_stock_list = ["AAPL", "NVDA", "TSLA", "MSFT", "META", "AMZN", "GOOGL", "AMD"]
tw_stock_dict = {
    "2330 台積電": "2330",
    "2454 聯發科": "2454",
    "2317 鴻海": "2317",
    "2303 聯電": "2303",
    "2603 長榮": "2603",
    "2609 陽明": "2609",
    "2615 萬海": "2615",
    "0050 元大台灣50": "0050",
    "00878 國泰永續高股息": "00878"
}

raw_symbol = "" 
if market == "加密貨幣":
    selected_item = st.sidebar.selectbox("🔥 常見加密貨幣", crypto_list)
    raw_symbol = selected_item
elif market == "美股":
    selected_item = st.sidebar.selectbox("🇺🇸 常見美股", us_stock_list)
    raw_symbol = selected_item
else: 
    tw_display_list = list(tw_stock_dict.keys())
    selected_item = st.sidebar.selectbox("🇹🇼 常見台股", tw_display_list)
    raw_symbol = tw_stock_dict[selected_item]

search_input = st.sidebar.text_input("🔍 快速搜尋 / 代碼輸入", placeholder="例如: 2330 或 BTC")
if search_input.strip():
    raw_symbol = search_input.strip().upper()

final_symbol = raw_symbol
if market == "加密貨幣":
    if "USD" not in final_symbol and "-" not in final_symbol:
        final_symbol += "-USD"
elif market == "台股":
    if final_symbol.isdigit() or (len(final_symbol) == 4 and final_symbol.isdigit()):
        final_symbol += ".TW"
    elif not final_symbol.endswith(".TW") and not final_symbol.endswith(".TWO"):
        final_symbol += ".TW"

st.session_state.chart_symbol = final_symbol
symbol = st.session_state.chart_symbol 
st.sidebar.success(f"目前交易標的：{symbol}")

interval_ui = st.sidebar.radio("K 線週期", ["15分鐘", "1小時", "4小時", "日線"], index=3)
st.sidebar.markdown("---")
st.sidebar.markdown("### 👁️ 視覺化開關")
show_six = st.sidebar.checkbox("顯示 六道乾坤帶 (EMA)", value=True)
show_zigzag = st.sidebar.checkbox("顯示 ZigZag 結構", value=True)
show_fvg = st.sidebar.checkbox("顯示 FVG 缺口", value=True)
show_fib = st.sidebar.checkbox("顯示 Fib 止盈", value=True)
show_div = st.sidebar.checkbox("顯示 RSI 背離", value=True)
show_orders = st.sidebar.checkbox("顯示 掛單 (TP/SL)", value=True)

if st.sidebar.button("🔄 強制刷新盤勢"):
    try: st.cache_data.clear()
    except: pass
    st.rerun()

# --- Data fetch params ---
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
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        rs = gain.rolling(14).mean() / (loss.rolling(14).mean().replace(0, np.nan))
        df['RSI'] = 100 - (100 / (1 + rs))
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA60'] = df['Close'].ewm(span=60, adjust=False).mean()
        df['EMA120'] = df['Close'].ewm(span=120, adjust=False).mean()
        df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(14).mean()
        return df.dropna(how='all')
    except: return None

# --- Indicators ---
def calculate_zigzag(df, depth=12):
    try:
        df = df.copy()
        df['max_roll'] = df['High'].rolling(window=depth, center=True).max()
        df['min_roll'] = df['Low'].rolling(window=depth, center=True).min()
        pivots = []
        last_type = None
        for i in range(len(df)):
            try:
                if not np.isnan(df['max_roll'].iloc[i]) and df['High'].iloc[i] == df['max_roll'].iloc[i]:
                    if last_type != 'high':
                        pivots.append({'idx': df.index[i], 'val': float(df['High'].iloc[i]), 'type': 'high'}); last_type='high'
                    elif pivots and df['High'].iloc[i] > pivots[-1]['val']:
                        pivots[-1] = {'idx': df.index[i], 'val': float(df['High'].iloc[i]), 'type': 'high'}
                elif not np.isnan(df['min_roll'].iloc[i]) and df['Low'].iloc[i] == df['min_roll'].iloc[i]:
                    if last_type != 'low':
                        pivots.append({'idx': df.index[i], 'val': float(df['Low'].iloc[i]), 'type': 'low'}); last_type='low'
                    elif pivots and df['Low'].iloc[i] < pivots[-1]['val']:
                        pivots[-1] = {'idx': df.index[i], 'val': float(df['Low'].iloc[i]), 'type': 'low'}
            except: continue
        return pivots
    except: return []

def calculate_fvg(df):
    try:
        bull, bear = [], []
        h = df['High'].values; l = df['Low'].values; c = df['Close'].values; t = df.index
        start_idx = max(2, len(df)-300)
        for i in range(start_idx, len(df)):
            if l[i] > h[i-2] and c[i-1] > h[i-2]:
                bull.append({'start': t[i-2], 'top': float(l[i]), 'bottom': float(h[i-2]), 'active': True})
            if h[i] < l[i-2] and c[i-1] < l[i-2]:
                bear.append({'start': t[i-2], 'top': float(l[i-2]), 'bottom': float(h[i]), 'active': True})
            for f in bull:
                if f['active'] and l[i] < f['top']: f['active'] = False
            for f in bear:
                if f['active'] and h[i] > f['bottom']: f['active'] = False
        return [f for f in bull if f['active']], [f for f in bear if f['active']]
    except: return [], []

def detect_div(df):
    try:
        rsi = df['RSI'].values; close = df['Close'].values
        highs = argrelextrema(rsi, np.greater, order=5)[0]; lows = argrelextrema(rsi, np.less, order=5)[0]
        bull, bear = [], []
        if len(lows) >= 2:
            for i in range(len(lows)-1):
                curr, prev = lows[i+1], lows[i]
                if close[curr] < close[prev] and rsi[curr] > rsi[prev] and rsi[curr] < 50:
                    bull.append(df.index[curr])
        if len(highs) >= 2:
            for i in range(len(highs)-1):
                curr, prev = highs[i+1], highs[i]
                if close[curr] > close[prev] and rsi[curr] < rsi[prev] and rsi[curr] > 50:
                    bear.append(df.index[curr])
        return bull, bear
    except: return [], []

def calculate_score_v17(pivots, last, df, bull_fvg, bear_fvg, bull_div, bear_div):
    score = 0; struct_txt = "盤整"
    try:
        if len(pivots) >= 4:
            vh = [p['val'] for p in pivots if p['type']=='high']; vl = [p['val'] for p in pivots if p['type']=='low']
            if len(vh) >= 2 and len(vl) >= 2:
                if vh[-1] > vh[-2] and vl[-1] > vl[-2]: score += 3; struct_txt="多頭 (+3)"
                elif vh[-1] < vh[-2] and vl[-1] < vl[-2]: score -= 3; struct_txt="空頭 (-3)"
    except: pass
    six_txt = "盤整"
    ema20, ema60, ema120 = last.get('EMA20', np.nan), last.get('EMA60', np.nan), last.get('EMA120', np.nan)
    if last['Close'] > ema20 > ema60 > ema120: score += 2; six_txt="順勢多 (+2)"
    elif last['Close'] < ema20 < ema60 < ema120: score -= 2; six_txt="順勢空 (-2)"
    elif last['Close'] > ema60: score += 1; six_txt="偏多 (+1)"
    elif last['Close'] < ema60: score -= 1; six_txt="偏空 (-1)"
    fvg_txt = "無"
    try:
        if bull_fvg and (last['Close'] - bull_fvg[-1]['top']) / last['Close'] < 0.02: score += 2; fvg_txt="支撐位 (+2)"
        elif bear_fvg and (bear_fvg[-1]['bottom'] - last['Close']) / last['Close'] < 0.02: score -= 2; fvg_txt="壓力位 (-2)"
    except: pass
    div_txt = "無"
    try:
        if bull_div and (df.index[-1] - bull_div[-1]).days < 3: score += 2; div_txt="底背離 (+2)"
        elif bear_div and (df.index[-1] - bear_div[-1]).days < 3: score -= 2; div_txt="頂背離 (-2)"
    except: pass
    rsi_txt = "中性"
    if last['RSI'] < 30: score += 1; rsi_txt="超賣 (+1)"
    elif last['RSI'] > 70: score -= 1; rsi_txt="超買 (-1)"
    return score, struct_txt, six_txt, fvg_txt, div_txt, rsi_txt

def generate_ai_report(symbol, price, score, struct, six, fvg, div, rsi_txt, buy_sl, sell_sl, tp1, tp2, entry_zone, risk_warning):
    report = f"**【市場掃描】** {symbol} 現價 **{fmt_price(price)}**。\n"
    abs_score = abs(score)
    direction = "做多" if score > 0 else "做空"
    color_emoji = "🟢" if score > 0 else "🔴"
    if risk_warning: report += f"⚠️ **風險提示**：{risk_warning}\n\n"
    elif abs_score >= 8: report += f"🔥 **強力{direction}訊號 (評分: {score}/10)**！\n\n"
    elif abs_score >= 5: report += f"{color_emoji} **偏向{direction} (評分: {score}/10)**。\n\n"
    else: report += f"⚖️ **盤整觀望 (評分: {score}/10)**。\n\n"
    report += "**【交易計畫】**"
    if risk_warning and "破" in risk_warning: report += f"\n⛔ 結構已破壞，暫無交易建議。"
    elif score >= 0: report += f"\n🛒 **建議入場**: **{entry_zone}**\n🎯 **止盈 TP1**: **{fmt_price(tp1)}**\n🛡️ **止損 SL**: **{fmt_price(buy_sl)}**"
    else: report += f"\n🛒 **建議空點**: **{entry_zone}**\n🎯 **止盈 TP1**: **{fmt_price(tp1)}**\n🛡️ **止損 SL**: **{fmt_price(sell_sl)}**"
    return report

# --- Position & Order Management ---
def close_position(pos_index, percentage=100, reason="手動平倉", exit_price=None):
    if pos_index >= len(st.session_state.positions): return
    pos = st.session_state.positions[pos_index]
    if exit_price is None:
        exit_price = get_current_price(pos['symbol'])
        if exit_price is None: exit_price = pos['entry']
    
    close_margin = pos['margin'] * (percentage / 100)
    direction = 1 if pos['type'] == 'Long' else -1
    try: pnl_pct = ((exit_price - pos['entry']) / pos['entry']) * pos['lev'] * direction * 100
    except: pnl_pct = 0
    pnl_usdt = close_margin * (pnl_pct / 100)
    
    st.session_state.balance += (close_margin + pnl_usdt)
    st.session_state.history.append({
        "時間": datetime.now().strftime("%m-%d %H:%M"),
        "幣種": pos['symbol'],
        "動作": f"平倉 {percentage}%",
        "入場": pos['entry'],
        "出場": exit_price,
        "損益(U)": round(pnl_usdt, 2),
        "獲利%": round(pnl_pct, 2),
        "原因": reason
    })
    
    if percentage == 100:
        st.session_state.positions.pop(pos_index)
        st.toast(f"✅ {pos['symbol']} 已全部平倉，獲利 {pnl_usdt:.2f} U")
    else:
        st.session_state.positions[pos_index]['margin'] -= close_margin
        st.toast(f"✅ {pos['symbol']} 部分平倉 ({percentage}%)，入袋 {pnl_usdt:.2f} U")
    st.rerun()

def cancel_order(pos_index, order_type):
    if pos_index < len(st.session_state.positions):
        if order_type == 'TP':
            st.session_state.positions[pos_index]['tp'] = 0.0
            st.session_state.positions[pos_index]['tp_ratio'] = 0
        elif order_type == 'SL':
            st.session_state.positions[pos_index]['sl'] = 0.0
            st.session_state.positions[pos_index]['sl_ratio'] = 0
        st.toast(f"🗑️ 已撤銷 {order_type} 委託單")
        st.rerun()

def cancel_pending_order(index):
    if index < len(st.session_state.pending_orders):
        order = st.session_state.pending_orders.pop(index)
        st.session_state.balance += order['margin'] # 退還本金
        st.toast(f"🗑️ 已撤銷掛單: {order['symbol']} @ {fmt_price(order['entry'])}")
        st.rerun()

def check_pending_orders(symbol, curr_price):
    # 檢查掛單是否成交
    triggered_indexes = []
    for i, order in enumerate(st.session_state.pending_orders):
        if order['symbol'] == symbol:
            is_filled = False
            # Limit Order Logic: 
            # Long: Fill if Price <= Entry
            # Short: Fill if Price >= Entry
            if order['type'] == 'Long' and curr_price <= order['entry']:
                is_filled = True
            elif order['type'] == 'Short' and curr_price >= order['entry']:
                is_filled = True
            
            if is_filled:
                triggered_indexes.append(i)
                # 轉為持倉
                new_pos = {
                    "symbol": order['symbol'], "type": order['type'],
                    "entry": order['entry'], # 用掛單價成交 (模擬)
                    "lev": order['lev'], "margin": order['margin'],
                    "tp": order['tp'], "sl": order['sl'],
                    "time": datetime.now().strftime('%m-%d %H:%M'),
                    "tp_ratio": 100, "sl_ratio": 100
                }
                st.session_state.positions.append(new_pos)
                st.toast(f"⚡ 掛單成交！{order['symbol']} {order['type']} @ {fmt_price(order['entry'])}")
    
    # 移除已成交的掛單 (反向遍歷以免索引跑掉)
    for i in sorted(triggered_indexes, reverse=True):
        st.session_state.pending_orders.pop(i)

# --- Main ---
df = get_data(symbol, period, interval)

if df is not None and not df.empty:
    last = df.iloc[-1]; curr_price = float(last['Close'])
    
    # 每次刷新都要檢查掛單
    check_pending_orders(symbol, curr_price)

    # Sidebar wallet/positions
    st.sidebar.markdown("---")
    with st.sidebar.expander("🏦 我的錢包與持倉", expanded=True):
        st.metric("💰 總資產 (USDT)", f"${st.session_state.balance:,.2f}")
        
        tab_pos, tab_ord, tab_hist = st.tabs(["🔥 持倉", "📋 委託單", "📜 歷史"])
        
        # --- Tab 1: 持倉列表 ---
        with tab_pos:
            if st.session_state.positions:
                for i, pos in list(enumerate(st.session_state.positions)):
                    live_price = curr_price if pos['symbol'] == symbol else get_current_price(pos['symbol'])
                    if live_price:
                        direction = 1 if pos['type'] == 'Long' else -1
                        try: pnl_pct = ((live_price - pos['entry']) / pos['entry']) * pos['lev'] * direction * 100
                        except: pnl_pct = 0
                        pnl_usdt = pos['margin'] * (pnl_pct / 100)
                        if pos['type'] == 'Long': liq = pos['entry'] * (1 - 1/pos['lev'])
                        else: liq = pos['entry'] * (1 + 1/pos['lev'])

                        # UI Card
                        c_title, c_jump = st.columns([4, 1])
                        c_title.markdown(f"**#{i+1} {pos['symbol']}**")
                        if pos['symbol'] != symbol and c_jump.button("🔍", key=f"jump_{i}"):
                            st.session_state.chart_symbol = pos['symbol']; st.rerun()

                        pnl_color = "#00C853" if pnl_usdt >= 0 else "#FF3D00"
                        side_icon = "🟢" if pos['type'] == 'Long' else "🔴"
                        open_time = pos.get('time', '剛剛') 

                        st.markdown(f"""
                        <div style="background-color: #262730; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid {pnl_color};">
                            <div style="display: flex; justify-content: space-between; font-size: 12px; color: #aaaaaa; margin-bottom: 4px;">
                                <span>{side_icon} {pos['type']} x{pos['lev']}</span>
                                <span>🕒 {open_time}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                                <div>
                                    <div style="font-size: 12px; color: #aaaaaa;">未結盈虧 (U)</div>
                                    <div style="font-size: 16px; font-weight: bold; color: {pnl_color};">{pnl_usdt:+.2f} U</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 12px; color: #aaaaaa;">回報率 (%)</div>
                                    <div style="font-size: 16px; font-weight: bold; color: {pnl_color};">{pnl_pct:+.2f}%</div>
                                </div>
                            </div>
                            <div style="margin-top: 8px; font-size: 11px; color: #cccccc; display: flex; justify-content: space-between;">
                                <span>開倉: {fmt_price(pos['entry'])}</span>
                                <span>現價: {fmt_price(live_price)}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Triggers
                        reason = None
                        trigger_ratio = 100
                        if (pos['type']=='Long' and live_price <= liq) or (pos['type']=='Short' and live_price >= liq):
                            reason="💀 爆倉"
                        elif pos.
