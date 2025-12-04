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
st.title("🏦 全方位戰情室 AI (v35.0 高階委託版)")

# --- Session init ---
if 'balance' not in st.session_state: st.session_state.balance = 10000.0
if 'positions' not in st.session_state: st.session_state.positions = []
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
    """
    根據 ROE% 反推價格
    Long: Price = Entry * (1 + (ROE%/100)/Lev)
    Short: Price = Entry * (1 - (ROE%/100)/Lev)
    """
    if entry == 0: return 0.0
    direction = 1 if "Long" in direction_str or "做多" in direction_str else -1
    # ROE = (Price - Entry)/Entry * Lev * Dir
    # Price = Entry * (1 + ROE/(Lev*Dir))
    # 這裡 roe_pct 傳入如 20.0 代表 20%
    try:
        price = entry * (1 + (roe_pct / 100) / (leverage * direction))
        return float(price)
    except:
        return 0.0

def calc_roe_from_price(entry, leverage, direction_str, target_price):
    """
    根據價格反推 ROE%
    """
    if entry == 0: return 0.0
    direction = 1 if "Long" in direction_str or "做多" in direction_str else -1
    try:
        roe = ((target_price - entry) / entry) * leverage * direction * 100
        return float(roe)
    except:
        return 0.0

# --- Sidebar UI: market + symbol selection ---
st.sidebar.header("🎯 市場與標的")

# 1. 市場選擇
market = st.sidebar.radio("選擇市場", ["加密貨幣", "美股", "台股"], index=0, key="market_radio")
st.session_state.market = market

# 2. 定義常見標的資料庫
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

# 3. 處理下拉選單邏輯
raw_symbol = "" 

if market == "加密貨幣":
    selected_item = st.sidebar.selectbox("🔥 常見加密貨幣", crypto_list)
    raw_symbol = selected_item
elif market == "美股":
    selected_item = st.sidebar.selectbox("🇺🇸 常見美股", us_stock_list)
    raw_symbol = selected_item
else: # 台股
    tw_display_list = list(tw_stock_dict.keys())
    selected_item = st.sidebar.selectbox("🇹🇼 常見台股", tw_display_list)
    raw_symbol = tw_stock_dict[selected_item]

# 4. 處理手動搜尋
search_input = st.sidebar.text_input("🔍 快速搜尋 / 代碼輸入", placeholder="例如: 2330 或 BTC")
if search_input.strip():
    raw_symbol = search_input.strip().upper()

# 5. 格式化 Symbol
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

# interval controls
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
    try:
        st.cache_data.clear()
    except:
        pass
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
    except:
        return None

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
            except:
                continue
        return pivots
    except:
        return []

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
                if f['active'] and l[i] < f['top']:
                    f['active'] = False
            for f in bear:
                if f['active'] and h[i] > f['bottom']:
                    f['active'] = False
        return [f for f in bull if f['active']], [f for f in bear if f['active']]
    except:
        return [], []

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
    except:
        return [], []

def calculate_score_v17(pivots, last, df, bull_fvg, bear_fvg, bull_div, bear_div):
    score = 0; struct_txt = "盤整"
    try:
        if len(pivots) >= 4:
            vh = [p['val'] for p in pivots if p['type']=='high']; vl = [p['val'] for p in pivots if p['type']=='low']
            if len(vh) >= 2 and len(vl) >= 2:
                if vh[-1] > vh[-2] and vl[-1] > vl[-2]: score += 3; struct_txt="多頭 (+3)"
                elif vh[-1] < vh[-2] and vl[-1] < vl[-2]: score -= 3; struct_txt="空頭 (-3)"
    except:
        pass
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
    except:
        pass
    div_txt = "無"
    try:
        if bull_div and (df.index[-1] - bull_div[-1]).days < 3: score += 2; div_txt="底背離 (+2)"
        elif bear_div and (df.index[-1] - bear_div[-1]).days < 3: score -= 2; div_txt="頂背離 (-2)"
    except:
        pass
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
    if risk_warning and "破" in risk_warning:
        report += f"\n⛔ 結構已破壞，暫無交易建議。"
    elif score >= 0:
        report += f"\n🛒 **建議入場**: **{entry_zone}**\n🎯 **止盈 TP1**: **{fmt_price(tp1)}**\n🛡️ **止損 SL**: **{fmt_price(buy_sl)}**"
    else:
        report += f"\n🛒 **建議空點**: **{entry_zone}**\n🎯 **止盈 TP1**: **{fmt_price(tp1)}**\n🛡️ **止損 SL**: **{fmt_price(sell_sl)}**"
    return report

# --- Position close ---
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

# --- Main ---
df = get_data(symbol, period, interval)

if df is not None and not df.empty:
    last = df.iloc[-1]; curr_price = float(last['Close'])

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
                            st.session_state.chart_symbol = pos['symbol']
                            st.rerun()

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

                        # Check Triggers (TP/SL)
                        reason = None
                        trigger_ratio = 100 # default
                        
                        if (pos['type']=='Long' and live_price <= liq) or (pos['type']=='Short' and live_price >= liq):
                            reason="💀 爆倉"
                        elif pos.get('tp',0)>0 and ((pos['type']=='Long' and live_price >= pos['tp']) or (pos['type']=='Short' and live_price <= pos['tp'])):
                            reason="🎯 止盈觸發"
                            trigger_ratio = pos.get('tp_ratio', 100)
                            # Reset TP trigger after fire
                            st.session_state.positions[i]['tp'] = 0.0
                        elif pos.get('sl',0)>0 and ((pos['type']=='Long' and live_price <= pos['sl']) or (pos['type']=='Short' and live_price >= pos['sl'])):
                            reason="🛡️ 止損觸發"
                            trigger_ratio = pos.get('sl_ratio', 100)
                            # Reset SL trigger after fire
                            st.session_state.positions[i]['sl'] = 0.0
                            
                        if reason:
                            close_position(i, trigger_ratio, reason, live_price); break
                        
                        # Manual Close Ratio
                        ratio_key = f"ratio_{i}"
                        close_ratio = st.radio(
                            "比例", [25, 50, 75, 100], 
                            horizontal=True, 
                            index=3, 
                            key=ratio_key, 
                            label_visibility="collapsed",
                            format_func=lambda x: f"{x}%"
                        )
                        if st.button(f"⚡ 市價平倉 ({close_ratio}%)", key=f"btn_close_{i}", use_container_width=True):
                            final_ratio = st.session_state[ratio_key]
                            close_position(i, final_ratio, "手動市價", live_price); break
                        
                        st.divider()
            else:
                st.info("目前無持倉")
                
        # --- Tab 2: 委託單 (掛單管理) - 全新升級 ---
        with tab_ord:
            has_orders = False
            if st.session_state.positions:
                for i, pos in enumerate(st.session_state.positions):
                    # 初始化舊資料可能缺少的欄位
                    if 'tp_ratio' not in pos: st.session_state.positions[i]['tp_ratio'] = 100
                    if 'sl_ratio' not in pos: st.session_state.positions[i]['sl_ratio'] = 100
                    
                    # 顯示現有 TP
                    if pos.get('tp', 0) > 0:
                        has_orders = True
                        with st.expander(f"🎯 止盈 (TP) - {pos['symbol']}", expanded=True):
                            st.write(f"觸發價: **{fmt_price(pos['tp'])}** (平倉 {pos.get('tp_ratio', 100)}%)")
                            # 修改介面
                            input_mode = st.radio("修改方式", ["價格", "ROE %"], horizontal=True, key=f"mode_tp_{i}")
                            
                            c_val, c_ratio = st.columns([2, 1])
                            if input_mode == "價格":
                                new_val = c_val.number_input("價格", value=float(pos['tp']), key=f"mod_tp_v_{i}")
                            else:
                                # 反推當前 ROE
                                curr_roe = calc_roe_from_price(pos['entry'], pos['lev'], pos['type'], pos['tp'])
                                target_roe = c_val.number_input("盈虧 %", value=float(curr_roe), step=5.0, key=f"mod_tp_r_{i}")
                                new_val = calc_price_from_roe(pos['entry'], pos['lev'], pos['type'], target_roe)
                                c_val.caption(f"對應價格: {fmt_price(new_val)}")
                            
                            new_ratio = c_ratio.selectbox("平倉 %", [25, 50, 75, 100], index=[25,50,75,100].index(pos.get('tp_ratio', 100)), key=f"mod_tp_rat_{i}")
                            
                            col_upd, col_can = st.columns(2)
                            if col_upd.button("更新", key=f"btn_mod_tp_{i}", use_container_width=True):
                                st.session_state.positions[i]['tp'] = new_val
                                st.session_state.positions[i]['tp_ratio'] = new_ratio
                                st.toast("✅ 止盈單已更新"); st.rerun()
                            if col_can.button("撤銷", key=f"btn_can_tp_{i}", use_container_width=True):
                                cancel_order(i, 'TP')

                    # 顯示現有 SL
                    if pos.get('sl', 0) > 0:
                        has_orders = True
                        with st.expander(f"🛡️ 止損 (SL) - {pos['symbol']}", expanded=True):
                            st.write(f"觸發價: **{fmt_price(pos['sl'])}** (平倉 {pos.get('sl_ratio', 100)}%)")
                            
                            input_mode_sl = st.radio("修改方式", ["價格", "ROE %"], horizontal=True, key=f"mode_sl_{i}")
                            c_val_sl, c_ratio_sl = st.columns([2, 1])
                            
                            if input_mode_sl == "價格":
                                new_val_sl = c_val_sl.number_input("價格", value=float(pos['sl']), key=f"mod_sl_v_{i}")
                            else:
                                curr_roe_sl = calc_roe_from_price(pos['entry'], pos['lev'], pos['type'], pos['sl'])
                                target_roe_sl = c_val_sl.number_input("盈虧 %", value=float(curr_roe_sl), step=5.0, key=f"mod_sl_r_{i}")
                                new_val_sl = calc_price_from_roe(pos['entry'], pos['lev'], pos['type'], target_roe_sl)
                                c_val_sl.caption(f"對應價格: {fmt_price(new_val_sl)}")

                            new_ratio_sl = c_ratio_sl.selectbox("平倉 %", [25, 50, 75, 100], index=[25,50,75,100].index(pos.get('sl_ratio', 100)), key=f"mod_sl_rat_{i}")
                            
                            col_upd_sl, col_can_sl = st.columns(2)
                            if col_upd_sl.button("更新", key=f"btn_mod_sl_{i}", use_container_width=True):
                                st.session_state.positions[i]['sl'] = new_val_sl
                                st.session_state.positions[i]['sl_ratio'] = new_ratio_sl
                                st.toast("✅ 止損單已更新"); st.rerun()
                            if col_can_sl.button("撤銷", key=f"btn_can_sl_{i}", use_container_width=True):
                                cancel_order(i, 'SL')

                    # 新增委託單
                    if pos.get('tp', 0) == 0 and pos.get('sl', 0) == 0:
                        st.markdown(f"**{pos['symbol']}** 暫無掛單")
                        with st.expander("➕ 新增委託單 (含試算)"):
                            add_mode = st.radio("輸入單位", ["價格 (Price)", "盈虧率 (ROE %)"], horizontal=True, key=f"add_mode_{i}")
                            
                            c1, c2 = st.columns(2)
                            # 初始化變數
                            final_tp_price = 0.0
                            final_sl_price = 0.0
                            
                            if add_mode == "價格 (Price)":
                                final_tp_price = c1.number_input("止盈價格 TP", min_value=0.0, key=f"add_tp_p_{i}")
                                final_sl_price = c2.number_input("止損價格 SL", min_value=0.0, key=f"add_sl_p_{i}")
                            else:
                                tp_roe = c1.number_input("止盈 % (例如 30)", min_value=0.0, value=30.0, step=5.0, key=f"add_tp_r_{i}")
                                sl_roe = c2.number_input("止損 % (例如 -10)", max_value=0.0, value=-10.0, step=5.0, key=f"add_sl_r_{i}")
                                # 自動換算
                                final_tp_price = calc_price_from_roe(pos['entry'], pos['lev'], pos['type'], tp_roe)
                                final_sl_price = calc_price_from_roe(pos['entry'], pos['lev'], pos['type'], sl_roe)
                                c1.success(f"價格: {fmt_price(final_tp_price)}")
                                c2.error(f"價格: {fmt_price(final_sl_price)}")
                            
                            st.write("觸發後平倉比例:")
                            ratio_choice = st.radio("選擇比例", [25, 50, 75, 100], index=3, horizontal=True, key=f"add_ratio_{i}", format_func=lambda x: f"{x}%")
                            
                            if st.button("確認添加", key=f"btn_add_ord_{i}", use_container_width=True):
                                if final_tp_price > 0:
                                    st.session_state.positions[i]['tp'] = final_tp_price
                                    st.session_state.positions[i]['tp_ratio'] = ratio_choice
                                if final_sl_price > 0:
                                    st.session_state.positions[i]['sl'] = final_sl_price
                                    st.session_state.positions[i]['sl_ratio'] = ratio_choice
                                st.toast("✅ 委託單已添加"); st.rerun()
                        st.divider()

            if not has_orders and not st.session_state.positions:
                st.info("無持倉，無法掛單")
            elif not has_orders:
                st.info("目前無活躍委託單")

        # --- Tab 3: 歷史 ---
        with tab_hist:
            if st.session_state.history:
                hist_df = pd.DataFrame(st.session_state.history[::-1])
                st.dataframe(hist_df[['幣種','動作','獲利%','損益(U)','時間']], hide_index=True)
            else:
                st.info("暫無交易紀錄")

        # Open new position area
        st.markdown("##### 🚀 開立新倉位")
        col_s1, col_s2 = st.columns(2)
        trade_type = col_s1.selectbox("方向", ["🟢 做多 (Long)", "🔴 做空 (Short)"], key="new_side")
        leverage = col_s2.number_input("槓桿", 1, 125, 20, key="new_lev")
        principal = st.number_input("本金 (U)", 10.0, float(st.session_state.balance), 1000.0, key="new_amt")
        with st.expander("進階設定 (TP/SL)"):
            set_tp = st.number_input("止盈 TP", value=0.0, format="%.8f", key="new_tp")
            set_sl = st.number_input("止損 SL", value=0.0, format="%.8f", key="new_sl")
        if st.button("確認下單", type="primary", use_container_width=True):
            if principal > st.session_state.balance:
                st.error("餘額不足！")
            else:
                new_pos = {"symbol": symbol, "type": "Long" if "做多" in trade_type else "Short",
                           "entry": curr_price, "lev": leverage, "margin": principal,
                           "tp": set_tp, "sl": set_sl, "time": datetime.now().strftime('%m-%d %H:%M'),
                           "tp_ratio": 100, "sl_ratio": 100} # Initialize ratios
                st.session_state.positions.append(new_pos); st.session_state.balance -= principal; st.rerun()

    # Analysis
    pivots = calculate_zigzag(df)
    bull_fvg, bear_fvg = calculate_fvg(df)
    bull_div, bear_div = detect_div(df)
    score, struct_t, six_t, fvg_t, div_t, rsi_t = calculate_score_v17(pivots, last, df, bull_fvg, bear_fvg, bull_div, bear_div)

    atr = float(last['ATR']) if not pd.isna(last['ATR']) else float(last['Close'])*0.02
    pivot_lows = [p['val'] for p in pivots if p['type']=='low']; pivot_highs = [p['val'] for p in pivots if p['type']=='high']
    buy_sl = pivot_lows[-1] if pivot_lows else float(last['Close']) - 2*atr
    sell_sl = pivot_highs[-1] if pivot_highs else float(last['Close']) + 2*atr
    if buy_sl >= last['Close']: buy_sl = float(last['Close']) - 2*atr
    if sell_sl <= last['Close']: sell_sl = float(last['Close']) + 2*atr

    tp1 = 0; tp2 = 0; entry_zone = "現價"; risk_warning = ""
    if len(pivots) >= 2:
        lh = [p['val'] for p in pivots if p['type']=='high'][-1]
        ll = [p['val'] for p in pivots if p['type']=='low'][-1]
        diff = abs(lh - ll)
        if score >= 0:
            tp1 = lh; tp2 = ll + diff * 1.618
            fib_low = ll + diff * 0.382; fib_high = ll + diff * 0.618
            if last['Close'] < fib_high and last['Close'] > buy_sl: entry_zone = f"{fmt_price(last['Close'])} (現價優)"
            else: entry_zone = f"{fmt_price(fib_low)} ~ {fmt_price(fib_high)}"
            if last['Close'] >= tp1:
                tp1 = ll + diff * 1.272; tp2 = ll + diff * 1.618; risk_warning = "價格創高，止盈上移"
            elif last['Close'] < buy_sl:
                risk_warning = "❌ 結構破壞 (跌破止損)"; score = 0
        else:
            tp1 = ll; tp2 = lh - diff * 1.618
            fib_low = lh - diff * 0.618; fib_high = lh - diff * 0.382
            if last['Close'] > fib_low and last['Close'] < sell_sl: entry_zone = f"{fmt_price(last['Close'])} (現價優)"
            else: entry_zone = f"{fmt_price(fib_low)} ~ {fmt_price(fib_high)}"
            if last['Close'] <= tp1:
                tp1 = lh - diff * 1.272; tp2 = lh - diff * 1.618; risk_warning = "價格創低，止盈下移"
            elif last['Close'] > sell_sl:
                risk_warning = "❌ 結構破壞 (突破止損)"; score = 0

    st.info("🛡️ **AI 實戰風控報告**")
    st.markdown(generate_ai_report(symbol, last['Close'], score, struct_t, six_t, fvg_t, div_t, rsi_t, buy_sl, sell_sl, tp1, tp2, entry_zone, risk_warning))
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    action_label = "觀望"
    if risk_warning and "破" in risk_warning:
        action_label = "⛔ " + risk_warning; score_display = "N/A"
    else:
        if score >= 8: action_label = "🔥 強力買進"
        elif score >= 5: action_label = "🟢 買進"
        elif score <= -8: action_label = "💀 強力賣出"
        elif score <= -5: action_label = "🔴 賣出"
        score_display = f"{score}/10"

    m1.metric("AI 評級", score_display, action_label)
    m2.metric("建議入場", entry_zone.split("~")[0] if "~" in entry_zone else entry_zone, "校正後")
    if score >= 0:
        m3.metric("止盈 TP1", fmt_price(tp1), "目標"); m4.metric("止損 SL", fmt_price(buy_sl), "防守", delta_color="inverse")
    else:
        m3.metric("止盈 TP1", fmt_price(tp1), "目標", delta_color="inverse"); m4.metric("止損 SL", fmt_price(sell_sl), "防守", delta_color="normal")

    # Chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7,0.3])
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='價格', line=dict(color='white', width=2)), row=1, col=1)
    if show_six:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20', line=dict(width=1), fill='tonexty'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA60'], name='EMA60', line=dict(width=1)), row=1, col=1)
    if show_fvg:
        for f in bull_fvg:
            fig.add_shape(type="rect", x0=f['start'], x1=df.index[-1], y0=f['bottom'], y1=f['top'], fillcolor="rgba(0,255,0,0.2)", line_width=0, xref='x', yref='y')
        for f in bear_fvg:
            fig.add_shape(type="rect", x0=f['start'], x1=df.index[-1], y0=f['bottom'], y1=f['top'], fillcolor="rgba(255,0,0,0.15)", line_width=0, xref='x', yref='y')
    if show_zigzag and pivots:
        px = [p['idx'] for p in pivots]; py = [p['val'] for p in pivots]
        fig.add_trace(go.Scatter(x=px, y=py, mode='lines+markers', name='ZigZag', line=dict(color='orange', width=3), marker_size=6), row=1, col=1)
        for i in range(2, len(pivots)):
            p = pivots[i]; prev = pivots[i-2]
            try:
                txt = ("HH" if p['val']>prev['val'] else "LH") if p['type']=='high' else ("HL" if p['val']>prev['val'] else "LL")
                clr = 'red' if p['type']=='high' else '#00FF00'
                fig.add_annotation(x=p['idx'], y=p['val'], text=f"<b>{txt}</b>", showarrow=False, font=dict(color=clr, size=12), yshift=20 if p['type']=='high' else -20)
            except:
                continue
    if show_fib and tp1 > 0:
        fig.add_hline(y=tp1, line_dash="dash", line_color="yellow", annotation_text=f"TP1 {fmt_price(tp1)}")
        fig.add_hline(y=tp2, line_dash="dash", line_color="#00FF00", annotation_text=f"TP2 {fmt_price(tp2)}")
    
    # Show orders on chart
    if show_orders and st.session_state.positions:
        for pos in st.session_state.positions:
            if pos['symbol'] == symbol:
                if pos.get('tp', 0) > 0:
                    fig.add_hline(y=pos['tp'], line_dash="dashdot", line_color="#00FF00", annotation_text=f"止盈 {pos.get('tp_ratio',100)}% @ {fmt_price(pos['tp'])}")
                if pos.get('sl', 0) > 0:
                    fig.add_hline(y=pos['sl'], line_dash="dashdot", line_color="#FF0000", annotation_text=f"止損 {pos.get('sl_ratio',100)}% @ {fmt_price(pos['sl'])}")

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1); fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    fig.update_layout(title=f"{symbol} 實戰分析圖", template="plotly_dark", height=800)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error(f"❌ 找不到 {symbol} 數據。")
