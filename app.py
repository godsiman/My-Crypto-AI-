import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os

# --- Page setup ---
st.set_page_config(page_title="全方位戰情室 AI (v95.0)", layout="wide", page_icon="🏦")
st.markdown("### 🏦 全方位戰情室 AI (v95.0 精準狙擊回測版)")

# --- [核心] NpEncoder ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

# --- Persistence ---
DATA_FILE = "trade_data_live.json"

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
        st.session_state.symbol_input = "" 
        st.session_state.init_done = True

    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                st.session_state.balance = float(data.get("balance", 10000.0))
                st.session_state.positions = data.get("positions", [])
                st.session_state.pending_orders = data.get("pending_orders", [])
                st.session_state.history = data.get("history", [])
        except: pass

load_data()

# --- Helpers ---
def fmt_price(val):
    if val is None: return "N/A"
    try:
        valf = float(val)
        if valf < 1.0: return f"${valf:.6f}"
        elif valf < 20: return f"${valf:.4f}"
        else: return f"${valf:,.2f}"
    except: return str(val)

def get_current_price(sym):
    try:
        ticker = yf.Ticker(sym)
        fi = getattr(ticker, 'fast_info', None)
        if fi and getattr(fi, 'last_price', None):
            return float(fi.last_price)
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except: pass
    return None

def get_locked_funds():
    locked = 0.0
    for p in st.session_state.positions: locked += float(p.get('margin', 0.0))
    for o in st.session_state.pending_orders: locked += float(o.get('margin', 0.0))
    return locked

# --- Indicators ---
def calculate_indicators(df):
    if df is None or df.empty: return df
    df = df.copy()
    
    # VWAP
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['Typical_Price'] * df['Volume']
    df['Total_VP'] = df['VP'].cumsum()
    df['Total_Vol'] = df['Volume'].cumsum()
    df['VWAP'] = df['Total_VP'] / df['Total_Vol']
    
    # ADX (趨勢強度)
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    df['+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    df['-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(14).mean()
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14).mean() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14).mean() / df['ATR'])
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].ewm(alpha=1/14).mean()

    df['EMA7'] = df['Close'].ewm(span=7).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA60'] = df['Close'].ewm(span=60).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean().replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    df['MA20'] = df['Close'].rolling(20).mean()
    df['STD20'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + (df['STD20'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['STD20'] * 2)
    
    return df

# --- Chart Data ---
def get_chart_data(symbol, interval_ui):
    if interval_ui == "15分鐘": period, interval = "1mo", "15m"
    elif interval_ui == "1小時": period, interval = "6mo", "1h"
    elif interval_ui == "4小時": period, interval = "6mo", "1h"
    else: period, interval = "2y", "1d"
    
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty: return None
        if interval_ui == "4小時":
            agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df = df.resample('4h').agg(agg).dropna()
        df = calculate_indicators(df)
        return df
    except: return None

# --- [大腦升級] 精準狙擊回測 ---
def run_backtest_sniper(df, initial_capital=10000):
    if df is None or len(df) < 50: return None
    
    capital = initial_capital
    position = 0 # 0: Cash, 1: Long, -1: Short
    entry_price = 0.0
    equity_curve = []
    trades = []
    
    # 這裡的邏輯與 get_institutional_strategy 保持一致，甚至更嚴格
    for i in range(50, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        timestamp = df.index[i]
        price = curr['Close']
        
        # 1. 環境過濾 (ADX)
        # 如果 ADX < 20，視為盤整垃圾時間，強制空手
        is_trending = curr['ADX'] > 20
        
        # 2. 綜合評分
        score = 0
        
        # VWAP 濾網
        if price > curr['VWAP']: score += 1
        else: score -= 1
        
        # 均線趨勢
        if price > curr['EMA20'] and curr['EMA20'] > curr['EMA60']: score += 1
        elif price < curr['EMA20'] and curr['EMA20'] < curr['EMA60']: score -= 1
        
        # 動能 MACD
        if curr['MACD'] > curr['Signal']: score += 1
        else: score -= 1
        
        # 3. 決策信號 (只有在趨勢明確時才開倉)
        signal = 0
        if is_trending:
            if score >= 3: signal = 1   # 強力做多條件
            elif score <= -3: signal = -1 # 強力做空條件
        
        # --- 執行交易 (模擬) ---
        
        # 進場邏輯
        if position == 0:
            if signal == 1:
                position = 1
                entry_price = price
                trades.append({'time': timestamp, 'type': '🟢 做多 (Long)', 'price': price, 'balance': capital})
            elif signal == -1:
                position = -1
                entry_price = price
                trades.append({'time': timestamp, 'type': '🔴 做空 (Short)', 'price': price, 'balance': capital})
        
        # 出場邏輯 (或是反手)
        elif position == 1: # 持多單
            # 如果分數轉弱 (<0) 就平倉，不一定要等到 -3 反手
            if score < 0:
                pnl = (price - entry_price) / entry_price * capital
                capital += pnl
                position = 0 # 平倉觀望
                trades.append({'time': timestamp, 'type': '⚪ 平多 (Close Long)', 'price': price, 'pnl': pnl, 'balance': capital})
            elif signal == -1: # 強力反手訊號
                pnl = (price - entry_price) / entry_price * capital
                capital += pnl
                position = -1
                entry_price = price
                trades.append({'time': timestamp, 'type': '🔄 反手做空', 'price': price, 'pnl': pnl, 'balance': capital})
        
        elif position == -1: # 持空單
            if score > 0:
                pnl = (entry_price - price) / entry_price * capital
                capital += pnl
                position = 0 # 平倉觀望
                trades.append({'time': timestamp, 'type': '⚪ 平空 (Close Short)', 'price': price, 'pnl': pnl, 'balance': capital})
            elif signal == 1: # 強力反手訊號
                pnl = (entry_price - price) / entry_price * capital
                capital += pnl
                position = 1
                entry_price = price
                trades.append({'time': timestamp, 'type': '🔄 反手做多', 'price': price, 'pnl': pnl, 'balance': capital})
                
        # 記錄每日權益
        curr_equity = capital
        if position == 1:
            curr_equity += (price - entry_price) / entry_price * capital
        elif position == -1:
            curr_equity += (entry_price - price) / entry_price * capital
            
        equity_curve.append({'time': timestamp, 'equity': curr_equity})
        
    return pd.DataFrame(equity_curve), pd.DataFrame(trades)

# --- AI Strategy (Live) ---
@st.cache_data(ttl=120)
def get_institutional_strategy(symbol, current_interval_ui):
    macro_intervals = {"M": ("1mo","5y"), "W": ("1wk","2y"), "D": ("1d","1y")}
    macro_trends = {}
    macro_score = 0
    for tf, (inter, per) in macro_intervals.items():
        try:
            df_m = yf.Ticker(symbol).history(period=per, interval=inter)
            if not df_m.empty:
                df_m = calculate_indicators(df_m)
                last = df_m.iloc[-1]
                if last['Close'] > last['EMA20']:
                    macro_trends[tf] = "多頭"
                    macro_score += 1
                else:
                    macro_trends[tf] = "空頭"
                    macro_score -= 1
        except: macro_trends[tf] = "未知"

    df = get_chart_data(symbol, current_interval_ui)
    if df is None or len(df) < 30: return None
    last = df.iloc[-1]; prev = df.iloc[-2]
    micro_score = 0; signals = []
    
    # VWAP
    is_above_vwap = False
    if last['Close'] > last['VWAP']:
        signals.append("🏦 站上 VWAP (機構成本之上)")
        micro_score += 1
        is_above_vwap = True
    else:
        signals.append("⚠️ 跌破 VWAP (機構成本之下)")
        micro_score -= 1
        is_above_vwap = False

    # ADX
    adx_val = last.get('ADX', 0)
    if adx_val < 20:
        signals.append("💤 ADX < 20 (無趨勢盤整)")
        micro_score *= 0.5
    elif adx_val > 25:
        signals.append(f"🔥 ADX = {adx_val:.1f} (趨勢形成)")
        micro_score *= 1.2

    # Technicals
    if last['Close'] > last['EMA7']:
        signals.append("⚡ 站上短線 (EMA7)")
        micro_score += 1.5
    else:
        signals.append("⚠️ 跌破短線 (EMA7)")
        micro_score -= 1.5

    if last['Close'] > last['EMA20'] > last['EMA60']:
        signals.append("✅ 均線多頭排列")
        micro_score += 2
    elif last['Close'] < last['EMA20'] < last['EMA60']:
        signals.append("🔻 均線空頭排列")
        micro_score -= 2
        
    if last['MACD'] > last['Signal'] and prev['MACD'] <= prev['Signal']:
        signals.append("🚀 MACD 金叉")
        micro_score += 2
    
    final_score = (macro_score * 0.3) + (micro_score * 0.7)
    direction = "觀望"
    
    if final_score >= 2.0: 
        direction = "強力做多 (Strong Buy)"
        action_msg = "🤖 AI 建議：現在動能超強！別猶豫了，建議直接市價進場追擊！"
    elif final_score >= 0.5: 
        direction = "嘗試做多 (Buy)"
        action_msg = "🤖 AI 建議：趨勢是向上的，但現在價格有點高，掛單等回調再接比較安全。"
    elif final_score <= -2.0: 
        direction = "強力做空 (Strong Sell)"
        action_msg = "🤖 AI 建議：空頭很兇！反彈就是空，不要手軟！"
    elif final_score <= -0.5: 
        direction = "嘗試做空 (Sell)"
        action_msg = "🤖 AI 建議：看起來快跌了，掛高一點做空比較保險。"
    else:
        action_msg = "🤖 AI 建議：現在多空不明，先喝杯咖啡觀望一下吧。"

    if is_above_vwap:
        vwap_msg = "🟢 大戶還在顧盤！價格在成本線之上，趨勢有支撐，安啦！"
        vwap_type = "success"
    else:
        vwap_msg = "⚠️ 主力好像棄守了...價格跌破成本線，小心被套！"
        vwap_type = "warning"

    curr_price = last['Close']
    atr = last.get('ATR', curr_price * 0.02)
    recent_high = df['High'].tail(50).max()
    recent_low = df['Low'].tail(50).min()
    fib_0618 = recent_low + (recent_high - recent_low) * 0.618
    fib_0382 = recent_low + (recent_high - recent_low) * 0.382
    
    if final_score > 0:
        entry_target = max(fib_0618, last['EMA20']) 
        entry = entry_target if (curr_price - entry_target) > atr else curr_price
        tp = entry + (atr * 3)
        sl = entry - (atr * 1.5)
    else:
        entry_target = min(fib_0382, last['EMA20'])
        entry = entry_target if (entry_target - curr_price) > atr else curr_price
        tp = entry - (atr * 3)
        sl = entry + (atr * 1.5)

    return {
        "direction": direction,
        "score": final_score,
        "macro_trends": macro_trends,
        "signals": signals,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "df": df,
        "last_price": curr_price,
        "action_msg": action_msg,
        "vwap_msg": vwap_msg,
        "vwap_type": vwap_type
    }

# --- Callbacks ---
def on_select_change():
    raw_val = st.session_state.quick_select
    new_sym = raw_val.split(" ")[0]
    if st.session_state.market == "台股" and new_sym.isdigit(): new_sym += ".TW"
    if st.session_state.market == "加密貨幣" and "-" not in new_sym and "USD" not in new_sym: new_sym += "-USD"
    st.session_state.chart_symbol = new_sym
    st.session_state.symbol_input = "" 

def on_input_change():
    val = st.session_state.symbol_input.strip().upper()
    if val:
        if st.session_state.market == "台股" and val.isdigit(): val += ".TW"
        if st.session_state.market == "加密貨幣" and "-" not in val and "USD" not in val: val += "-USD"
        st.session_state.chart_symbol = val

def jump_to_symbol(target_symbol):
    st.session_state.chart_symbol = target_symbol
    st.session_state.symbol_input = "" 

# --- Dialogs ---
@st.dialog("⚡ 倉位管理")
def manage_position_dialog(i, pos, current_price):
    st.markdown(f"**{pos.get('symbol','--')}**")
    try:
        entry = float(pos.get('entry', 0))
        lev = float(pos.get('lev', 1))
        margin = float(pos.get('margin', 0))
        pos_type = pos.get('type', 'Long')
        d = 1 if pos_type == 'Long' else -1
        u_pnl = margin * (((current_price - entry) / entry) * lev * d)
        roe_pct = (u_pnl / margin) * 100 if margin > 0 else 0.0
        color = "green" if u_pnl >= 0 else "red"
        st.markdown(f"未結盈虧: <span style='color:{color}; font-weight:bold'>${u_pnl:+.2f} ({roe_pct:+.2f}%)</span>", unsafe_allow_html=True)
    except: entry=0; lev=1; pos_type='Long'

    tab_close, tab_tpsl = st.tabs(["平倉", "止盈止損"])
    with tab_close:
        ratio = st.radio("平倉 %", [25,50,75,100], 3, horizontal=True, key=f"dr_{i}")
        if st.button("確認平倉", key=f"btn_c_{i}", type="primary", use_container_width=True):
            close_position(i, ratio, "手動", current_price)
            st.rerun()
    with tab_tpsl:
        mode = st.radio("設定模式", ["價格", "ROE %"], horizontal=True, key=f"m_mode_{i}")
        new_tp = float(pos.get('tp', 0)); new_sl = float(pos.get('sl', 0))
        if mode == "價格":
            c1, c2 = st.columns(2)
            new_tp = c1.number_input("TP 價格", value=new_tp, key=f"ntp_p_{i}", format="%.6f")
            new_sl = c2.number_input("SL 價格", value=new_sl, key=f"nsl_p_{i}", format="%.6f")
        else:
            c1, c2 = st.columns(2)
            roe_tp = c1.number_input("止盈 %", value=0.0, key=f"ntp_r_{i}")
            roe_sl = c2.number_input("止損 %", value=0.0, key=f"nsl_r_{i}")
            direction = 1 if pos_type == 'Long' else -1
            if roe_tp > 0:
                calc_tp = entry * (1 + (roe_tp / 100.0) / lev * direction)
                c1.caption(f"預估: {fmt_price(calc_tp)}")
                new_tp = calc_tp
            if roe_sl > 0:
                calc_sl = entry * (1 - (roe_sl / 100.0) / lev * direction)
                c2.caption(f"預估: {fmt_price(calc_sl)}")
                new_sl = calc_sl
        if st.button("更新設定", key=f"btn_u_{i}", use_container_width=True):
            st.session_state.positions[i]['tp'] = new_tp
            st.session_state.positions[i]['sl'] = new_sl
            save_data()
            st.toast("✅ 已更新")
            st.rerun()

def close_position(pos_index, percentage, reason, exit_price):
    if pos_index >= len(st.session_state.positions): return
    pos = st.session_state.positions[pos_index]
    close_ratio = percentage / 100.0
    margin = float(pos.get('margin', 0))
    close_margin = margin * close_ratio 
    direction = 1 if pos.get('type') == 'Long' else -1
    entry = float(pos.get('entry', 1))
    lev = float(pos.get('lev', 1))
    pnl = close_margin * (((exit_price - entry) / entry) * lev * direction)
    roe_pct = (pnl / close_margin) * 100 if close_margin > 0 else 0.0
    st.session_state.balance += (close_margin + pnl)
    st.session_state.history.append({
        "時間": datetime.now().strftime("%m-%d %H:%M"),
        "幣種": pos.get('symbol'),
        "動作": f"平{percentage}%",
        "價格": exit_price,
        "盈虧": f"{pnl:+.2f} ({roe_pct:+.2f}%)",
        "原因": reason
    })
    if percentage == 100: st.session_state.positions.pop(pos_index)
    else: st.session_state.positions[pos_index]['margin'] -= close_margin
    save_data()

def cancel_order(idx):
    if idx < len(st.session_state.pending_orders):
        st.session_state.pending_orders.pop(idx)
        save_data()
        st.toast("已撤銷")

# --- Sidebar ---
st.sidebar.header("🎯 戰情室設定")
market = st.sidebar.radio("市場", ["加密貨幣", "美股", "台股"], index=0)
st.session_state.market = market
interval_ui = st.sidebar.radio("⏱️ K線週期", ["15分鐘", "1小時", "4小時", "日線"], index=3)

if market == "加密貨幣":
    targets = ["BTC-USD 比特幣", "ETH-USD 以太坊", "SOL-USD 索拉納", "DOGE-USD 狗狗幣", "XRP-USD 瑞波幣", "BNB-USD 幣安幣", "DNX-USD Dynex"]
elif market == "美股":
    targets = ["NVDA 輝達", "TSLA 特斯拉", "AAPL 蘋果", "MSFT 微軟", "AMD 超微", "COIN Coinbase"]
else:
    targets = ["2330.TW 台積電", "2317.TW 鴻海", "2454.TW 聯發科", "2603.TW 長榮", "0050.TW 元大台灣50"]

st.sidebar.markdown("---")
st.sidebar.write("🔍 搜尋/選擇")
st.sidebar.text_input("輸入代碼 (Enter 確認)", key="symbol_input", on_change=on_input_change)
st.sidebar.selectbox("快速選擇", targets, key="quick_select", on_change=on_select_change)
symbol = st.session_state.chart_symbol

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ 重置數據"):
    if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
    st.session_state.clear()
    st.rerun()

# --- Main Logic ---
with st.spinner(f"正在連線戰情中心 {symbol}..."):
    ai_res = get_institutional_strategy(symbol, interval_ui)

if ai_res:
    curr_price = ai_res['last_price']
    df_chart = ai_res['df']
    
    c1, c2, c3 = st.columns([2, 1, 1])
    is_up = df_chart.iloc[-1]['Close'] >= df_chart.iloc[-1]['Open']
    p_color = "#00C853" if is_up else "#FF3D00"
    if curr_price < 1.0: price_display = f"${curr_price:.6f}"
    else: price_display = f"${curr_price:,.2f}"

    c1.markdown(f"""
    <div style='display: flex; align-items: center; line-height: 1.5; padding-top: 5px; padding-bottom: 5px; white-space: nowrap; overflow: visible;'>
        <span style='font-size: 40px; font-weight: bold; margin-right: 15px; color: #ffffff;'>{symbol}</span>
        <span style='font-size: 30px; color: #cccccc; margin-right: 15px;'>({interval_ui})</span>
        <span style='font-size: 42px; color: {p_color}; font-weight: bold;'>{price_display}</span>
    </div>
    """, unsafe_allow_html=True)
    
    balance = st.session_state.balance
    locked = get_locked_funds()
    available = balance - locked
    
    total_u_pnl = 0.0
    total_margin_used = 0.0
    for p in st.session_state.positions:
        try:
            cur = get_current_price(p['symbol'])
            if cur:
                d = 1 if p['type']=='Long' else -1
                m = float(p.get('margin', 0))
                pnl = m * (((cur - p['entry'])/p['entry']) * p['lev'] * d)
                total_u_pnl += pnl
                total_margin_used += m
        except: pass
    total_roe = (total_u_pnl / total_margin_used * 100) if total_margin_used > 0 else 0.0
    equity = balance + total_u_pnl

    m1, m2, m3 = st.columns(3)
    m1.metric("帳戶淨值 (Equity)", f"${equity:,.2f}")
    m2.metric("可用餘額 (Available)", f"${available:,.2f}")
    m3.metric("總未結盈虧 (PnL)", f"${total_u_pnl:+.2f}", delta=f"{total_roe:+.2f}%")

    st.divider()

    # --- Dashboard ---
    st.subheader("🧠 AI 戰略指揮中心")
    col_macro, col_signal, col_action = st.columns([1, 1.5, 1.5])
    with col_macro:
        st.markdown("#### 🔭 宏觀趨勢")
        def get_trend_icon(t): return "🟢 多頭" if t=="多頭" else ("🔴 空頭" if t=="空頭" else "⚪ 未知")
        st.write(f"**月線 (M):** {get_trend_icon(ai_res['macro_trends'].get('M'))}")
        st.write(f"**週線 (W):** {get_trend_icon(ai_res['macro_trends'].get('W'))}")
        st.write(f"**日線 (D):** {get_trend_icon(ai_res['macro_trends'].get('D'))}")
        
    with col_signal:
        st.markdown("#### 📡 技術形態訊號")
        if not ai_res['signals']: st.info("暫無明顯形態")
        else:
            for sig in ai_res['signals']: st.markdown(f"- {sig}")
                
    with col_action:
        st.markdown(f"#### 🚀 戰術建議: {ai_res['direction']}")
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("建議入場", fmt_price(ai_res['entry']))
        ac2.metric("目標止盈", fmt_price(ai_res['tp']), delta="TP")
        ac3.metric("防守止損", fmt_price(ai_res['sl']), delta="SL", delta_color="inverse")

    st.divider()

    # --- Chart ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['VWAP'], line=dict(color='orange', width=2), name='VWAP'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA7'], line=dict(color='white', width=1.5), name='EMA7 (短線)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA20'], line=dict(color='yellow', width=1), name='EMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA60'], line=dict(color='cyan', width=1), name='EMA60'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Upper'], line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'), name='BB上軌'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Lower'], line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'), name='BB下軌'), row=1, col=1)
    for pos in st.session_state.positions:
        if pos['symbol'] == symbol:
            fig.add_hline(y=pos['entry'], line_dash="dash", line_color="orange", annotation_text=f"持倉 {pos['type']}")
    
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='violet', width=2), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    fig.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), dragmode='pan', title_text=f"{symbol} - {interval_ui} (台北時間)")
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Trading ---
    tab_trade, tab_orders, tab_history, tab_backtest = st.tabs(["⚡ 下單交易", "📋 訂單管理", "📜 歷史訂單", "📈 策略回測"])
    
    with tab_trade:
        col_ctrl, col_info = st.columns([2, 1])
        with col_ctrl:
            c_t1, c_t2, c_t3 = st.columns(3)
            trade_type = c_t1.selectbox("方向", ["做多 (Long)", "做空 (Short)"], index=0 if "多" in ai_res['direction'] else 1)
            lev = c_t2.slider("槓桿", 1, 125, 20)
            amt = c_t3.number_input("本金 (U)", min_value=10.0, value=float(st.session_state.trade_amt_box))
            st.session_state.trade_amt_box = amt
            
            with st.expander("進階 (止盈止損)", expanded=True):
                mode = st.radio("單位", ["價格", "ROE %"], horizontal=True)
                rec_tp = ai_res['tp']; rec_sl = ai_res['sl']
                if mode == "價格":
                    t_tp = st.number_input("止盈價格", value=float(rec_tp), format="%.6f")
                    t_sl = st.number_input("止損價格", value=float(rec_sl), format="%.6f")
                else:
                    roe_tp = st.number_input("止盈 ROE %", value=0.0)
                    roe_sl = st.number_input("止損 ROE %", value=0.0)
                    t_tp, t_sl = 0.0, 0.0
                    direction = 1 if "多" in trade_type else -1
                    if roe_tp > 0: t_tp = curr_price * (1 + (roe_tp / 100) / lev * direction)
                    if roe_sl > 0: t_sl = curr_price * (1 - (roe_sl / 100) / lev * direction)
                t_entry = st.number_input("掛單價格 (0=市價)", value=0.0, format="%.6f")

            if st.button("🚀 下單 / 掛單", type="primary", use_container_width=True):
                final_entry = curr_price if t_entry == 0 else t_entry
                if mode == "ROE %":
                    direction = 1 if "多" in trade_type else -1
                    if roe_tp > 0: t_tp = final_entry * (1 + (roe_tp / 100) / lev * direction)
                    if roe_sl > 0: t_sl = final_entry * (1 - (roe_sl / 100) / lev * direction)

                if amt > available:
                    st.error(f"可用餘額不足！ (可用: ${available:.2f})")
                else:
                    new_pos = {
                        "symbol": symbol,
                        "type": "Long" if "多" in trade_type else "Short",
                        "entry": final_entry,
                        "lev": lev,
                        "margin": amt,
                        "tp": t_tp,
                        "sl": t_sl,
                        "time": datetime.now().strftime("%m-%d %H:%M")
                    }
                    if t_entry == 0:
                        st.session_state.positions.append(new_pos)
                        st.toast(f"✅ 市價成交！")
                    else:
                        st.session_state.pending_orders.append(new_pos)
                        st.toast(f"⏳ 掛單提交！")
                    save_data()
                    st.rerun()
        
        with col_info:
            st.info(ai_res['action_msg'])
            if ai_res['vwap_type'] == 'success': st.success(ai_res['vwap_msg'])
            else: st.warning(ai_res['vwap_msg'])

    with tab_orders:
        st.subheader("🔥 持倉中")
        if not st.session_state.positions: st.caption("無持倉")
        else:
            for i, pos in enumerate(st.session_state.positions):
                p_sym = pos['symbol']
                p_cur = get_current_price(p_sym)
                if p_cur:
                    d = 1 if pos['type']=='Long' else -1
                    pnl = pos['margin'] * (((p_cur - pos['entry'])/pos['entry']) * pos['lev'] * d)
                    roe_pct = (pnl / pos['margin']) * 100
                    if roe_pct <= -100.0:
                        close_position(i, 100, "💀 爆倉 (-100%)", p_cur)
                        st.toast(f"⚠️ {p_sym} 已爆倉！保證金歸零")
                        st.rerun()
                    clr = "#00C853" if pnl >= 0 else "#FF3D00"
                    c_btn, c_info, c_mng = st.columns([1.5, 3, 1])
                    c_btn.button(f"📊 {p_sym}", key=f"nav_p_{i}", on_click=jump_to_symbol, args=(p_sym,))
                    c_info.markdown(f"""
                    <div style='font-size:14px'>
                        <b>{pos['type']} x{pos['lev']}</b> <span style='color:#aaa'>| 本金 ${pos['margin']:.0f}</span><br>
                        盈虧: <span style='color:{clr}; font-weight:bold'>${pnl:+.2f} ({roe_pct:+.2f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)
                    if c_mng.button("⚙️", key=f"mng_{i}"): manage_position_dialog(i, pos, p_cur)
                    st.divider()

        st.subheader("⏳ 掛單中")
        if not st.session_state.pending_orders: st.caption("無掛單")
        else:
            for i, ord in enumerate(st.session_state.pending_orders):
                o_sym = ord['symbol']
                c_btn, c_info, c_cnl = st.columns([1.5, 3, 1])
                c_btn.button(f"📊 {o_sym}", key=f"nav_o_{i}", on_click=jump_to_symbol, args=(o_sym,))
                c_info.markdown(f"{ord['type']} x{ord['lev']} @ <b>{fmt_price(ord['entry'])}</b>", unsafe_allow_html=True)
                if c_cnl.button("❌", key=f"cnl_{i}"): cancel_order(i); st.rerun()
                st.divider()

    with tab_history:
        st.subheader("📜 歷史戰績")
        if not st.session_state.history: st.info("暫無歷史紀錄")
        else:
            hist_df = pd.DataFrame(st.session_state.history)
            hist_df = hist_df.iloc[::-1]
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

    with tab_backtest:
        st.subheader(f"📈 {symbol} 歷史回測 (精準狙擊模式)")
        st.caption("AI 將模擬過去 500 根 K 線，嚴格執行 VWAP + ADX + EMA 策略。")
        if st.button("🚀 開始回測"):
            with st.spinner("正在穿越時空進行模擬交易..."):
                eq_curve, trades_log = run_backtest_sniper(df_chart, 10000)
            if eq_curve is not None and not eq_curve.empty:
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=eq_curve['time'], y=eq_curve['equity'], mode='lines', name='資金曲線', line=dict(color='#00C853')))
                fig_bt.update_layout(template="plotly_dark", title="回測資金增長", height=400)
                st.plotly_chart(fig_bt, use_container_width=True)
                
                initial = 10000
                final = eq_curve['equity'].iloc[-1]
                total_ret = (final - initial) / initial * 100
                win_count = len(trades_log[trades_log['pnl'] > 0]) if not trades_log.empty and 'pnl' in trades_log else 0
                total_trades = len(trades_log[trades_log['type'].str.contains('Close')]) if not trades_log.empty else 0
                win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
                
                m1, m2, m3 = st.columns(3)
                m1.metric("期初本金", "$10,000")
                m2.metric("期末淨值", f"${final:,.2f}", delta=f"{total_ret:+.2f}%")
                m3.metric("勝率 (Win Rate)", f"{win_rate:.1f}%", f"共 {total_trades} 筆交易")
                
                if not trades_log.empty:
                    st.write("交易明細：")
                    st.dataframe(trades_log, use_container_width=True)
            else:
                st.warning("數據不足，無法回測")

else:
    st.error(f"❌ 無法讀取 {symbol}，請確認代碼或網路連線。")
