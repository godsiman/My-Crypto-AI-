# app.py - 全功能 Streamlit 戰情室 (整合回測 + 模擬下單 + SuperTrend)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os
from streamlit_autorefresh import st_autorefresh

# --- Page setup ---
st.set_page_config(page_title="全方位戰情室 AI (v99.0)", layout="wide", page_icon="🏦")

# --- Auto refresh (kept but heavy operations are cached) ---
count = st_autorefresh(interval=60000, limit=None, key="market_refresh")

st.markdown("### 🏦 全方位戰情室 AI (v99.0 自動戰鬥版)")

# --- JSON encoder for numpy types ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

# --- Persistence file ---
DATA_FILE = "trade_data_live.json"

# --- Initialize session state ---
def init_session_state():
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

init_session_state()

# --- Load / Save data helpers ---
def save_data_to_file():
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

def load_data_from_file():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                st.session_state.balance = float(data.get("balance", st.session_state.balance))
                st.session_state.positions = data.get("positions", st.session_state.positions)
                st.session_state.pending_orders = data.get("pending_orders", st.session_state.pending_orders)
                st.session_state.history = data.get("history", st.session_state.history)
        except Exception as e:
            # ignore load errors
            pass

load_data_from_file()

# --- Format price helper ---
def fmt_price(val):
    if val is None: return "N/A"
    try:
        valf = float(val)
        if valf < 1.0: return f"${valf:.6f}"
        elif valf < 20: return f"${valf:.4f}"
        else: return f"${valf:,.2f}"
    except:
        return str(val)

# --- CACHED price fetch for reducing yfinance calls ---
@st.cache_data(ttl=30)
def fetch_price(sym):
    # small wrapper: try fast_info then fallback to 1m history
    try:
        ticker = yf.Ticker(sym)
        fi = getattr(ticker, 'fast_info', None)
        if fi and getattr(fi, 'last_price', None):
            return float(fi.last_price)
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass
    return None

def get_current_price(sym):
    return fetch_price(sym)

# --- get_chart_data with cache (per symbol+interval) ---
@st.cache_data(ttl=60)
def fetch_ohlcv(symbol, period, interval):
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_chart_data(symbol, interval_ui):
    # map UI to yfinance period/interval
    if interval_ui == "15分鐘":
        period, interval = "1mo", "15m"
    elif interval_ui == "1小時":
        period, interval = "6mo", "1h"
    elif interval_ui == "4小時":
        period, interval = "6mo", "1h"  # fetch 1h then resample
    else:
        period, interval = "2y", "1d"

    df = fetch_ohlcv(symbol, period, interval)
    if df is None or df.empty:
        return None

    if interval_ui == "4小時":
        # resample to 4h
        agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        df = df.resample('4h').agg(agg).dropna()

    df = calculate_indicators(df)
    return df

# --- SuperTrend (optimized-ish but clear) ---
def calculate_supertrend(df, period=9, multiplier=3.9):
    # This implementation follows standard SuperTrend logic
    df = df.copy()
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    hl2 = (high + low) / 2
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    # initialize
    final_ub.iloc[0] = basic_ub.iloc[0]
    final_lb.iloc[0] = basic_lb.iloc[0]
    supertrend.iloc[0] = basic_ub.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        if basic_ub.iloc[i] < final_ub.iloc[i-1] or close.iloc[i-1] > final_ub.iloc[i-1]:
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i-1]

        if basic_lb.iloc[i] > final_lb.iloc[i-1] or close.iloc[i-1] < final_lb.iloc[i-1]:
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i-1]

        if direction.iloc[i-1] == 1:
            if close.iloc[i] <= final_lb.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_ub.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lb.iloc[i]
        else:
            if close.iloc[i] >= final_ub.iloc[i]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lb.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_ub.iloc[i]

    df['SuperTrend'] = supertrend
    df['ST_Direction'] = direction
    return df

# --- Indicators (EMA52, RSI proxy, MACD) ---
def calculate_indicators(df):
    if df is None or df.empty: return df
    df = df.copy()
    df = calculate_supertrend(df, period=9, multiplier=3.9)
    df['EMA52'] = df['Close'].ewm(span=52, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']

    return df

# --- Simple function to compute locked funds ---
def get_locked_funds():
    locked = 0.0
    for p in st.session_state.positions:
        locked += float(p.get('margin', 0.0))
    for o in st.session_state.pending_orders:
        locked += float(o.get('margin', 0.0))
    return locked

# --- Check and execute pending orders for a symbol (called after price update) ---
def check_pending_orders(symbol, current_price):
    if current_price is None: return
    triggered_indices = []
    for i, order in enumerate(list(st.session_state.pending_orders)):
        if order.get('symbol') != symbol: continue
        is_long = order['type'] == 'Long'
        target_price = float(order['entry'])
        triggered = False
        # Limit order logic: long triggers when current_price <= target_price; short triggers when >=
        if is_long and current_price <= target_price: triggered = True
        if (not is_long) and current_price >= target_price: triggered = True

        if triggered:
            # execute
            new_pos = order.copy()
            new_pos['entry'] = float(current_price)
            new_pos['time'] = datetime.now().strftime("%m-%d %H:%M")
            st.session_state.positions.append(new_pos)
            triggered_indices.append(i)
            st.toast(f"⚡ 掛單成交：{order['symbol']} {order['type']} @ {fmt_price(current_price)}")

    # remove triggered by index (reverse order)
    for idx in sorted(triggered_indices, reverse=True):
        st.session_state.pending_orders.pop(idx)

    if triggered_indices:
        save_data_to_file()

# --- Backtest engine (Next Open execution) ---
def run_backtest_supertrend(df, initial_capital=10000):
    if df is None or len(df) < 100:
        return None, None
    capital = initial_capital
    position = 0
    entry_price = 0.0
    equity_curve = []
    trades = []
    # run from index 100 to len-2 to use next open
    for i in range(100, len(df)-1):
        curr = df.iloc[i]
        next_candle = df.iloc[i+1]
        timestamp = df.index[i]
        st_bull = curr['ST_Direction'] == 1
        st_bear = curr['ST_Direction'] == -1
        trend_bull = curr['Close'] > curr['EMA52']
        trend_bear = curr['Close'] < curr['EMA52']
        qqe_bull = (curr['RSI'] > 50) and (curr['Hist'] > 0)
        qqe_bear = (curr['RSI'] < 50) and (curr['Hist'] < 0)

        buy_signal = st_bull and trend_bull and qqe_bull
        sell_signal = st_bear and trend_bear and qqe_bear

        exit_long = (position == 1) and st_bear
        exit_short = (position == -1) and st_bull

        exec_price = next_candle['Open']
        exec_time = df.index[i+1]

        # exit first
        if exit_long and position == 1:
            pnl = (exec_price - entry_price) / entry_price * capital
            capital += pnl
            position = 0
            trades.append({'time': str(exec_time), 'type': '平多(次開)', 'price': float(exec_price), 'pnl': float(pnl), 'balance': float(capital)})
        elif exit_short and position == -1:
            pnl = (entry_price - exec_price) / entry_price * capital
            capital += pnl
            position = 0
            trades.append({'time': str(exec_time), 'type': '平空(次開)', 'price': float(exec_price), 'pnl': float(pnl), 'balance': float(capital)})

        # then entry
        if position == 0:
            if buy_signal:
                position = 1
                entry_price = exec_price
                trades.append({'time': str(exec_time), 'type': '做多', 'price': float(exec_price), 'balance': float(capital)})
            elif sell_signal:
                position = -1
                entry_price = exec_price
                trades.append({'time': str(exec_time), 'type': '做空', 'price': float(exec_price), 'balance': float(capital)})

        # equity estimation by close
        curr_equity = capital
        if position == 1:
            curr_equity += (curr['Close'] - entry_price) / entry_price * capital
        elif position == -1:
            curr_equity += (entry_price - curr['Close']) / entry_price * capital
        equity_curve.append({'time': str(timestamp), 'equity': float(curr_equity)})

    return pd.DataFrame(equity_curve), pd.DataFrame(trades)

# --- AI style "strategy" analyzer (live helper) ---
@st.cache_data(ttl=30)
def get_supertrend_strategy(symbol, current_interval_ui):
    df = get_chart_data(symbol, current_interval_ui)
    if df is None or len(df) < 50:
        return None
    last = df.iloc[-1]
    st_dir = "多頭 (綠)" if last['ST_Direction'] == 1 else "空頭 (紅)"
    ema_dir = "多頭 (價>EMA52)" if last['Close'] > last['EMA52'] else "空頭 (價<EMA52)"
    if last['RSI'] > 50 and last['Hist'] > 0:
        qqe_status = "🔵 藍柱 (多)"
    elif last['RSI'] < 50 and last['Hist'] < 0:
        qqe_status = "🔴 紅柱 (空)"
    else:
        qqe_status = "⚪ 灰色 (盤整)"

    score = 0
    score += 1 if last['ST_Direction'] == 1 else -1
    score += 1 if last['Close'] > last['EMA52'] else -1
    if last['RSI'] > 50 and last['Hist'] > 0:
        score += 1
    elif last['RSI'] < 50 and last['Hist'] < 0:
        score -= 1

    if score >= 3:
        direction = "強力做多 (Strong Buy)"
        action_msg = "🔥 全訊號共振！SuperTrend 翻綠 + 站上趨勢線 + 動能強勢，強力買入！"
    elif score <= -3:
        direction = "強力做空 (Strong Sell)"
        action_msg = "❄️ 全訊號共振！SuperTrend 翻紅 + 跌破趨勢線 + 動能轉弱，強力做空！"
    elif score == 1:
        direction = "偏多震盪"
        action_msg = "📈 趨勢偏多，但動能不足，等待回調或 QQE 轉藍。"
    elif score == -1:
        direction = "偏空震盪"
        action_msg = "📉 趨勢偏空，但短線有支撐，等待反彈或 QQE 轉紅。"
    else:
        direction = "觀望"
        action_msg = "🤖 AI 掃描中：目前訊號不共振，建議觀望。"

    curr_price = float(last['Close'])
    st_line = float(last['SuperTrend'])
    if score > 0:
        entry = curr_price
        sl = st_line
        tp = entry + (entry - sl) * 2
    else:
        entry = curr_price
        sl = st_line
        tp = entry - (sl - entry) * 2

    return {
        "direction": direction,
        "action_msg": action_msg,
        "st_dir": st_dir,
        "ema_dir": ema_dir,
        "qqe_status": qqe_status,
        "entry": float(entry),
        "tp": float(tp),
        "sl": float(sl),
        "df": df,
        "last_price": float(curr_price)
    }

# --- Callbacks / UI helpers ---
def on_select_change():
    raw_val = st.session_state.quick_select
    new_sym = raw_val.split(" ")[0]
    if st.session_state.market == "台股" and new_sym.isdigit():
        new_sym += ".TW"
    if st.session_state.market == "加密貨幣" and "-" not in new_sym and "USD" not in new_sym:
        new_sym += "-USD"
    st.session_state.chart_symbol = new_sym
    st.session_state.symbol_input = ""

def on_input_change():
    val = st.session_state.symbol_input.strip().upper()
    if not val: return
    if st.session_state.market == "台股" and val.isdigit():
        val += ".TW"
    if st.session_state.market == "加密貨幣" and "-" not in val and "USD" not in val:
        val += "-USD"
    st.session_state.chart_symbol = val

def jump_to_symbol(target_symbol):
    st.session_state.chart_symbol = target_symbol
    st.session_state.symbol_input = ""

# --- Dialog: manage position (Streamlit dialog) ---
@st.dialog("⚡ 倉位管理")
def manage_position_dialog(i, pos, current_price):
    st.markdown(f"**{pos.get('symbol','--')}**")
    try:
        entry = float(pos.get('entry', 0))
        lev = float(pos.get('lev', 1))
        margin = float(pos.get('margin', 0))
        d = 1 if pos.get('type') == 'Long' else -1
        u_pnl = margin * (((current_price - entry) / entry) * lev * d)
        roe_pct = (u_pnl / margin) * 100 if margin > 0 else 0.0
        color = "green" if u_pnl >= 0 else "red"
        st.markdown(f"未結盈虧: <span style='color:{color}; font-weight:bold'>${u_pnl:+.2f} ({roe_pct:+.2f}%)</span>", unsafe_allow_html=True)
    except:
        pass

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
            d = 1 if pos.get('type')=='Long' else -1
            if roe_tp > 0: new_tp = entry * (1 + (roe_tp / 100.0)/lev * d)
            if roe_sl > 0: new_sl = entry * (1 - (roe_sl / 100.0)/lev * d)

        if st.button("更新設定", key=f"btn_u_{i}", use_container_width=True):
            st.session_state.positions[i]['tp'] = new_tp
            st.session_state.positions[i]['sl'] = new_sl
            save_data_to_file()
            st.toast("✅ 已更新")
            st.rerun()

# --- Close position helper ---
def close_position(pos_index, percentage, reason, exit_price):
    if pos_index >= len(st.session_state.positions):
        return
    pos = st.session_state.positions[pos_index]
    close_ratio = percentage / 100.0
    margin = float(pos.get('margin', 0))
    close_margin = margin * close_ratio
    d = 1 if pos.get('type') == 'Long' else -1
    entry = float(pos.get('entry', 1))
    lev = float(pos.get('lev', 1))
    pnl = close_margin * (((exit_price - entry) / entry) * lev * d)
    roe_pct = (pnl / close_margin) * 100 if close_margin > 0 else 0.0
    # add funds back: principal + pnl
    st.session_state.balance += (close_margin + pnl)
    st.session_state.history.append({
        "時間": datetime.now().strftime("%m-%d %H:%M"),
        "幣種": pos.get('symbol'),
        "動作": f"平{percentage}%",
        "價格": float(exit_price),
        "盈虧": f"{pnl:+.2f} ({roe_pct:+.2f}%)",
        "原因": reason
    })
    if percentage == 100:
        st.session_state.positions.pop(pos_index)
    else:
        st.session_state.positions[pos_index]['margin'] -= close_margin
    save_data_to_file()

# --- Cancel pending order ---
def cancel_order(idx):
    if idx < len(st.session_state.pending_orders):
        st.session_state.pending_orders.pop(idx)
        save_data_to_file()
        st.toast("✅ 已撤銷掛單")

# --- Sidebar UI ---
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
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    # clear session
    st.session_state.clear()
    init_session_state()
    st.experimental_rerun()

# --- Main logic and UI rendering ---
with st.spinner(f"正在監控 {symbol} 即時數據... (每60秒刷新)"):
    ai_res = get_supertrend_strategy(symbol, interval_ui)

if ai_res is None:
    st.error(f"❌ 無法讀取 {symbol}，請確認代碼或網路連線。")
else:
    curr_price = ai_res['last_price']
    df_chart = ai_res['df']

    # check and execute pending orders for this symbol
    check_pending_orders(symbol, curr_price)

    # Top header: symbol & price
    c1, c2, c3 = st.columns([2, 1, 1])
    is_up = df_chart.iloc[-1]['Close'] >= df_chart.iloc[-1]['Open']
    p_color = "#00C853" if is_up else "#FF3D00"
    price_display = fmt_price(curr_price)

    c1.markdown(f"""
    <div style='display: flex; align-items: center; line-height: 1.5; padding-top: 5px; padding-bottom: 5px; white-space: nowrap; overflow: visible;'>
        <span style='font-size: 40px; font-weight: bold; margin-right: 15px; color: #ffffff;'>{symbol}</span>
        <span style='font-size: 30px; color: #cccccc; margin-right: 15px;'>({interval_ui})</span>
        <span style='font-size: 42px; color: {p_color}; font-weight: bold;'>{price_display}</span>
    </div>
    """, unsafe_allow_html=True)

    # Account stats
    balance = st.session_state.balance
    locked = get_locked_funds()
    available = balance - locked
    total_u_pnl = 0.0
    total_margin = 0.0
    for p in st.session_state.positions:
        try:
            cur = get_current_price(p['symbol'])
            if cur is not None:
                d = 1 if p['type']=='Long' else -1
                m = float(p.get('margin', 0))
                pnl = m * (((cur - p['entry'])/p['entry']) * p['lev'] * d)
                total_u_pnl += pnl
                total_margin += m
        except:
            pass
    total_roe = (total_u_pnl/total_margin)*100 if total_margin>0 else 0.0
    equity = balance + total_u_pnl

    m1, m2, m3 = st.columns(3)
    m1.metric("帳戶淨值 (Equity)", f"${equity:,.2f}")
    m2.metric("可用餘額 (Available)", f"${available:,.2f}")
    m3.metric("總未結盈虧 (PnL)", f"${total_u_pnl:+.2f}", delta=f"{total_roe:+.2f}%")

    st.divider()

    # Dashboard
    st.subheader("🧠 超級趨勢過濾系統 (SuperTrend + EMA52 + QQE)")
    col_k, col_s, col_a = st.columns([1, 1.5, 1.5])
    with col_k:
        st.markdown("#### 🔑 關鍵指標")
        st.write(f"**SuperTrend:** {ai_res['st_dir']}")
        st.write(f"**趨勢過濾 (EMA52):** {ai_res['ema_dir']}")
        st.write(f"**QQE 動能:** {ai_res['qqe_status']}")
    with col_s:
        st.markdown("#### 📢 戰情分析")
        st.info(ai_res['action_msg'])
    with col_a:
        st.markdown(f"#### 🚀 建議點位 ({ai_res['direction']})")
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("建議入場", fmt_price(ai_res['entry']))
        ac2.metric("SuperTrend 止損", fmt_price(ai_res['sl']), delta="SL", delta_color="inverse")
        ac3.metric("目標止盈 (1:2)", fmt_price(ai_res['tp']), delta="TP")

    st.divider()

    # Chart area
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='K線'), row=1, col=1)

    st_color = ['green' if d==1 else 'red' for d in df_chart['ST_Direction']]
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SuperTrend'], mode='markers', marker=dict(color=st_color, size=3), name='SuperTrend'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA52'], line=dict(width=2), name='EMA52 (趨勢線)'), row=1, col=1)

    # plot positions entry lines
    for pos in st.session_state.positions:
        if pos['symbol'] == symbol:
            fig.add_hline(y=pos['entry'], line_dash="dash", line_color="orange", annotation_text=f"持倉 {pos['type']}")

    colors = ['#2962FF' if h > 0 else '#FF1744' for h in df_chart['Hist']]
    fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Hist'], name='QQE 動能 (MACD)', marker_color=colors), row=2, col=1)

    fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0), dragmode='pan', title_text=f"{symbol} - {interval_ui} (SuperTrend Strategy)")
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Trading Tabs ---
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
                mode = st.radio("單位", ["價格", "ROE %"], horizontal=True, key="mode_tp_sl")
                rec_tp = ai_res['tp']; rec_sl = ai_res['sl']
                if mode == "價格":
                    t_tp = st.number_input("止盈價格", value=float(rec_tp), format="%.6f", key="t_tp_price")
                    t_sl = st.number_input("止損價格", value=float(rec_sl), format="%.6f", key="t_sl_price")
                else:
                    roe_tp = st.number_input("止盈 ROE %", value=0.0, key="t_tp_roe")
                    roe_sl = st.number_input("止損 ROE %", value=0.0, key="t_sl_roe")
                    t_tp, t_sl = 0.0, 0.0
                    d = 1 if "多" in trade_type else -1
                    if roe_tp > 0: t_tp = curr_price * (1 + (roe_tp/100.0)/lev * d)
                    if roe_sl > 0: t_sl = curr_price * (1 - (roe_sl/100.0)/lev * d)
                t_entry = st.number_input("掛單價格 (0=市價)", value=0.0, format="%.6f", key="t_entry_price")

            if st.button("🚀 下單 / 掛單", type="primary", use_container_width=True):
                final_entry = curr_price if float(t_entry) == 0 else float(t_entry)
                if mode == "ROE %":
                    d = 1 if "多" in trade_type else -1
                    if roe_tp > 0: t_tp = final_entry * (1 + (roe_tp/100.0)/lev * d)
                    if roe_sl > 0: t_sl = final_entry * (1 - (roe_sl/100.0)/lev * d)

                if amt > available:
                    st.error(f"可用餘額不足！ (可用: ${available:.2f})")
                else:
                    new_pos = {
                        "symbol": symbol, "type": "Long" if "多" in trade_type else "Short",
                        "entry": float(final_entry), "lev": float(lev), "margin": float(amt),
                        "tp": float(t_tp if t_tp is not None else 0.0), "sl": float(t_sl if t_sl is not None else 0.0),
                        "time": datetime.now().strftime("%m-%d %H:%M")
                    }
                    if float(t_entry) == 0.0:
                        # market execution: deduct margin immediately
                        st.session_state.positions.append(new_pos)
                        st.toast("✅ 市價成交！")
                        # when market executed, lock margin (already in positions)
                        st.session_state.balance -= amt
                    else:
                        st.session_state.pending_orders.append(new_pos)
                        st.toast("⏳ 掛單提交！等待價格觸發...")
                    save_data_to_file()
                    st.rerun()

        with col_info:
            st.info("☝️ 已自動填入 SuperTrend 止損建議")
            st.caption("止損點位設在超級趨勢線上，當趨勢反轉時自動離場。")

    with tab_orders:
        st.subheader("🔥 持倉中")
        if not st.session_state.positions:
            st.caption("無持倉")
        else:
            for i, pos in enumerate(st.session_state.positions):
                p_sym = pos['symbol']
                p_cur = get_current_price(p_sym)
                if p_cur is None: p_cur = curr_price if p_sym == symbol else None
                if p_cur is not None:
                    d = 1 if pos['type']=='Long' else -1
                    pnl = pos['margin'] * (((p_cur - pos['entry'])/pos['entry']) * pos['lev'] * d)
                    roe_pct = (pnl / pos['margin']) * 100 if pos['margin']>0 else 0.0
                    if roe_pct <= -100.0:
                        # forced liquidation
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

        st.subheader("⏳ 掛單中 (自動監控)")
        if not st.session_state.pending_orders:
            st.caption("無掛單")
        else:
            for i, ord in enumerate(st.session_state.pending_orders):
                o_sym = ord['symbol']
                c_btn, c_info, c_cnl = st.columns([1.5, 3, 1])
                c_btn.button(f"📊 {o_sym}", key=f"nav_o_{i}", on_click=jump_to_symbol, args=(o_sym,))
                c_info.markdown(f"{ord['type']} x{ord['lev']} @ <b>{fmt_price(ord['entry'])}</b>", unsafe_allow_html=True)
                if c_cnl.button("❌", key=f"cnl_{i}"):
                    cancel_order(i)
                    st.rerun()
                st.divider()

    with tab_history:
        st.subheader("📜 歷史戰績")
        if not st.session_state.history:
            st.info("暫無歷史紀錄")
        else:
            hist_df = pd.DataFrame(st.session_state.history)
            hist_df = hist_df.iloc[::-1]
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

    with tab_backtest:
        st.subheader(f"📈 {symbol} 歷史回測 (SuperTrend 戰法 - 無未來函數版)")
        st.caption("策略：SuperTrend + EMA52 + QQE | 進場邏輯：訊號確認後，於「次根K線開盤」進場 (Next Open)")
        if st.button("🚀 開始回測"):
            with st.spinner("正在模擬真實交易情境..."):
                eq_curve, trades_log = run_backtest_supertrend(df_chart, 10000)
            if eq_curve is not None and not eq_curve.empty:
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=eq_curve['time'], y=eq_curve['equity'], mode='lines', name='資金曲線'))
                fig_bt.update_layout(template="plotly_dark", title="回測資金增長", height=400)
                st.plotly_chart(fig_bt, use_container_width=True)

                initial = 10000
                final = eq_curve['equity'].iloc[-1]
                total_ret = (final - initial) / initial * 100
                win_count = len(trades_log[trades_log['pnl'] > 0]) if not trades_log.empty and 'pnl' in trades_log else 0
                total_trades = len(trades_log[trades_log['type'].str.contains('平')]) if not trades_log.empty else 0
                win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

                m1, m2, m3 = st.columns(3)
                m1.metric("期初本金", "$10,000")
                m2.metric("期末淨值", f"${final:,.2f}", delta=f"{total_ret:+.2f}%")
                m3.metric("勝率", f"{win_rate:.1f}%", f"共 {total_trades} 筆交易")

                if not trades_log.empty:
                    st.write("交易明細：")
                    st.dataframe(trades_log, use_container_width=True)
            else:
                st.warning("數據不足，無法回測")
