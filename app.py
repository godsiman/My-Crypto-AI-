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
st.set_page_config(page_title="全方位戰情室 AI (v84.0)", layout="wide", page_icon="🏦")
st.markdown("### 🏦 全方位戰情室 AI (v84.0 中文選單版)")

# --- [核心] NpEncoder ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

# --- Persistence ---
DATA_FILE = "trade_data_v84.json"

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
        except:
            st.session_state.balance = 10000.0

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
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA60'] = df['Close'].ewm(span=60).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean().replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(14).mean()
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

# --- AI Strategy ---
@st.cache_data(ttl=300)
def get_ai_strategy(symbol):
    intervals = {"M": "1mo", "W": "1wk", "D": "1d"}
    periods = {"M": "5y", "W": "2y", "D": "1y"}
    scores = {"M": 0, "W": 0, "D": 0}
    trends = {}
    last_price = 0
    
    for tf, interval in intervals.items():
        try:
            df = yf.Ticker(symbol).history(period=periods[tf], interval=interval)
            if not df.empty:
                df = calculate_indicators(df)
                last = df.iloc[-1]
                last_price = last['Close']
                
                score = 0
                trend = "震盪"
                if last['Close'] > last['EMA20'] > last['EMA60']: score += 2; trend = "多頭"
                elif last['Close'] < last['EMA20'] < last['EMA60']: score -= 2; trend = "空頭"
                if last['RSI'] > 70: score -= 0.5
                if last['RSI'] < 30: score += 0.5
                scores[tf] = score
                trends[tf] = trend
        except: pass
    
    total_score = scores.get("M",0)*0.3 + scores.get("W",0)*0.3 + scores.get("D",0)*0.4
    direction = "觀望"
    if total_score >= 1.5: direction = "強力做多"
    elif total_score >= 0.5: direction = "嘗試做多"
    elif total_score <= -1.5: direction = "強力做空"
    elif total_score <= -0.5: direction = "嘗試做空"

    return {"direction": direction, "score": total_score, "trends": trends, "last_price": last_price}

# --- Callbacks ---
def on_select_change():
    # [更新] 這裡會收到 "BTC-USD 比特幣"，需要用 split 切割出代碼
    raw_val = st.session_state.quick_select
    new_sym = raw_val.split(" ")[0] # 取空格前的部分
    
    # 再次確認後綴 (雖然清單已經有了，但防呆)
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
    except: 
        entry = 0; lev = 1; pos_type = 'Long'

    tab_close, tab_tpsl = st.tabs(["平倉", "止盈止損"])
    
    with tab_close:
        ratio = st.radio("平倉 %", [25,50,75,100], 3, horizontal=True, key=f"dr_{i}")
        if st.button("確認平倉", key=f"btn_c_{i}", type="primary", use_container_width=True):
            close_position(i, ratio, "手動", current_price)
            st.rerun()

    with tab_tpsl:
        mode = st.radio("設定模式", ["價格", "ROE %"], horizontal=True, key=f"m_mode_{i}")
        
        current_tp = float(pos.get('tp', 0))
        current_sl = float(pos.get('sl', 0))
        
        new_tp = current_tp
        new_sl = current_sl

        if mode == "價格":
            c1, c2 = st.columns(2)
            new_tp = c1.number_input("TP 價格", value=current_tp, key=f"ntp_p_{i}", format="%.6f")
            new_sl = c2.number_input("SL 價格", value=current_sl, key=f"nsl_p_{i}", format="%.6f")
        else:
            c1, c2 = st.columns(2)
            roe_tp = c1.number_input("止盈 % (例: 50)", value=0.0, key=f"ntp_r_{i}")
            roe_sl = c2.number_input("止損 % (例: 20)", value=0.0, key=f"nsl_r_{i}")
            
            calc_tp = 0.0
            calc_sl = 0.0
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
            st.toast("✅ 止盈止損已更新")
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
    st.session_state.balance += (close_margin + pnl)
    
    st.session_state.history.append({
        "時間": datetime.now().strftime("%m-%d %H:%M"),
        "幣種": pos.get('symbol'),
        "動作": f"平{percentage}%",
        "價格": exit_price,
        "盈虧": round(pnl, 2),
        "原因": reason
    })

    if percentage == 100:
        st.session_state.positions.pop(pos_index)
    else:
        st.session_state.positions[pos_index]['margin'] -= close_margin
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

# [更新] 帶中文名稱的選單清單
if market == "加密貨幣":
    targets = [
        "BTC-USD 比特幣", 
        "ETH-USD 以太坊", 
        "SOL-USD 索拉納", 
        "DOGE-USD 狗狗幣", 
        "XRP-USD 瑞波幣", 
        "BNB-USD 幣安幣", 
        "DNX-USD Dynex"
    ]
elif market == "美股":
    targets = [
        "NVDA 輝達", 
        "TSLA 特斯拉", 
        "AAPL 蘋果", 
        "MSFT 微軟", 
        "AMD 超微", 
        "COIN Coinbase"
    ]
else:
    targets = [
        "2330.TW 台積電", 
        "2317.TW 鴻海", 
        "2454.TW 聯發科", 
        "2603.TW 長榮", 
        "0050.TW 元大台灣50"
    ]

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

with st.spinner(f"正在分析 {symbol} ..."):
    ai_res = get_ai_strategy(symbol)
    df_chart = get_chart_data(symbol, interval_ui)

if ai_res and df_chart is not None:
    curr_price = ai_res['last_price']
    last_row = df_chart.iloc[-1]
    atr = last_row.get('ATR', curr_price * 0.02)
    
    if "多" in ai_res['direction']:
        rec_entry = curr_price
    else:
        rec_entry = curr_price

    # --- Header UI ---
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
    
    equity = balance + total_u_pnl
    total_roe = (total_u_pnl / total_margin_used * 100) if total_margin_used > 0 else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("錢包餘額", f"${balance:,.2f}")
    m2.metric("可用餘額", f"${available:,.2f}")
    m3.metric("總未結盈虧", f"${total_u_pnl:+.2f}", delta=f"{total_roe:+.2f}%")

    st.divider()

    # --- Chart ---
    df_plot = df_chart.copy()
    if df_plot.index.tz is None: df_plot.index = df_plot.index.tz_localize('UTC')
    df_plot.index = df_plot.index.tz_convert('Asia/Taipei')

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA20'], line=dict(color='yellow', width=1), name='EMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA60'], line=dict(color='cyan', width=1), name='EMA60'), row=1, col=1)
    for pos in st.session_state.positions:
        if pos['symbol'] == symbol:
            fig.add_hline(y=pos['entry'], line_dash="dash", line_color="orange", annotation_text=f"持倉 {pos['type']}")
    
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], line=dict(color='violet', width=2), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    fig.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), dragmode='pan', title_text=f"{symbol} - {interval_ui} (台北時間)")
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Trading ---
    tab_trade, tab_orders = st.tabs(["⚡ 下單交易", "📋 訂單管理"])
    
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
                base_price = curr_price
                
                if mode == "價格":
                    t_tp = st.number_input("止盈價格", value=0.0, format="%.6f")
                    t_sl = st.number_input("止損價格", value=0.0, format="%.6f")
                else:
                    roe_tp = st.number_input("止盈 ROE %", value=0.0)
                    roe_sl = st.number_input("止損 ROE %", value=0.0)
                    
                    t_tp, t_sl = 0.0, 0.0
                    if roe_tp > 0:
                        direction = 1 if "多" in trade_type else -1
                        t_tp = base_price * (1 + (roe_tp / 100) / lev * direction)
                        st.caption(f"預估止盈: {fmt_price(t_tp)}")
                    if roe_sl > 0:
                        direction = 1 if "多" in trade_type else -1
                        t_sl = base_price * (1 - (roe_sl / 100) / lev * direction)
                        st.caption(f"預估止損: {fmt_price(t_sl)}")

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
            st.info(f"**AI 建議**: {ai_res['direction']}\n\n信心分數: {ai_res['score']:.1f}")
            st.caption(f"日線趨勢: {ai_res['trends']['D']}")

    with tab_orders:
        st.subheader("🔥 持倉中")
        if not st.session_state.positions:
            st.caption("無持倉")
        else:
            for i, pos in enumerate(st.session_state.positions):
                p_sym = pos['symbol']
                p_cur = get_current_price(p_sym)
                if p_cur:
                    d = 1 if pos['type']=='Long' else -1
                    pnl = pos['margin'] * (((p_cur - pos['entry'])/pos['entry']) * pos['lev'] * d)
                    roe_pct = (pnl / pos['margin']) * 100
                    
                    # [自動爆倉]
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
                    
                    if c_mng.button("⚙️", key=f"mng_{i}"):
                        manage_position_dialog(i, pos, p_cur)
                    st.divider()

        st.subheader("⏳ 掛單中")
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

else:
    st.error(f"❌ 無法讀取 {symbol}，請確認代碼或網路連線。")
