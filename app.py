import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os
import pytz # 引入 pytz 處理時區

# --- Page setup ---
st.set_page_config(page_title="全方位戰情室 AI (v71.0)", layout="wide", page_icon="🏦")
st.markdown("### 🏦 全方位戰情室 AI (v71.0 時區導航版)")

# --- [核心] NpEncoder ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

# --- Persistence System ---
DATA_FILE = "trade_data_v71.json"

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
        fi = getattr(ticker, 'fast_info', None)
        if fi and getattr(fi, 'last_price', None):
            return float(fi.last_price)
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except: pass
    return None

# --- [核心] 技術指標 ---
def calculate_indicators(df):
    if df is None or df.empty: return df
    df = df.copy()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA60'] = df['Close'].ewm(span=60).mean()
    df['EMA120'] = df['Close'].ewm(span=120).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean().replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(14).mean()

    df['max_roll'] = df['High'].rolling(10, center=True).max()
    df['min_roll'] = df['Low'].rolling(10, center=True).min()
    
    return df

# --- [核心] 多週期分析 ---
@st.cache_data(ttl=300)
def get_mtf_analysis(symbol):
    intervals = {"M": "1mo", "W": "1wk", "D": "1d"}
    periods = {"M": "5y", "W": "2y", "D": "1y"}
    data_store = {}
    
    for tf, interval in intervals.items():
        try:
            df = yf.Ticker(symbol).history(period=periods[tf], interval=interval)
            if not df.empty:
                data_store[tf] = calculate_indicators(df)
        except: pass
            
    if not data_store: return None

    scores = {"M": 0, "W": 0, "D": 0}
    trends = {}
    
    for tf, df in data_store.items():
        last = df.iloc[-1]
        score = 0
        trend_str = "震盪"
        
        if last['Close'] > last['EMA20'] > last['EMA60']:
            score += 2; trend_str = "多頭排列"
        elif last['Close'] < last['EMA20'] < last['EMA60']:
            score -= 2; trend_str = "空頭排列"
        else:
            if last['Close'] > last['EMA60']: trend_str = "多頭回調"
            elif last['Close'] < last['EMA60']: trend_str = "空頭反彈"
        
        if last['RSI'] > 70: score -= 0.5
        if last['RSI'] < 30: score += 0.5
        
        scores[tf] = score
        trends[tf] = trend_str

    total_score = (scores.get("M",0) * 0.3) + (scores.get("W",0) * 0.3) + (scores.get("D",0) * 0.4)
    
    direction = "觀望"
    if total_score >= 1.5: direction = "強力做多"
    elif total_score >= 0.5: direction = "嘗試做多"
    elif total_score <= -1.5: direction = "強力做空"
    elif total_score <= -0.5: direction = "嘗試做空"
    
    last_d = data_store.get("D", data_store.get("W", data_store.get("M")))
    if last_d is None: return None
    
    last_row = last_d.iloc[-1]
    curr_price = last_row['Close']
    atr = last_row.get('ATR', curr_price*0.02)
    
    if total_score > 0:
        entry = curr_price
        if last_row['EMA20'] < curr_price: entry = (curr_price + last_row['EMA20']) / 2
        tp = entry + (atr * 3)
        sl = entry - (atr * 1.5)
    else:
        entry = curr_price
        if last_row['EMA20'] > curr_price: entry = (curr_price + last_row['EMA20']) / 2
        tp = entry - (atr * 3)
        sl = entry + (atr * 1.5)

    main_chart_df = data_store.get("D")
    if main_chart_df is None:
        main_chart_df = data_store.get("W", data_store.get("M"))

    return {
        "score": total_score,
        "direction": direction,
        "trends": trends,
        "entry": float(entry),
        "tp": float(tp),
        "sl": float(sl),
        "last_price": float(curr_price),
        "df_chart": main_chart_df
    }

# --- Dialogs ---
@st.dialog("⚡ 倉位管理")
def manage_position_dialog(i, pos, current_price):
    st.markdown(f"**{pos.get('symbol','--')}** ({pos.get('type','--')} x{float(pos.get('lev',1)):.0f})")
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
        ord = st.session_state.pending_orders.pop(idx)
        st.session_state.balance += float(ord.get('margin', 0))
        save_data()
        st.toast("🗑️ 掛單已撤銷")

# --- Sidebar ---
st.sidebar.header("🎯 戰情室設定")
market = st.sidebar.radio("市場", ["加密貨幣", "美股", "台股"], index=0)
st.session_state.market = market

# [新增] 時區選擇
st.sidebar.markdown("---")
timezone_option = st.sidebar.selectbox("📅 K線時區", ["Asia/Taipei", "UTC", "America/New_York", "Asia/Tokyo"])
tz_map = {
    "Asia/Taipei": "Asia/Taipei",
    "UTC": "UTC",
    "America/New_York": "America/New_York",
    "Asia/Tokyo": "Asia/Tokyo"
}
selected_tz = tz_map[timezone_option]

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
if st.sidebar.button("🗑️ 清空數據 (重置)"):
    if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
    st.session_state.clear()
    st.rerun()

# --- Main Logic ---

with st.spinner(f"正在連線 {symbol} 分析中..."):
    ai_data = get_mtf_analysis(symbol)

if ai_data:
    curr_price = ai_data.get('last_price', 0.0)
    st.session_state.ai_entry = ai_data['entry']
    st.session_state.ai_tp = ai_data['tp']
    st.session_state.ai_sl = ai_data['sl']

    # Header
    c1, c2, c3 = st.columns([2, 1, 1])
    df = ai_data.get('df_chart')
    p_color = "#00C853"
    if df is not None and not df.empty:
        if df.iloc[-1]['Close'] < df.iloc[-1]['Open']: p_color = "#FF3D00"
    
    c1.markdown(f"<h1 style='margin:0'>{symbol} <span style='color:{p_color}'>${curr_price:,.2f}</span></h1>", unsafe_allow_html=True)
    c2.metric("可用餘額", f"${st.session_state.balance:,.2f}")
    
    total_u_pnl = 0
    for p in st.session_state.positions:
        try:
            cur = get_current_price(p['symbol'])
            if cur:
                d = 1 if p['type']=='Long' else -1
                total_u_pnl += p['margin'] * (((cur - p['entry'])/p['entry']) * p['lev'] * d)
        except: pass
    c3.metric("總未結盈虧", f"${total_u_pnl:+.2f}", delta_color="normal")

    # Chart
    if df is not None:
        # [新增] 時區轉換邏輯
        df_plot = df.copy()
        if df_plot.index.tz is None:
            df_plot.index = df_plot.index.tz_localize('UTC') # yfinance 默認 UTC 但有時無 tz info
        df_plot.index = df_plot.index.tz_convert(selected_tz)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA20'], line=dict(color='yellow', width=1), name='EMA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['EMA60'], line=dict(color='cyan', width=1), name='EMA60'), row=1, col=1)
        
        # 繪製持倉線 (需要轉換時間嗎？不需要，因為是水平線 y=價格)
        for pos in st.session_state.positions:
            if pos['symbol'] == symbol:
                fig.add_hline(y=pos['entry'], line_dash="dash", line_color="orange", annotation_text=f"持倉 {pos['type']}")
        
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], line=dict(color='violet', width=2), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        
        fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), dragmode='pan')
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 無法顯示圖表數據")

    # Trading Area
    tab_trade, tab_orders = st.tabs(["⚡ 下單交易", "📋 訂單管理"])
    
    with tab_trade:
        col_ctrl, col_info = st.columns([2, 1])
        with col_ctrl:
            c_t1, c_t2, c_t3 = st.columns(3)
            trade_type = c_t1.selectbox("方向", ["做多 (Long)", "做空 (Short)"], index=0 if "多" in ai_data['direction'] else 1)
            lev = c_t2.slider("槓桿", 1, 125, 20)
            amt = c_t3.number_input("本金 (U)", min_value=10.0, value=float(st.session_state.trade_amt_box))
            st.session_state.trade_amt_box = amt
            
            with st.expander("進階 (掛單/止盈損)", expanded=False):
                t_tp = st.number_input("止盈", value=st.session_state.ai_tp)
                t_sl = st.number_input("止損", value=st.session_state.ai_sl)
                t_entry = st.number_input("掛單價格 (0=市價)", value=0.0)

            if st.button("🚀 下單 / 掛單", type="primary", use_container_width=True):
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
                        st.toast(f"✅ 市價成交！ {symbol}")
                    else:
                        st.session_state.pending_orders.append(new_pos)
                        st.session_state.balance -= amt
                        st.toast(f"⏳ 掛單提交！ {symbol}")
                    save_data()
                    st.rerun()
        
        with col_info:
            st.info(f"**AI 建議**: {ai_data['direction']}\n\n信心分數: {ai_data['score']:.1f}")
            st.caption(f"趨勢: {ai_data['trends']['D']}")

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
                    clr = "#00C853" if pnl >= 0 else "#FF3D00"
                    
                    # [新增] 點擊跳轉按鈕佈局
                    c_btn, c_info, c_mng = st.columns([1.5, 3, 1])
                    
                    # 按下幣名按鈕 -> 切換圖表
                    if c_btn.button(f"📊 {p_sym}", key=f"nav_pos_{i}"):
                        st.session_state.chart_symbol = p_sym
                        st.rerun()
                    
                    c_info.markdown(f"""
                    <div style='font-size:14px'>
                        <b>{pos['type']} x{pos['lev']}</b> <span style='color:#aaa'>| 本金 ${pos['margin']:.0f}</span><br>
                        盈虧: <span style='color:{clr}; font-weight:bold'>${pnl:+.2f}</span>
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
                c_btn, c_info, c_cancel = st.columns([1.5, 3, 1])
                
                # 按下幣名按鈕 -> 切換圖表
                if c_btn.button(f"📊 {o_sym}", key=f"nav_ord_{i}"):
                    st.session_state.chart_symbol = o_sym
                    st.rerun()
                
                c_info.markdown(f"""
                <div style='font-size:13px; color:#ccc'>
                    {ord['type']} x{ord['lev']} @ <b>${fmt_price(ord['entry'])}</b><br>
                    <span style='color:#aaa'>本金 ${ord['margin']:.0f}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if c_cancel.button("❌", key=f"cnl_{i}"):
                    cancel_order(i)
                    st.rerun()
                st.divider()

else:
    st.error(f"❌ 數據連接失敗: {symbol}")
