import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- 1. 頁面設定 (必須在第一行) ---
st.set_page_config(page_title="交易戰情室 v31.0", layout="wide")
st.title("🛡️ 交易戰情室 AI (v31.0 絕對運行版)")

# --- 2. Session 初始化 ---
if 'balance' not in st.session_state: st.session_state.balance = 10000.0
if 'positions' not in st.session_state: st.session_state.positions = [] 
if 'history' not in st.session_state: st.session_state.history = []
if 'chart_symbol' not in st.session_state: st.session_state.chart_symbol = "BTC-USD"

# --- 3. 工具函數 (手寫指標，不依賴外部庫，避免報錯) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fmt_price(val):
    if val is None: return "N/A"
    if val < 0.01: return f"${val:.6f}"
    elif val < 20: return f"${val:.4f}"
    else: return f"${val:,.2f}"

# --- 4. 側邊欄設定 ---
st.sidebar.header("🎯 市場與標的")
user_input = st.sidebar.text_input("輸入代碼 (例: BTC-USD, NVDA, 2330.TW)", value=st.session_state.chart_symbol)
st.session_state.chart_symbol = user_input.strip().upper()
interval = st.sidebar.selectbox("週期", ["15m", "1h", "1d"], index=2)

# --- 5. 獲取數據 ---
@st.cache_data(ttl=60)
def get_data(symbol, interval):
    try:
        # 自動調整期間，避免數據過多卡死
        period_map = {"15m": "5d", "1h": "1mo", "1d": "1y"}
        df = yf.Ticker(symbol).history(period=period_map.get(interval, "1y"), interval=interval)
        
        if df.empty: return None
        
        # 計算指標
        df['RSI'] = calculate_rsi(df['Close'])
        
        # 簡單的均線趨勢
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        
        return df
    except Exception as e:
        return None

df = get_data(st.session_state.chart_symbol, interval)

# --- 6. 主畫面邏輯 ---
if df is not None:
    last = df.iloc[-1]
    curr_price = last['Close']
    
    # 顯示價格
    c1, c2, c3 = st.columns(3)
    c1.metric("當前價格", fmt_price(curr_price))
    c2.metric("RSI 強度", f"{last['RSI']:.1f}")
    
    trend = "盤整"
    if last['Close'] > last['MA20'] > last['MA60']: trend = "📈 多頭趨勢"
    elif last['Close'] < last['MA20'] < last['MA60']: trend = "📉 空頭趨勢"
    c3.metric("趨勢判斷", trend)

    # --- 7. 模擬交易功能 (側邊欄) ---
    st.sidebar.markdown("---")
    with st.sidebar.expander("🏦 模擬交易", expanded=True):
        st.write(f"💰 餘額: **${st.session_state.balance:,.2f}**")
        
        # 開倉介面
        with st.form("order_form"):
            side = st.selectbox("方向", ["做多 (Long)", "做空 (Short)"])
            lev = st.number_input("槓桿", 1, 125, 10)
            amt = st.number_input("本金", 1.0, float(st.session_state.balance), 1000.0)
            tp = st.number_input("止盈價 (選填)", 0.0)
            sl = st.number_input("止損價 (選填)", 0.0)
            submitted = st.form_submit_button("🚀 下單")
            
            if submitted:
                new_pos = {
                    "symbol": st.session_state.chart_symbol,
                    "type": "Long" if "多" in side else "Short",
                    "entry": curr_price,
                    "lev": lev,
                    "margin": amt,
                    "tp": tp,
                    "sl": sl,
                    "time": datetime.now().strftime("%m-%d %H:%M")
                }
                st.session_state.positions.append(new_pos)
                st.session_state.balance -= amt
                st.rerun()

        # 持倉列表
        if st.session_state.positions:
            st.markdown("---")
            for i, pos in enumerate(st.session_state.positions):
                # 簡易損益計算
                p_now = curr_price if pos['symbol'] == st.session_state.chart_symbol else pos['entry'] # 簡化：非當前幣種不跳動
                
                direction = 1 if pos['type'] == "Long" else -1
                pnl_pct = ((p_now - pos['entry']) / pos['entry']) * pos['lev'] * direction * 100
                pnl_u = pos['margin'] * pnl_pct / 100
                
                st.caption(f"{pos['symbol']} ({pos['type']} {pos['lev']}x)")
                col_a, col_b = st.columns(2)
                col_a.write(f"未實現: {pnl_u:+.2f} U")
                if col_b.button("平倉", key=f"close_{i}"):
                    st.session_state.balance += (pos['margin'] + pnl_u)
                    st.session_state.positions.pop(i)
                    st.rerun()
                st.divider()

    # --- 8. 繪圖 ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='價格', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='yellow', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='MA60', line=dict(color='cyan', width=1)))
    fig.update_layout(height=600, template="plotly_dark", title=f"{st.session_state.chart_symbol} 走勢圖")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error(f"找不到 {st.session_state.chart_symbol} 的數據，請確認代碼正確 (例如 BTC-USD, 2330.TW)")
