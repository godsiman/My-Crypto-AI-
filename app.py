import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from datetime import datetime

# --- 1. 頁面設定 (必須在第一行) ---
st.set_page_config(page_title="全方位戰情室 AI (v31.0)", layout="wide")
st.title("🏦 全方位戰情室 AI (v31.0 基金經理人部署版)")

# --- 2. Session 初始化 ---
if 'balance' not in st.session_state: st.session_state.balance = 10000.0
if 'positions' not in st.session_state: st.session_state.positions = [] 
if 'history' not in st.session_state: st.session_state.history = []
# 控制當前顯示的幣種
if 'chart_symbol' not in st.session_state: st.session_state.chart_symbol = "BTC-USD"

# --- 3. 工具函數 ---
def fmt_price(val):
    """ 智能價格格式化 """
    if val is None: return "N/A"
    if val < 0.01: return f"${val:.6f}"
    elif val < 20: return f"${val:.4f}"
    else: return f"${val:,.2f}"

def get_current_price(sym):
    """ 獲取最新價格 (用於後台計算損益) """
    try:
        ticker = yf.Ticker(sym)
        if hasattr(ticker, 'fast_info') and ticker.fast_info.last_price:
            return ticker.fast_info.last_price
        # 回退方案
        hist = ticker.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
    except:
        return None
    return None

# --- 4. 側邊欄設定 ---
st.sidebar.header("🎯 市場與標的")

# 智能搜尋框 (預設值連動 Session)
user_symbol_input = st.sidebar.text_input("🔍 快速搜尋 / 代碼輸入", value=st.session_state.chart_symbol)

def smart_parse(s):
    s = s.strip().upper()
    us_stocks = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "PLTR", "MSTR", "COIN", "GOOG", "META", "AMZN", "NFLX", "INTC", "SMCI"]
    if "-" in s or "." in s: return s
    if s.isdigit(): return f"{s}.TW"
    if s in us_stocks: return s
    return f"{s}-USD"

symbol = smart_parse(user_symbol_input)

# 更新 Session
if symbol != st.session_state.chart_symbol:
    st.session_state.chart_symbol = symbol

interval_ui = st.sidebar.radio("K 線週期", ["15分鐘", "1小時", "4小時", "日線"], index=3)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👁️ 視覺化開關")
show_six = st.sidebar.checkbox("顯示 六道乾坤帶", value=True)
show_zigzag = st.sidebar.checkbox("顯示 ZigZag 結構", value=True)
show_fvg = st.sidebar.checkbox("顯示 FVG 缺口", value=True)
show_fib = st.sidebar.checkbox("顯示 Fib 止盈", value=True)
show_div = st.sidebar.checkbox("顯示 RSI 背離", value=True)

if st.sidebar.button("🔄 強制刷新盤勢"):
    st.cache_data.clear()

# --- 5. 核心數據處理 ---
def get_params(ui_selection):
    if "15分鐘" in ui_selection: return "5d", "15m"
    elif "1小時" in ui_selection: return "1mo", "1h"
    elif "4小時" in ui_selection: return "6mo", "1h"
    else: return "2y", "1d"

period, interval = get_params(interval_ui)

@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty: return None
        if interval == "1h" and "6mo" in period: 
            logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df = df.resample('4h').apply(logic).dropna()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA60'] = df['Close'].ewm(span=60, adjust=False).mean()
        df['EMA120'] = df['Close'].ewm(span=120, adjust=False).mean()
        df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(14).mean()
        return df
    except: return None

# --- 6. 指標演算法 ---
def calculate_zigzag(df, depth=12):
    try:
        df['max_roll'] = df['High'].rolling(window=depth, center=True).max()
        df['min_roll'] = df['Low'].rolling(window=depth, center=True).min()
        pivots = []
        last_type = None
        for i in range(len(df)):
            if df['High'].iloc[i] == df['max_roll'].iloc[i]:
                if last_type != 'high': pivots.append({'idx': df.index[i], 'val': df['High'].iloc[i], 'type': 'high'}); last_type='high'
                elif df['High'].iloc[i] > pivots[-1]['val']: pivots[-1] = {'idx': df.index[i], 'val': df['High'].iloc[i], 'type': 'high'}
            elif df['Low'].iloc[i] == df['min_roll'].iloc[i]:
                if last_type != 'low': pivots.append({'idx': df.index[i], 'val': df['Low'].iloc[i], 'type': 'low'}); last_type='low'
                elif df['Low'].iloc[i] < pivots[-1]['val']: pivots[-1] = {'idx': df.index[i], 'val': df['Low'].iloc[i], 'type': 'low'}
        return pivots
    except: return []

def calculate_fvg(df):
    try:
        bull, bear = [], []
        h, l, c, t = df['High'].values, df['Low'].values, df['Close'].values, df.index
        start = max(2, len(df)-300)
        for i in range(start, len(df)):
            if l[i] > h[i-2] and c[i-1] > h[i-2]: 
                bull.append({'start': t[i-2], 'top': l[i], 'bottom': h[i-2], 'active': True})
            if h[i] < l[i-2] and c[i-1] < l[i-2]: 
                bear.append({'start': t[i-2], 'top': l[i-2], 'bottom': h[i], 'active': True})
            for f in bull: 
                if f['active'] and l[i] < f['top']: f['active'] = False
            for f in bear:
                if f['active'] and h[i] > f['bottom']: f['active'] = False
        return [f for f in bull if f['active']], [f for f in bear if f['active']]
    except: return [], []

def detect_div(df):
    try:
        rsi, close = df['RSI'].values, df['Close'].values
        highs = argrelextrema(rsi, np.greater, order=5)[0]
        lows = argrelextrema(rsi, np.less, order=5)[0]
        bull, bear = [], []
        if len(lows)>=2:
            for i in range(len(lows)-1):
                curr, prev = lows[i+1], lows[i]
                if close[curr]<close[prev] and rsi[curr]>rsi[prev] and rsi[curr]<50: bull.append(df.index[curr])
        if len(highs)>=2:
            for i in range(len(highs)-1):
                curr, prev = highs[i+1], highs[i]
                if close[curr]>close[prev] and rsi[curr]<rsi[prev] and rsi[curr]>50: bear.append(df.index[curr])
        return bull, bear
    except: return [], []

def calculate_score_v17(pivots, last, df, bull_fvg, bear_fvg, bull_div, bear_div):
    score = 0
    struct_txt = "盤整"
    if len(pivots) >= 4:
        vh = [p['val'] for p in pivots if p['type']=='high']
        vl = [p['val'] for p in pivots if p['type']=='low']
        if len(vh)>=2 and len(vl)>=2:
            if vh[-1]>vh[-2] and vl[-1]>vl[-2]: score += 3; struct_txt="多頭 (+3)"
            elif vh[-1]<vh[-2] and vl[-1]<vl[-2]: score -= 3; struct_txt="空頭 (-3)"
    six_txt = "盤整"
    ema20, ema60, ema120 = last['EMA20'], last['EMA60'], last['EMA120']
    if last['Close'] > ema20 > ema60 > ema120: score += 2; six_txt="順勢多 (+2)"
    elif last['Close'] < ema20 < ema60 < ema120: score -= 2; six_txt="順勢空 (-2)"
    elif last['Close'] > ema60: score += 1; six_txt="偏多 (+1)"
    elif last['Close'] < ema60: score -= 1; six_txt="偏空 (-1)"
    fvg_txt = "無"
    if bull_fvg and (last['Close']-bull_fvg[-1]['top'])/last['Close']<0.02: score += 2; fvg_txt="支撐位 (+2)"
    elif bear_fvg and (bear_fvg[-1]['bottom']-last['Close'])/last['Close']<0.02: score -= 2; fvg_txt="壓力位 (-2)"
    div_txt = "無"
    if bull_div and (df.index[-1]-bull_div[-1]).days < 3: score += 2; div_txt="底背離 (+2)"
    elif bear_div and (df.index[-1]-bear_div[-1]).days < 3: score -= 2; div_txt="頂背離 (-2)"
    rsi_txt = "中性"
    if last['RSI'] < 30: score += 1; rsi_txt="超賣 (+1)"
    elif last['RSI'] > 70: score -= 1; rsi_txt="超買 (-1)"
    return score, struct_txt, six_txt, fvg_txt, div_txt, rsi_txt

def generate_ai_report(symbol, price, score, struct, six, fvg, div, rsi_txt, buy_sl, sell_sl, tp1, tp2, entry_zone, risk_warning):
    report = f"**【市場掃描】** {symbol} 現價 **{fmt_price(price)}**。\\n"
    abs_score = abs(score)
    direction = "做多" if score > 0 else "做空"
    color_emoji = "🟢" if score > 0 else "🔴"
    if risk_warning: report += f"⚠️ **風險提示**：{risk_warning}"
    elif abs_score >= 8: report += f"🔥 **強力{direction}訊號 (評分: {score}/10)**！"
    elif abs_score >= 5: report += f"{color_emoji} **偏向{direction} (評分: {score}/10)**。"
    else: report += f"⚖️ **盤整觀望 (評分: {score}/10)**。"
    report += "\\n\\n**【交易計畫】**"
    if risk_warning and "破" in risk_warning: report += f"\\n⛔ 結構已破壞，暫無交易建議。"
    elif score >= 0: report += f"\\n🛒 **建議入場**: **{entry_zone}**\\n🎯 **止盈 TP1**: **{fmt_price(tp1)}**\\n🛡️ **止損 SL**: **{fmt_price(buy_sl)}**"
    else: report += f"\\n🛒 **建議空點**: **{entry_zone}**\\n🎯 **止盈 TP1**: **{fmt_price(tp1)}**\\n🛡️ **止損 SL**: **{fmt_price(sell_sl)}**"
    return report

# --- 7. 平倉函數 ---
def close_position(pos_index, percentage=100, reason="手動平倉", exit_price=0):
    if pos_index >= len(st.session_state.positions): return
    pos = st.session_state.positions[pos_index]
    
    close_margin = pos['margin'] * (percentage / 100)
    direction = 1 if pos['type'] == 'Long' else -1
    pnl_pct = ((exit_price - pos['entry']) / pos['entry']) * pos['lev'] * direction * 100
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
    else:
        st.session_state.positions[pos_index]['margin'] -= close_margin

# --- 主程式 ---
df = get_data(symbol, period, interval)

if df is not None:
    last = df.iloc[-1]
    curr_price = last['Close']
    
    # ---------------------------
    # 🏦 基金經理人專區 (Sidebar)
    # ---------------------------
    st.sidebar.markdown("---")
    with st.sidebar.expander("🏦 我的錢包與持倉", expanded=True):
        st.metric("💰 總資產 (USDT)", f"${st.session_state.balance:,.2f}")
        
        # 持倉列表
        if st.session_state.positions:
            st.markdown("##### 🔥 持倉列表")
            for i, pos in enumerate(st.session_state.positions):
                # 全域監控：抓取該倉位的即時價格
                live_price = curr_price if pos['symbol'] == symbol else get_current_price(pos['symbol'])
                
                if live_price:
                    direction = 1 if pos['type'] == 'Long' else -1
                    pnl_pct = ((live_price - pos['entry']) / pos['entry']) * pos['lev'] * direction * 100
                    pnl_usdt = pos['margin'] * (pnl_pct / 100)
                    
                    if pos['type'] == 'Long': liq = pos['entry'] * (1 - 1/pos['lev'])
                    else: liq = pos['entry'] * (1 + 1/pos['lev'])
                    
                    with st.container():
                        # 標題 + 跳轉按鈕
                        c_title, c_jump = st.columns([3, 1])
                        c_title.markdown(f"**#{i+1} {pos['symbol']}**")
                        if pos['symbol'] != symbol:
                            if c_jump.button("🔍", key=f"jump_{i}"):
                                st.session_state.chart_symbol = pos['symbol']
                                st.rerun()
                        
                        c1, c2 = st.columns(2)
                        c1.write(f"{pos['type']} {pos['lev']}x")
                        # 損益顏色與小數位修正
                        color = "green" if pnl_usdt >= 0 else "red"
                        c2.markdown(f":{color}[**{pnl_usdt:+.2f} U**]")
                        
                        st.caption(f"均價: {fmt_price(pos['entry'])}")
                        
                        # 自動平倉檢查
                        reason = None
                        if (pos['type']=='Long' and live_price <= liq) or (pos['type']=='Short' and live_price >= liq): reason="💀 爆倉"
                        elif pos['tp']>0 and ((pos['type']=='Long' and live_price >= pos['tp']) or (pos['type']=='Short' and live_price <= pos['tp'])): reason="🎯 止盈"
                        elif pos['sl']>0 and ((pos['type']=='Long' and live_price <= pos['sl']) or (pos['type']=='Short' and live_price >= pos['sl'])): reason="🛡️ 止損"
                        
                        if reason: 
                            close_position(i, 100, reason, live_price)
                            st.rerun()

                        if st.button(f"平倉", key=f"close_{i}"):
                            close_position(i, 100, "手動", live_price)
                            st.rerun()
                        st.divider()
                else:
                    st.warning(f"讀取中 {pos['symbol']}...")
        else:
            st.info("空倉中...")

        # 開倉區
        st.markdown("##### 🚀 開立新倉位")
        col_s1, col_s2 = st.columns(2)
        trade_type = col_s1.selectbox("方向", ["🟢 做多 (Long)", "🔴 做空 (Short)"], key="new_side")
        leverage = col_s2.number_input("槓桿", 1, 125, 20, key="new_lev")
        
        # 資金全開
        principal = st.number_input("本金 (U)", 10.0, float(st.session_state.balance), 1000.0, key="new_amt")
        
        with st.expander("進階設定 (TP/SL)"):
            set_tp = st.number_input("止盈 TP", value=0.0, format="%.4f", key="new_tp")
            set_sl = st.number_input("止損 SL", value=0.0, format="%.4f", key="new_sl")
        
        if st.button("確認下單", type="primary"):
            if principal > st.session_state.balance:
                st.error("餘額不足！")
            else:
                new_pos = {
                    "symbol": symbol,
                    "type": "Long" if "做多" in trade_type else "Short",
                    "entry": curr_price,
                    "lev": leverage,
                    "margin": principal,
                    "tp": set_tp,
                    "sl": set_sl,
                    "time": datetime.now().strftime('%m-%d %H:%M')
                }
                st.session_state.positions.append(new_pos)
                st.session_state.balance -= principal
                st.rerun()

        if st.session_state.history:
            with st.sidebar.expander("📜 歷史交易"):
                hist_df = pd.DataFrame(st.session_state.history[::-1])
                st.dataframe(hist_df[['幣種', '獲利%', '損益(U)', '時間']], hide_index=True)

    # --- 主分析邏輯 ---
    pivots = calculate_zigzag(df)
    bull_fvg, bear_fvg = calculate_fvg(df)
    bull_div, bear_div = detect_div(df)
    score, struct_t, six_t, fvg_t, div_t, rsi_t = calculate_score_v17(pivots, last, df, bull_fvg, bear_fvg, bull_div, bear_div)

    atr = last['ATR'] if not pd.isna(last['ATR']) else last['Close']*0.02
    pivot_lows = [p['val'] for p in pivots if p['type']=='low']
    pivot_highs = [p['val'] for p in pivots if p['type']=='high']
    buy_sl = pivot_lows[-1] if pivot_lows else last['Close'] - 2*atr
    sell_sl = pivot_highs[-1] if pivot_highs else last['Close'] + 2*atr
    
    if buy_sl >= last['Close']: buy_sl = last['Close'] - 2*atr 
    if sell_sl <= last['Close']: sell_sl = last['Close'] + 2*atr

    tp1 = 0; tp2 = 0; entry_zone = "現價"; risk_warning = "" 
    if len(pivots) >= 2:
        lh = [p['val'] for p in pivots if p['type']=='high'][-1]
        ll = [p['val'] for p in pivots if p['type']=='low'][-1]
        diff = abs(lh - ll)
        if score >= 0: 
            tp1 = lh; tp2 = ll + diff * 1.618
            fib_low = ll + diff * 0.382; fib_high = ll + diff * 0.618
            if last['Close'] < fib_high and last['Close'] > buy_sl: 
                entry_zone = f"{fmt_price(last['Close'])} (現價優)"
            else: 
                entry_zone = f"{fmt_price(fib_low)} ~ {fmt_price(fib_high)}"
            if last['Close'] >= tp1:
                tp1 = ll + diff * 1.272; tp2 = ll + diff * 1.618; risk_warning = "價格創高，止盈上移"
            elif last['Close'] < buy_sl: risk_warning = "❌ 結構破壞 (跌破止損)"; score = 0
        else:
            tp1 = ll; tp2 = lh - diff * 1.618
            fib_low = lh - diff * 0.618; fib_high = lh - diff * 0.382
            if last['Close'] > fib_low and last['Close'] < sell_sl: 
                entry_zone = f"{fmt_price(last['Close'])} (現價優)"
            else: 
                entry_zone = f"{fmt_price(fib_low)} ~ {fmt_price(fib_high)}"
            if last['Close'] <= tp1:
                tp1 = lh - diff * 1.272; tp2 = lh - diff * 1.618; risk_warning = "價格創低，止盈下移"
            elif last['Close'] > sell_sl: risk_warning = "❌ 結構破壞 (突破止損)"; score = 0

    st.info("🛡️ **AI 實戰風控報告**")
    st.markdown(generate_ai_report(symbol, last['Close'], score, struct_t, six_t, fvg_t, div_t, rsi_t, buy_sl, sell_sl, tp1, tp2, entry_zone, risk_warning))
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    action_label = "觀望"
    if risk_warning and "破" in risk_warning: action_label = "⛔ " + risk_warning; score_display = "N/A"
    else:
        if score >= 8: action_label = "🔥 強力買進"
        elif score >= 5: action_label = "🟢 買進"
        elif score <= -8: action_label = "💀 強力賣出"
        elif score <= -5: action_label = "🔴 賣出"
        score_display = f"{score}/10"
    
    m1.metric("AI 評級", score_display, action_label)
    m2.metric("建議入場", entry_zone.split("~")[0] if "~" in entry_zone else "現價", "校正後")
    if score >= 0:
        m3.metric("止盈 TP1", fmt_price(tp1), "目標")
        m4.metric("止損 SL", fmt_price(buy_sl), "防守", delta_color="inverse")
    else:
        m3.metric("止盈 TP1", fmt_price(tp1), "目標", delta_color="inverse")
        m4.metric("止損 SL", fmt_price(sell_sl), "防守", delta_color="normal")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='價格', line=dict(color='white', width=2)), row=1, col=1)
    if show_six:
        ribbon_color = 'rgba(0, 255, 0, 0.6)' if last['EMA20'] > last['EMA60'] else 'rgba(255, 0, 0, 0.6)'
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='趨勢帶', line=dict(color=ribbon_color, width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA60'], name='生命線', line=dict(color='rgba(255,255,255,0.5)', width=1)), row=1, col=1)
    if show_fvg:
        for f in bull_fvg: fig.add_shape(type="rect", x0=f['start'], x1=df.index[-1], y0=f['bottom'], y1=f['top'], fillcolor="rgba(0,255,0,0.4)", line_width=0, row=1, col=1)
        for f in bear_fvg: fig.add_shape(type="rect", x0=f['start'], x1=df.index[-1], y0=f['bottom'], y1=f['top'], fillcolor="rgba(255,0,0,0.4)", line_width=0, row=1, col=1)
    if show_zigzag and pivots:
        px = [p['idx'] for p in pivots]; py = [p['val'] for p in pivots]
        fig.add_trace(go.Scatter(x=px, y=py, mode='lines+markers', name='ZigZag', line=dict(color='orange', width=3), marker_size=6), row=1, col=1)
        for i in range(2, len(pivots)):
            p = pivots[i]; prev = pivots[i-2]
            txt = ("HH" if p['val']>prev['val'] else "LH") if p['type']=='high' else ("HL" if p['val']>prev['val'] else "LL")
            clr = 'red' if p['type']=='high' else '#00FF00'
            fig.add_annotation(x=p['idx'], y=p['val'], text=f"<b>{txt}</b>", showarrow=False, font=dict(color=clr, size=14), yshift=20 if p['type']=='high' else -20, row=1, col=1)
    if show_fib and tp1 > 0:
        fig.add_hline(y=tp1, line_dash="dash", line_color="yellow", annotation_text=f"TP1 {fmt_price(tp1)}", row=1, col=1)
        fig.add_hline(y=tp2, line_dash="dash", line_color="#00FF00", annotation_text=f"TP2 {fmt_price(tp2)}", row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='cyan', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    fig.update_layout(title=f"{symbol} 實戰分析圖", template="plotly_dark", height=800)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error(f"❌ 找不到 {symbol} 數據。")
