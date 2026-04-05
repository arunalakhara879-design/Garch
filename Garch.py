import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>⚡ Volatility Risk Engine (GARCH Based)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Designed for Position Sizing & Risk Control</p>", unsafe_allow_html=True)

# ---------------- BRANDING ----------------
st.markdown("""
<div style='text-align:right; font-size:14px;'>
<b>Developed by: Piyush Dave</b><br>
🔗 <a href='https://www.linkedin.com' target='_blank'>LinkedIn Profile</a>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT ----------------
col1, col2 = st.columns([1,1])

with col1:
    ticker = st.text_input("Stock Symbol", "RELIANCE.NS").upper().strip()
    period = st.selectbox("Data Period", ["3mo", "6mo", "1y"])
    capital = st.number_input("Capital (₹)", value=100000)
    risk_pct = st.number_input("Risk per Trade (%)", value=1.0) / 100

    st.markdown("### 🎛️ Model Sensitivity Controls")

    omega = st.slider("Omega (Base Volatility)", 0.0000001, 0.00001, 0.000001, format="%.7f")
    alpha = st.slider("Alpha (Market Shock Sensitivity)", 0.01, 0.5, 0.1)
    beta = st.slider("Beta (Trend Persistence)", 0.1, 0.98, 0.85)

    run = st.button("🚀 Run Analysis")

# ---------------- RIGHT PANEL ----------------
with col2:
    st.markdown("## 📘 What This Tool Does")
    st.markdown("""
This tool uses an advanced financial model (**GARCH**) to estimate **future market volatility**  
and helps you decide **how much quantity to trade safely**.
""")

# ---------------- RUN ----------------
if run:
    try:
        df = yf.download(ticker, period=period, interval="1d")

        if df.empty:
            st.error("No data found. Check ticker symbol.")
            st.stop()

        df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df = df.dropna()

        phi = alpha + beta
        if phi >= 1:
            st.error("❌ Invalid: Alpha + Beta must be < 1")
            st.stop()

        sigma2 = np.var(df["Returns"])
        for r in df["Returns"]:
            sigma2 = omega + alpha * (r**2) + beta * sigma2

        h = 21
        long_var = omega / (1 - phi)

        forecasts = []
        temp_sigma2 = sigma2

        for i in range(h):
            temp_sigma2 = long_var + phi * (temp_sigma2 - long_var)
            forecasts.append(np.sqrt(temp_sigma2))

        forecast_vol = np.mean(forecasts)
        sigma_month = forecast_vol * np.sqrt(21)

        price = float(df["Close"].iloc[-1])
        risk_cash = capital * risk_pct
        move_per_share = sigma_month * price

        qty = max(1, int(risk_cash / move_per_share))
        max_qty = int(capital / price)

        # OUTPUT
        st.markdown("## 📊 Risk-Based Trading Output")

        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Monthly Volatility", f"{sigma_month*100:.2f}%")
        c2.metric("Current Stock Price", f"₹{price:.2f}")
        c3.metric("Risk Control Position Size", qty)

        st.markdown("### 📋 Trade Decision Summary")

        table = pd.DataFrame({
            "Parameter": [
                "Available Capital",
                "Max Quantity (Capital Based)",
                "Risk-Based Quantity",
                "Final Suggested Quantity"
            ],
            "Value": [
                f"₹{capital:,.0f}",
                max_qty,
                qty,
                min(max_qty, qty)
            ]
        })

        st.table(table)

        if sigma_month > 0.05:
            st.error("🔴 High Risk Market → Reduce Exposure")
        elif sigma_month < 0.02:
            st.success("🟢 Low Risk Market → Opportunity to Scale")
        else:
            st.info("🟡 Moderate Risk → Balanced Positioning")

    except Exception as e:
        st.error(f"Error: {e}")
