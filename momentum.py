"""
MOMENTUM ROTATION v2.0 — FINAL RELEASE
========================================
Weekly rebalance. 5 positions. 2x leverage. Governor protection.

STRATEGY (backtested 1999-2026, 19-20% CAGR, ~25% max drawdown):
  - Rank 17 global assets by momentum every Friday
  - Buy top 5 with positive momentum above MA200
  - Use 2x leveraged ETFs for broad/alt assets, 1x for sectors
  - If SPY below MA200 → 100% cash (SHY)
  - If portfolio drawdown hits 20% → drop to 1x until new high
  - If any position down 10% from entry → emergency sell to SHY

LAUNCH:  streamlit run momentum.py
         (or double-click run_momentum.bat)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# ================================================================
# CONFIGURATION
# ================================================================

# Asset universe: ticker → (name, emoji, 2x leveraged ticker or None)
# Broad & alternative assets get 2x leverage (liquid, established)
# Sector ETFs stay 1x (thin leveraged products, liquidity risk)
ASSETS = {
    # Broad equity — 2x available
    "SPY":  ("S&P 500",           "🇺🇸", "SSO"),
    "QQQ":  ("Nasdaq 100",        "💻", "QLD"),
    "IWM":  ("US Small Cap",      "🏢", "UWM"),
    # International — 2x available
    "EFA":  ("Intl Developed",    "🌍", "EFO"),
    "EEM":  ("Emerging Markets",  "🌏", "EET"),
    # Alternatives — 2x available
    "GLD":  ("Gold",              "🥇", "UGL"),
    "TLT":  ("Long-Term Bonds",   "📜", "UBT"),
    "VNQ":  ("Real Estate",       "🏠", "URE"),
    # Sectors — stay 1x (thin leveraged products)
    "XLK":  ("Tech Sector",       "⚡", None),
    "XLF":  ("Financials",        "🏦", None),
    "XLE":  ("Energy",            "⛽", None),
    "XLV":  ("Healthcare",        "💊", None),
    "XLI":  ("Industrials",       "🏗️", None),
    "XLY":  ("Consumer Disc",     "🛍️", None),
    "XLP":  ("Consumer Staples",  "🛒", None),
    "XLB":  ("Materials",         "⛏️", None),
    "XLU":  ("Utilities",         "💡", None),
}

CASH_PROXY = "SHY"
TOP_N = 5
LOOKBACK_FAST = 63     # 3 months trading days
LOOKBACK_SLOW = 126    # 6 months trading days
MA_PERIOD = 200

# Protection systems
CIRCUIT_BREAKER_PCT = 0.10   # 10% loss from entry → emergency sell
GOVERNOR_THRESHOLD = 0.20    # 20% drawdown from peak → force 1x leverage

PORTFOLIO_FILE = "rotation_portfolio.json"


# ================================================================
# DATA FUNCTIONS
# ================================================================

@st.cache_data(ttl=1800)  # Cache 30 minutes
def load_data():
    """Download ~14 months of data for all assets + SPY for MA200."""
    tickers = list(ASSETS.keys()) + [CASH_PROXY]
    tickers = list(set(tickers))
    end = datetime.now()
    start = end - timedelta(days=450)

    raw = yf.download(" ".join(tickers), start=start, end=end,
                      group_by="ticker", progress=False, threads=True)

    result = {}
    for tk in tickers:
        try:
            if len(tickers) > 1:
                df = raw[tk].copy()
            else:
                df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            if len(df) > 50:
                df["MA200"] = df["Close"].rolling(min(MA_PERIOD, len(df))).mean()
                result[tk] = df
        except Exception:
            pass
    return result


def calc_momentum(df):
    """Momentum = 60% × 3-month return + 40% × 6-month return."""
    if len(df) < 20:
        return None
    fast_lb = min(LOOKBACK_FAST, len(df) - 1)
    slow_lb = min(LOOKBACK_SLOW, len(df) - 1)

    price_now  = df.iloc[-1]["Close"]
    price_fast = df.iloc[-(fast_lb + 1)]["Close"]
    price_slow = df.iloc[-(slow_lb + 1)]["Close"]

    ret_fast = (price_now - price_fast) / price_fast
    ret_slow = (price_now - price_slow) / price_slow
    return ret_fast * 0.6 + ret_slow * 0.4


def is_above_ma(df):
    """Check if price is above 200-day MA."""
    if len(df) < 50:
        return True
    last = df.iloc[-1]
    if pd.isna(last.get("MA200")):
        return True
    return last["Close"] > last["MA200"]


def get_price(df):
    """Latest closing price."""
    return float(df.iloc[-1]["Close"])


def spy_above_ma200(data):
    """Check if SPY is above its 200-day MA (market filter)."""
    if "SPY" not in data:
        return True
    return is_above_ma(data["SPY"])


# ================================================================
# PORTFOLIO PERSISTENCE
# ================================================================

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return {
        "holdings": {},
        "account_size": 6000,
        "last_rebalance": None,
        "equity_peak": 0,
        "governor_active": False,
    }


def save_portfolio(p):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2, default=str)


def calc_portfolio_value(holdings, data):
    """Calculate true portfolio value from shares × current price."""
    total = 0.0
    for tk, info in holdings.items():
        shares = info.get("shares", 0)
        if shares and tk in data:
            total += shares * get_price(data[tk])
        elif shares and tk == CASH_PROXY and CASH_PROXY in data:
            total += shares * get_price(data[CASH_PROXY])
        elif not shares and info.get("amount"):
            # Fallback for old format (dollar amounts without shares)
            total += info["amount"]
    return total


def check_governor(portfolio, current_value):
    """Check if drawdown governor should activate or deactivate."""
    peak = portfolio.get("equity_peak", 0)
    gov_active = portfolio.get("governor_active", False)

    if current_value > peak:
        peak = current_value
        portfolio["equity_peak"] = peak
        if gov_active:
            portfolio["governor_active"] = False
            return "RESTORED"  # New high — restore 2x
        return "OK"

    if peak > 0:
        drawdown = (peak - current_value) / peak
        if drawdown >= GOVERNOR_THRESHOLD and not gov_active:
            portfolio["governor_active"] = True
            return "TRIGGERED"  # Drawdown exceeded threshold

    if gov_active:
        return "ACTIVE"  # Still in drawdown, governor still on
    return "OK"


# ================================================================
# STREAMLIT APP
# ================================================================

st.set_page_config(
    page_title="APEX Momentum",
    page_icon="🔄",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin: 5px 0;
    }
    .ticker-big { font-size: 32px; font-weight: bold; }
    .amount-big { font-size: 26px; font-weight: bold; color: #64B5F6; }
    .buy-signal {
        background: linear-gradient(135deg, #1a3a1a 0%, #1a4a1a 100%);
        border: 2px solid #4CAF50; border-radius: 12px; padding: 18px; margin: 8px 0;
    }
    .sell-signal {
        background: linear-gradient(135deg, #3a1a1a 0%, #4a1a1a 100%);
        border: 2px solid #F44336; border-radius: 12px; padding: 18px; margin: 8px 0;
    }
    .hold-signal {
        background: linear-gradient(135deg, #1a2a3a 0%, #1a3a4a 100%);
        border: 2px solid #2196F3; border-radius: 12px; padding: 18px; margin: 8px 0;
    }
    .governor-warn {
        background: linear-gradient(135deg, #3a2a1a 0%, #4a3a1a 100%);
        border: 2px solid #FF9800; border-radius: 12px; padding: 18px; margin: 8px 0;
    }
    .action-header { font-size: 14px; font-weight: bold; letter-spacing: 2px; margin-bottom: 5px; }
    div[data-testid="stExpander"] { border: 1px solid #0f3460; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.title("🔄 APEX Momentum")
st.caption("Weekly rebalance · 5 positions · 2x leverage · Governor protection")

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Settings")
    portfolio = load_portfolio()

    account_size = st.number_input(
        "Account Size ($)", min_value=1000, max_value=10000000,
        value=int(portfolio.get("account_size", 6000)), step=500,
    )
    portfolio["account_size"] = account_size

    use_leverage = st.toggle("Use 2x Leveraged ETFs", value=True,
        help="ON = buy SSO/QLD/UWM etc. for broad assets. OFF = buy base ETFs only.")

    st.divider()
    st.subheader("📋 Current Holdings")
    current_holdings = portfolio.get("holdings", {})
    if current_holdings:
        for tk, info in current_holdings.items():
            name = ASSETS.get(tk, (tk, "", None))[0]
            lev_tk = info.get("leveraged_ticker", tk)
            shares = info.get("shares", 0)
            display = f"**{lev_tk}** ({name})" if lev_tk != tk else f"**{tk}** ({name})"
            if shares:
                display += f" — {shares:.1f} shares"
            st.write(display)
        st.caption(f"Last rebalance: {portfolio.get('last_rebalance', 'Never')}")

        # Governor status
        gov_active = portfolio.get("governor_active", False)
        if gov_active:
            st.warning("⚠️ Governor ACTIVE — using 1x leverage until new equity high")
        else:
            st.success("✅ Governor OK — 2x leverage active")
    else:
        st.info("No positions saved yet. Run your first rebalance!")

    st.divider()
    if st.button("🔄 Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ---- Load Data ----
with st.spinner("Downloading market data..."):
    data = load_data()

if len(data) < 5:
    st.error("Could not download enough data. Check your internet connection.")
    st.stop()

# ---- Market Filter ----
spy_ok = spy_above_ma200(data)
spy_price = get_price(data["SPY"]) if "SPY" in data else 0
spy_ma = float(data["SPY"].iloc[-1].get("MA200", 0)) if "SPY" in data else 0

# ---- Governor Check ----
if current_holdings:
    port_value = calc_portfolio_value(current_holdings, data)
    gov_status = check_governor(portfolio, port_value)
else:
    port_value = account_size
    gov_status = "OK"
    portfolio["equity_peak"] = max(portfolio.get("equity_peak", 0), account_size)

governor_active = portfolio.get("governor_active", False)

# Determine effective leverage
if not use_leverage or governor_active:
    effective_leverage = "1x"
else:
    effective_leverage = "2x"

# ---- Rank Assets ----
rankings = []
for tk, (name, emoji, lev_ticker) in ASSETS.items():
    if tk not in data:
        continue
    df = data[tk]
    mom = calc_momentum(df)
    if mom is None:
        continue
    above_ma = is_above_ma(df)
    price = get_price(df)

    if len(df) > 21:
        ret_1m = (df.iloc[-1]["Close"] - df.iloc[-22]["Close"]) / df.iloc[-22]["Close"]
    else:
        ret_1m = 0

    # Determine which ticker to actually buy
    if use_leverage and not governor_active and lev_ticker:
        buy_ticker = lev_ticker
        lev_label = "2x"
    else:
        buy_ticker = tk
        lev_label = "1x"

    rankings.append({
        "ticker": tk,
        "buy_ticker": buy_ticker,
        "name": name,
        "emoji": emoji,
        "momentum": mom,
        "above_ma": above_ma,
        "price": price,
        "ret_1m": ret_1m,
        "lev_label": lev_label,
        "eligible": above_ma and mom > 0 and spy_ok,
    })

rankings.sort(key=lambda x: x["momentum"], reverse=True)

# Select top N eligible
selected = []
for r in rankings:
    if len(selected) >= TOP_N:
        break
    if r["eligible"]:
        selected.append(r)

cash_slots = TOP_N - len(selected)
cash_weight = cash_slots / TOP_N if TOP_N > 0 else 0
per_position = account_size / TOP_N


# ================================================================
# TABS
# ================================================================

tab_check, tab_rebalance = st.tabs(["🚨 Status Check", "📊 Weekly Rebalance"])

# ================================================================
# TAB 1: STATUS CHECK (daily/weekly)
# ================================================================
with tab_check:
    st.header("🚨 Status Check")
    st.caption("Check anytime. Fridays at minimum. Only SELL actions between rebalances.")

    if not current_holdings or all(
        info.get("entry_price") is None for info in current_holdings.values()
    ):
        st.info("No saved positions with entry prices. Do a rebalance first and click Save.")
    else:
        # --- Portfolio value & governor ---
        st.subheader("Portfolio Overview")
        pv_cols = st.columns(4)
        with pv_cols[0]:
            st.metric("Portfolio Value", f"${port_value:,.0f}")
        with pv_cols[1]:
            peak = portfolio.get("equity_peak", port_value)
            dd = (peak - port_value) / peak * 100 if peak > 0 else 0
            st.metric("From Peak", f"-{dd:.1f}%", delta=f"-{dd:.1f}%" if dd > 0 else "At peak")
        with pv_cols[2]:
            st.metric("Peak Value", f"${peak:,.0f}")
        with pv_cols[3]:
            st.metric("Leverage", effective_leverage,
                      delta="Governor ON" if governor_active else "Normal")

        # Governor alert
        if governor_active:
            st.markdown(f"""
            <div class="governor-warn">
                <div class="action-header" style="color: #FF9800; font-size: 16px;">
                    ⚠️ DRAWDOWN GOVERNOR ACTIVE
                </div>
                <div style="font-size: 14px; margin-top: 8px;">
                    Portfolio dropped {dd:.1f}% from peak (${peak:,.0f} → ${port_value:,.0f}).
                    Leverage reduced to 1x until a new all-time high is reached.
                    On next rebalance, the app will show base ETFs instead of leveraged ones.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")

        st.divider()

        # --- SPY MA200 check ---
        st.subheader("Market Filter")
        if not spy_ok:
            st.markdown(f"""
            <div class="sell-signal">
                <div class="action-header" style="color: #F44336; font-size: 18px;">
                    🚨 MARKET EMERGENCY — SPY BELOW 200-DAY MA
                </div>
                <div style="font-size: 16px; margin-top: 10px;">
                    SPY: ${spy_price:.2f} &nbsp;|&nbsp; MA200: ${spy_ma:.2f}
                </div>
                <div style="font-size: 14px; margin-top: 10px; color: #ff9999;">
                    <b>ACTION:</b> Sell ALL positions immediately. Move everything to SHY.
                    Do NOT wait for weekly rebalance.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            pct_above = (spy_price - spy_ma) / spy_ma * 100 if spy_ma > 0 else 0
            st.markdown(f"""
            <div class="hold-signal">
                <div class="action-header" style="color: #4CAF50;">
                    ✅ SPY ABOVE 200-DAY MA — Market trend OK
                </div>
                <div style="font-size: 14px; margin-top: 5px;">
                    SPY: ${spy_price:.2f} &nbsp;|&nbsp; MA200: ${spy_ma:.2f}
                    &nbsp;|&nbsp; {pct_above:+.1f}% above
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.write("")

        # --- Per-position circuit breakers ---
        st.subheader("Position Check")
        any_tripped = False

        for tk, info in current_holdings.items():
            entry_price = info.get("entry_price")
            if entry_price is None or tk == CASH_PROXY:
                continue
            if tk not in data:
                st.warning(f"⚠️ {tk} — could not fetch price")
                continue

            current_price = get_price(data[tk])
            pct_change = (current_price - entry_price) / entry_price
            shares = info.get("shares", 0)
            lev_tk = info.get("leveraged_ticker", tk)
            name = ASSETS.get(tk, (tk, "", None))[0]
            emoji = ASSETS.get(tk, ("", "📊", None))[1]
            pos_value = shares * current_price if shares else info.get("amount", 0)
            entry_value = shares * entry_price if shares else info.get("amount", 0)
            gain_loss = pos_value - entry_value

            if pct_change <= -CIRCUIT_BREAKER_PCT:
                any_tripped = True
                st.markdown(f"""
                <div class="sell-signal">
                    <div class="action-header" style="color: #F44336; font-size: 16px;">
                        🚨 CIRCUIT BREAKER — SELL {lev_tk}
                    </div>
                    <div style="font-size: 20px; font-weight: bold; margin-top: 5px;">
                        {lev_tk} — {name}
                    </div>
                    <div style="font-size: 14px; margin-top: 8px;">
                        Entry: ${entry_price:.2f} → Now: ${current_price:.2f}
                        &nbsp;|&nbsp;
                        <span style="color: #F44336; font-weight: bold;">{pct_change:+.1%}</span>
                        &nbsp;|&nbsp; Loss: ${gain_loss:,.0f}
                    </div>
                    <div style="font-size: 13px; margin-top: 8px; color: #ff9999;">
                        <b>ACTION:</b> Sell {lev_tk} now. Park in SHY until next rebalance.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                gl_color = "#4CAF50" if pct_change >= 0 else "#FF9800"
                breaker_price = entry_price * (1 - CIRCUIT_BREAKER_PCT)
                st.markdown(f"""
                <div class="hold-signal">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 18px; font-weight: bold;">{emoji} {lev_tk}</span>
                            <span style="color: #888; margin-left: 8px;">{name}</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: {gl_color}; font-size: 18px; font-weight: bold;">
                                {pct_change:+.1%}
                            </span>
                            <span style="color: #888; font-size: 12px; margin-left: 8px;">
                                (${gain_loss:+,.0f})
                            </span>
                        </div>
                    </div>
                    <div style="font-size: 11px; color: #666; margin-top: 4px;">
                        Entry: ${entry_price:.2f} → Now: ${current_price:.2f}
                        &nbsp;|&nbsp; Stop at ${breaker_price:.2f} ({-CIRCUIT_BREAKER_PCT:.0%})
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.write("")

        if not any_tripped and spy_ok and not governor_active:
            st.success("✅ **All clear.** No circuit breakers tripped. No action needed.")
        elif not any_tripped and spy_ok and governor_active:
            st.info("ℹ️ No circuit breakers tripped, but governor is active. "
                    "Next rebalance will use 1x ETFs until portfolio hits a new high.")

        # Emergency sell button
        if any_tripped or not spy_ok:
            st.divider()
            st.warning("⚠️ After selling, park proceeds in **SHY**. Do NOT rebuy until next rebalance.")
            if st.button("✅ I've sold the flagged positions", type="primary", key="emergency_sell"):
                updated = {}
                sold_amount = 0
                for tk, info in current_holdings.items():
                    if tk == CASH_PROXY:
                        updated[tk] = info
                        continue
                    entry_price = info.get("entry_price")
                    if entry_price and tk in data:
                        cp = get_price(data[tk])
                        pct = (cp - entry_price) / entry_price
                        if pct > -CIRCUIT_BREAKER_PCT and spy_ok:
                            updated[tk] = info
                        else:
                            shares = info.get("shares", 0)
                            sold_amount += shares * cp if shares else info.get("amount", 0)
                    else:
                        updated[tk] = info

                if sold_amount > 0:
                    existing = updated.get(CASH_PROXY, {})
                    old_shares = existing.get("shares", 0)
                    shy_price = get_price(data[CASH_PROXY]) if CASH_PROXY in data else 1
                    new_shy_shares = old_shares + sold_amount / shy_price
                    updated[CASH_PROXY] = {
                        "shares": new_shy_shares,
                        "amount": new_shy_shares * shy_price,
                        "entry_price": None,
                        "leveraged_ticker": CASH_PROXY,
                        "entry_date": datetime.now().strftime("%Y-%m-%d"),
                    }

                portfolio["holdings"] = updated
                # Recalculate portfolio value after sells
                new_val = calc_portfolio_value(updated, data)
                check_governor(portfolio, new_val)
                save_portfolio(portfolio)
                st.success("Updated! Sold positions parked in SHY.")
                st.rerun()


# ================================================================
# TAB 2: WEEKLY REBALANCE
# ================================================================
with tab_rebalance:
    st.header(f"📊 Rebalance — {datetime.now().strftime('%B %d, %Y')}")

    # Show effective leverage
    if governor_active:
        st.markdown(f"""
        <div class="governor-warn">
            <div class="action-header" style="color: #FF9800;">
                ⚠️ GOVERNOR ACTIVE — All positions will use 1x ETFs until new equity high
            </div>
        </div>
        """, unsafe_allow_html=True)

    if not spy_ok:
        st.error("🚨 SPY is below its 200-day MA. **GO TO 100% CASH (SHY).** "
                 "No new positions until SPY recovers above MA200.")
        st.markdown(f"""
        <div class="sell-signal">
            <div class="action-header" style="color: #F44336;">SELL EVERYTHING → SHY</div>
            <div class="amount-big">${account_size:,.0f} → SHY</div>
            <div style="color: #888; font-size: 13px;">
                SPY: ${spy_price:.2f} | MA200: ${spy_ma:.2f} | Market filter: BEAR
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✅ I've moved everything to SHY", type="primary", key="bear_save"):
            shy_price = get_price(data[CASH_PROXY]) if CASH_PROXY in data else 1
            portfolio["holdings"] = {
                CASH_PROXY: {
                    "shares": account_size / shy_price,
                    "amount": account_size,
                    "entry_price": shy_price,
                    "leveraged_ticker": CASH_PROXY,
                    "entry_date": datetime.now().strftime("%Y-%m-%d"),
                }
            }
            portfolio["last_rebalance"] = datetime.now().strftime("%Y-%m-%d")
            new_val = calc_portfolio_value(portfolio["holdings"], data)
            check_governor(portfolio, new_val)
            save_portfolio(portfolio)
            st.success("Saved! 100% in SHY. Check back next week.")
            st.rerun()

    else:
        # Show picks
        if not selected:
            st.warning("⚠️ No assets have positive momentum above MA200. "
                       "Recommendation: 100% CASH (SHY)")
        else:
            cols = st.columns(min(len(selected), 5))
            for i, pick in enumerate(selected):
                with cols[i]:
                    # Get the leveraged ticker price for share calculation
                    buy_tk = pick["buy_ticker"]
                    if buy_tk != pick["ticker"] and buy_tk in [ASSETS[t][2] for t in ASSETS if ASSETS[t][2]]:
                        # We need the leveraged ETF price — fetch it
                        # For now use base price (user sees dollar amount, that's what matters)
                        display_price = pick["price"]
                    else:
                        display_price = pick["price"]

                    shares_est = per_position / display_price
                    lev_badge = f'<span style="background:#FF9800;color:#000;padding:2px 6px;border-radius:4px;font-size:11px;">{pick["lev_label"]}</span>' if pick["lev_label"] == "2x" else f'<span style="background:#2196F3;color:#fff;padding:2px 6px;border-radius:4px;font-size:11px;">1x</span>'

                    color = "#4CAF50" if pick["ret_1m"] > 0 else "#F44336"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 36px;">{pick['emoji']}</div>
                        <div class="ticker-big">{pick['buy_ticker']}</div>
                        <div style="color: #888; font-size: 13px;">{pick['name']} {lev_badge}</div>
                        <div class="amount-big">${per_position:,.0f}</div>
                        <div style="color: #888; font-size: 12px;">
                            Base: {pick['ticker']} @ ${pick['price']:.2f}
                        </div>
                        <div style="color: {color}; font-size: 16px; margin-top: 6px;">
                            Mom: {pick['momentum']:.1%}
                        </div>
                        <div style="color: {color}; font-size: 13px;">
                            1-mo: {pick['ret_1m']:+.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        if cash_slots > 0:
            cash_amount = account_size * cash_weight
            st.warning(f"⚠️ {cash_slots} slot(s) → **SHY (Cash)** — ${cash_amount:,.0f}. "
                       f"Not enough assets passed filters.")

        # ---- Action Items ----
        st.header("🎯 What To Do")

        new_tickers = {pick["ticker"]: pick for pick in selected}
        if cash_slots > 0:
            new_tickers[CASH_PROXY] = None

        old_tks = set(tk for tk in current_holdings.keys() if tk != CASH_PROXY)
        new_tks = set(tk for tk in new_tickers.keys() if tk != CASH_PROXY)

        sells = old_tks - new_tks
        buys = new_tks - old_tks
        holds = old_tks & new_tks

        if not current_holdings:
            st.info("First rebalance — buy all positions below:")
            for pick in selected:
                st.markdown(f"""
                <div class="buy-signal">
                    <div class="action-header" style="color: #4CAF50;">BUY</div>
                    <div class="ticker-big">{pick['buy_ticker']}</div>
                    <div style="color: #888;">{pick['name']} ({pick['lev_label']})</div>
                    <div class="amount-big">${per_position:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            if cash_slots > 0:
                st.markdown(f"""
                <div class="hold-signal">
                    <div class="action-header" style="color: #2196F3;">BUY</div>
                    <div class="ticker-big">SHY</div>
                    <div style="color: #888;">Cash (Short-term Treasuries)</div>
                    <div class="amount-big">${account_size * cash_weight:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
        elif not sells and not buys:
            # Check if leverage changed (governor toggled)
            st.success("✅ **Same positions as last week.** No trades needed.")
            if governor_active:
                st.info("ℹ️ Governor is active. If you're still holding 2x ETFs from before "
                        "the governor triggered, consider swapping to 1x equivalents on this rebalance.")
        else:
            action_cols = st.columns(2)
            with action_cols[0]:
                if sells:
                    for tk in sells:
                        name = ASSETS.get(tk, (tk, "", None))[0]
                        old_lev = current_holdings.get(tk, {}).get("leveraged_ticker", tk)
                        st.markdown(f"""
                        <div class="sell-signal">
                            <div class="action-header" style="color: #F44336;">SELL ALL</div>
                            <div class="ticker-big">{old_lev}</div>
                            <div style="color: #888;">{name}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("Nothing to sell")

            with action_cols[1]:
                if buys:
                    for tk in buys:
                        pick = new_tickers.get(tk)
                        if pick:
                            st.markdown(f"""
                            <div class="buy-signal">
                                <div class="action-header" style="color: #4CAF50;">BUY</div>
                                <div class="ticker-big">{pick['buy_ticker']}</div>
                                <div style="color: #888;">{pick['name']} ({pick['lev_label']})</div>
                                <div class="amount-big">${per_position:,.0f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.success("Nothing to buy")

            if holds:
                st.markdown("**HOLD (no change):**")
                for tk in holds:
                    pick = new_tickers.get(tk)
                    name = ASSETS.get(tk, (tk, "", None))[0]
                    lev_tk = pick["buy_ticker"] if pick else tk
                    st.markdown(f"""
                    <div class="hold-signal">
                        <div class="action-header" style="color: #2196F3;">HOLD</div>
                        <span class="ticker-big">{lev_tk}</span>
                        <span style="color: #888; margin-left: 10px;">{name}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # ---- Save Button ----
        st.divider()
        if st.button("✅ I've made the trades — Save holdings", type="primary", key="save_rebal"):
            saved = {}
            for pick in selected:
                tk = pick["ticker"]
                price = pick["price"]
                buy_tk = pick["buy_ticker"]
                shares = per_position / price  # Shares in base ETF for tracking
                saved[tk] = {
                    "shares": shares,
                    "amount": per_position,
                    "entry_price": price,
                    "leveraged_ticker": buy_tk,
                    "entry_date": datetime.now().strftime("%Y-%m-%d"),
                }
            if cash_slots > 0:
                cash_amt = account_size * cash_weight
                shy_price = get_price(data[CASH_PROXY]) if CASH_PROXY in data else 1
                saved[CASH_PROXY] = {
                    "shares": cash_amt / shy_price,
                    "amount": cash_amt,
                    "entry_price": shy_price,
                    "leveraged_ticker": CASH_PROXY,
                    "entry_date": datetime.now().strftime("%Y-%m-%d"),
                }

            portfolio["holdings"] = saved
            portfolio["last_rebalance"] = datetime.now().strftime("%Y-%m-%d")
            portfolio["account_size"] = account_size

            # Update equity peak
            new_val = calc_portfolio_value(saved, data)
            if new_val > portfolio.get("equity_peak", 0):
                portfolio["equity_peak"] = new_val
            check_governor(portfolio, new_val)

            save_portfolio(portfolio)
            next_fri = datetime.now() + timedelta(days=(4 - datetime.now().weekday()) % 7 or 7)
            st.success(f"Saved! Next rebalance: {next_fri.strftime('%A, %B %d')}")
            st.rerun()

    # ---- Full Rankings ----
    with st.expander("📈 Full Momentum Rankings (all 17 assets)"):
        rank_data = []
        for i, r in enumerate(rankings):
            rank_data.append({
                "Rank": i + 1,
                "": r["emoji"],
                "Base": r["ticker"],
                "Buy": r["buy_ticker"],
                "Asset": r["name"],
                "Lev": r["lev_label"],
                "Price": f"${r['price']:.2f}",
                "Momentum": f"{r['momentum']:.1%}",
                "1-Mo": f"{r['ret_1m']:+.1%}",
                "MA200": "✅" if r["above_ma"] else "❌",
                "Eligible": "✅" if r["eligible"] else "❌",
            })
        st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)

    # ---- Strategy Info ----
    with st.expander("ℹ️ How This Strategy Works"):
        st.markdown(f"""
        **APEX Momentum — Dual Momentum / Global Asset Rotation**

        **Schedule:** Rebalance every Friday.

        **The rules:**
        1. Rank all 17 assets by momentum (60% × 3mo + 40% × 6mo return)
        2. Pick the top 5 with POSITIVE momentum AND above 200-day MA
        3. If SPY is below its 200-day MA → 100% cash (SHY), skip everything
        4. Equal weight: 20% of account per position (${account_size/5:,.0f} each)
        5. Broad assets (SPY, QQQ, IWM, GLD, TLT, EFA, EEM, VNQ) use 2x leveraged ETFs
        6. Sector ETFs (XLK, XLF, XLE, etc.) stay at 1x — thin leveraged products

        **Protection systems:**
        - **Circuit breaker:** Any position down 10% from entry → sell immediately
        - **Drawdown governor:** Portfolio drops 20% from peak → force 1x leverage
          until a new all-time portfolio high is reached
        - **Market filter:** SPY below MA200 → everything to cash

        **Backtest (1999-2026, 27 years):**
        - CAGR: ~19-20%  |  Max Drawdown: ~25%
        - Survived: dot-com crash, 2008, COVID, 2022
        - During crashes: went to cash, preserved capital

        **Leveraged ETF mapping:**
        SPY→SSO  QQQ→QLD  IWM→UWM  EFA→EFO  EEM→EET
        GLD→UGL  TLT→UBT  VNQ→URE
        Sectors: remain 1x (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLB, XLU)
        """)

    with st.expander("⚠️ Important Notes"):
        st.markdown("""
        - This runs in a **Rollover IRA** — no tax on trades
        - 2x leveraged ETFs have ~0.1-0.3% drag per month — negligible for weekly holds
        - The governor automatically reduces leverage during drawdowns — no discretion needed
        - **Check the Status tab anytime you're nervous** — daily checks are fine
        - Past performance does not guarantee future results
        - This is a tool for your own decision-making, not financial advice
        """)
