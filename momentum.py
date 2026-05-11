"""
MOMENTUM ROTATION v2.0 — FINAL RELEASE
========================================
Weekly rebalance. 5 positions. 2x leverage. Governor protection.

STRATEGY (backtested 1999-2026, 19-20% CAGR, ~25% max drawdown):
  - Rank 17 global assets by momentum every Friday
  - Buy top 5 with positive momentum above MA200
  - Use 2x leveraged ETFs for broad/alt assets, 1x for sectors
  - If SPY below MA200 → 100% cash (SPAXX)
  - If portfolio drawdown hits 20% → drop to 1x until new high
  - If any position down 10% from entry → emergency sell to SPAXX

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
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

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

CASH_PROXY = "SPAXX"     # Money market fund, always $1.00/share
TOP_N = 5
LOOKBACK_FAST = 63     # 3 months trading days
LOOKBACK_SLOW = 126    # 6 months trading days
MA_PERIOD = 200

# Protection systems
CIRCUIT_BREAKER_PCT = 0.10   # 10% loss from entry → emergency sell
GOVERNOR_THRESHOLD = 0.20    # 20% drawdown from peak → force 1x leverage

# Circuit Breaker Measurement Mode
# -----------------------------------------------------------------------------
# When holding a 2x leveraged ETF (UWM, EET, QLD, etc.), do we measure the -10%
# breaker against the BASE underlying or the LEVERAGED ETF we actually hold?
#
# Backtest comparison (1996-2026, all 3 modes produced identical 22.12% CAGR
# because Variant E weekly rotation handles risk before stops fire):
#   "BASE"      -  4 trips in 24yr, 50% recovery rate. Measures asset-class signal.
#   "LEVERAGED" -  2 trips in 24yr, 100% recovery rate (i.e. 2/2 false alarms).
#                  Whipsaws on routine 5% underlying moves amplified to 10% leveraged.
#   "NONE"      -  No intra-week stops. Pure weekly rotation handles all risk.
#
# DECISION: BASE — measures what the strategy ranks (base tickers) for consistency.
# Doesn't whipsaw on leverage amplification of normal volatility.
# All three behave identically in the backtest, so this is a philosophy choice.
BREAKER_MODE = "BASE"  # options: "BASE", "LEVERAGED", "NONE"

# Variant E: Safe haven rotation during bear markets
# When SPY < MA200, rotate into the best momentum safe haven at 1x
SAFE_HAVENS = ["GLD", "TLT"]  # Gold and bonds — go UP during crashes

PORTFOLIO_FILE = "rotation_portfolio.json"
LOG_FILE = "rebalance_log.csv"


def log_rebalance(action, holdings_list, spy_price, spy_ma, account_val,
                  leverage, governor_active, spy_ok):
    """Append a row to the rebalance log CSV. Auto-creates the file."""
    import csv
    now = datetime.now(ET)
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "date", "time_et", "action", "account_value",
                "spy_price", "spy_ma200", "spy_above_ma",
                "leverage", "governor_active",
                "pos1", "pos1_price",
                "pos2", "pos2_price",
                "pos3", "pos3_price",
                "pos4", "pos4_price",
                "pos5", "pos5_price",
            ])

        # Pad holdings list to 5 entries
        padded = holdings_list + [("SPAXX", 0.0)] * (5 - len(holdings_list))
        row = [
            now.strftime("%Y-%m-%d"),
            now.strftime("%I:%M %p ET"),
            action,
            f"{account_val:.0f}",
            f"{spy_price:.2f}",
            f"{spy_ma:.2f}",
            "YES" if spy_ok else "NO",
            leverage,
            "YES" if governor_active else "NO",
        ]
        for tk, price in padded[:5]:
            row.extend([tk, f"{price:.2f}"])
        writer.writerow(row)


# ================================================================
# DATA FUNCTIONS
# ================================================================

@st.cache_data(ttl=1800)  # Cache 30 minutes
def load_data():
    """Download ~14 months of data for all assets AND their leveraged ETF counterparts."""
    base_tickers = list(ASSETS.keys())
    # Also fetch leveraged tickers so we can store the correct entry price
    # when buying 2x positions like SSO, QLD, UWM, EET, UGL, UBT, URE, EFO
    leveraged_tickers = [info[2] for info in ASSETS.values() if info[2]]
    tickers = list(set(base_tickers + leveraged_tickers))
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


def get_buy_price(buy_ticker, base_ticker, base_price, data):
    """Return the actual price of the ticker being bought.

    For 1x sectors (XLK, XLE, etc.) buy_ticker == base_ticker, so base_price is correct.
    For 2x positions (UWM, EET, QLD, etc.) we MUST fetch the leveraged ticker's
    real price. UWM ≠ IWM × 2 — it's an entirely different traded instrument.
    Falling back to base_price would store a wildly wrong entry, breaking circuit breakers.
    """
    if buy_ticker == base_ticker:
        return base_price
    if buy_ticker in data and len(data[buy_ticker]) > 0:
        return float(data[buy_ticker]["Close"].iloc[-1])
    # Hard fallback: if leveraged data missing, return None so caller can flag the issue
    return None


def calc_portfolio_value(holdings, data):
    """Calculate true portfolio value from shares × current price.
    SPAXX (money market) is always valued at the stored dollar amount.

    For 2x positions, we hold the leveraged ticker (e.g. UWM, EET) — not the base.
    Must look up current price using leveraged_ticker, otherwise we'd multiply
    UWM shares by IWM's price, which is meaningless.
    """
    total = 0.0
    for tk, info in holdings.items():
        if tk == CASH_PROXY:
            # SPAXX is cash — always worth the stored dollar amount
            total += info.get("amount", 0)
        else:
            shares = info.get("shares", 0)
            # Use the actual ticker we hold (leveraged_ticker), fall back to base
            price_tk = info.get("leveraged_ticker", tk)
            if shares and price_tk in data:
                total += shares * get_price(data[price_tk])
            elif shares and tk in data:
                # Fallback if leveraged data missing
                total += shares * get_price(data[tk])
            elif info.get("amount"):
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
        color: #e0e0e0;
    }
    .ticker-big { font-size: 32px; font-weight: bold; color: #ffffff; }
    .amount-big { font-size: 26px; font-weight: bold; color: #64B5F6; }
    .buy-signal {
        background: linear-gradient(135deg, #1a3a1a 0%, #1a4a1a 100%);
        border: 2px solid #4CAF50; border-radius: 12px; padding: 18px; margin: 8px 0;
        color: #e0e0e0;
    }
    .sell-signal {
        background: linear-gradient(135deg, #3a1a1a 0%, #4a1a1a 100%);
        border: 2px solid #F44336; border-radius: 12px; padding: 18px; margin: 8px 0;
        color: #e0e0e0;
    }
    .hold-signal {
        background: linear-gradient(135deg, #1a2a3a 0%, #1a3a4a 100%);
        border: 2px solid #2196F3; border-radius: 12px; padding: 18px; margin: 8px 0;
        color: #e0e0e0;
    }
    .governor-warn {
        background: linear-gradient(135deg, #3a2a1a 0%, #4a3a1a 100%);
        border: 2px solid #FF9800; border-radius: 12px; padding: 18px; margin: 8px 0;
        color: #e0e0e0;
    }
    .action-header { font-size: 14px; font-weight: bold; letter-spacing: 2px; margin-bottom: 5px; }
    div[data-testid="stExpander"] { border: 1px solid #0f3460; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.title("🔄 APEX Momentum")
st.caption("Weekly rebalance · 5 positions · 2x leverage · Governor protection · Safe haven rotation")

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Settings")
    portfolio = load_portfolio()

    # account_size is the SEED capital — only used when portfolio is empty.
    # Once positions exist, the rebalance math uses LIVE portfolio value
    # so target slot size grows with the portfolio (proper compounding).
    account_size = st.number_input(
        "Starting Capital ($) — used only if no positions yet", min_value=1000, max_value=10000000,
        value=int(portfolio.get("account_size", 6000)), step=500,
        help="Once you have live holdings, the app calculates target slot size from your CURRENT portfolio value, not this number.",
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

# ---- Data Timestamp ----
# Show when the price data is from and when the app last refreshed
latest_date = None
if "SPY" in data:
    latest_date = data["SPY"].index[-1]
now = datetime.now(ET)
if latest_date is not None:
    st.caption(f"📡 Data as of: **{latest_date.strftime('%A, %B %d, %Y')}** market close "
               f"&nbsp;|&nbsp; App refreshed: **{now.strftime('%I:%M %p')} ET** "
               f"({now.strftime('%m/%d/%Y')})"
               f"&nbsp;|&nbsp; "
               f"{'🟢 Market hours' if now.weekday() < 5 and 9 <= now.hour < 16 else '🔴 Market closed'}")
else:
    st.caption(f"📡 App refreshed: {now.strftime('%I:%M %p')} ET ({now.strftime('%m/%d/%Y')})")

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

# ===== LIVE PORTFOLIO VALUE COMPUTATION =====
# CRITICAL: target slot size must grow with the portfolio for compounding to work.
# Static "$1,400 per slot forever" would prevent the strategy from compounding past
# its starting capital — exactly what destroys the backtest's 22% CAGR over 20 years.
#
# Logic:
# - If we have live holdings → pilot_active_value = sum of current position values
#   (excludes FGRTX, ALAYN reserves, anything outside the app's tracked holdings)
# - If no holdings yet (fresh start) → fall back to seed account_size
# - target_per_slot = pilot_active_value / TOP_N (5)
#
# Note: SPAXX inside the app's holdings counts as cash slot, included in pilot value.
# But ALAYN reserve sitting in Fidelity SPAXX is NOT in the app — it's not tracked.
current_holdings_for_value = portfolio.get("holdings", {})
if current_holdings_for_value:
    pilot_active_value = calc_portfolio_value(current_holdings_for_value, data)
    # Safety: if calc returned 0 (data issue), fall back to seed value
    if pilot_active_value <= 0:
        pilot_active_value = float(account_size)
        rebalance_basis_note = f"⚠️ Could not compute live value — using seed ${account_size:,}"
    else:
        rebalance_basis_note = f"Live pilot value (sum of current positions)"
else:
    # No holdings yet — first-time setup uses seed capital
    pilot_active_value = float(account_size)
    rebalance_basis_note = f"Initial seed (no positions yet)"

per_position = pilot_active_value / TOP_N

# ---- VARIANT E: Safe Haven Selection (bear market) ----
bear_pick = None  # Will be set if SPY < MA200 and a safe haven has momentum
if not spy_ok:
    best_haven_mom = 0
    for tk in SAFE_HAVENS:
        if tk not in data:
            continue
        df = data[tk]
        mom = calc_momentum(df)
        if mom is not None and mom > best_haven_mom and is_above_ma(df):
            haven_info = ASSETS.get(tk)
            if haven_info:
                best_haven_mom = mom
                bear_pick = {
                    "ticker": tk,
                    "buy_ticker": tk,  # Always 1x during bear mode
                    "name": haven_info[0],
                    "emoji": haven_info[1],
                    "momentum": mom,
                    "price": get_price(df),
                    "lev_label": "1x",
                    "ret_1m": (df.iloc[-1]["Close"] - df.iloc[-22]["Close"]) / df.iloc[-22]["Close"] if len(df) > 21 else 0,
                }


# ================================================================
# TABS
# ================================================================

tab_check, tab_rebalance, tab_edit = st.tabs(["🚨 Status Check", "📊 Weekly Rebalance", "✏️ Edit Holdings"])

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
            if bear_pick:
                st.markdown(f"""
                <div class="governor-warn">
                    <div class="action-header" style="color: #FF9800; font-size: 18px;">
                        ⚠️ BEAR MODE — SPY BELOW 200-DAY MA
                    </div>
                    <div style="font-size: 16px; margin-top: 10px;">
                        SPY: ${spy_price:.2f} &nbsp;|&nbsp; MA200: ${spy_ma:.2f}
                    </div>
                    <div style="font-size: 14px; margin-top: 10px; color: #FFD54F;">
                        <b>SAFE HAVEN:</b> {bear_pick['emoji']} {bear_pick['ticker']} ({bear_pick['name']})
                        has {bear_pick['momentum']:.1%} momentum.
                        Hold 100% in {bear_pick['ticker']} at 1x until SPY recovers above MA200.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="sell-signal">
                    <div class="action-header" style="color: #F44336; font-size: 18px;">
                        🚨 BEAR MODE — NO SAFE HAVEN AVAILABLE
                    </div>
                    <div style="font-size: 16px; margin-top: 10px;">
                        SPY: ${spy_price:.2f} &nbsp;|&nbsp; MA200: ${spy_ma:.2f}
                    </div>
                    <div style="font-size: 14px; margin-top: 10px; color: #ff9999;">
                        <b>ACTION:</b> Neither gold nor bonds have positive momentum.
                        Move everything to SPAXX. Do NOT wait for weekly rebalance.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            pct_above = (spy_price - spy_ma) / spy_ma * 100 if spy_ma > 0 else 0
            # Check if user is transitioning from bear to bull
            status_held = set(current_holdings.keys()) - {CASH_PROXY}
            status_transitioning = status_held.issubset(set(SAFE_HAVENS)) and len(status_held) > 0
            status_is_friday = datetime.now(ET).weekday() == 4

            if status_transitioning and not status_is_friday:
                st.markdown(f"""
                <div class="governor-warn">
                    <div class="action-header" style="color: #FF9800;">
                        ⏳ SPY CROSSED ABOVE MA200 — WAIT FOR FRIDAY TO SWITCH
                    </div>
                    <div style="font-size: 14px; margin-top: 5px;">
                        SPY: ${spy_price:.2f} &nbsp;|&nbsp; MA200: ${spy_ma:.2f}
                        &nbsp;|&nbsp; {pct_above:+.1f}% above.
                        Hold {', '.join(status_held)} until Friday to confirm this isn't a false crossover.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif status_transitioning and status_is_friday:
                st.markdown(f"""
                <div class="buy-signal">
                    <div class="action-header" style="color: #4CAF50;">
                        🟢 BULL MODE CONFIRMED — SWITCH TO FULL PORTFOLIO TODAY
                    </div>
                    <div style="font-size: 14px; margin-top: 5px;">
                        SPY: ${spy_price:.2f} &nbsp;|&nbsp; MA200: ${spy_ma:.2f}
                        &nbsp;|&nbsp; {pct_above:+.1f}% above.
                        Go to the Weekly Rebalance tab to see your 5 picks and execute.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
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
            # Use leveraged_ticker for current price lookup — that's what we actually hold.
            # For sectors, leveraged_ticker == tk, so this works for both 1x and 2x.
            price_tk = info.get("leveraged_ticker", tk)
            if price_tk not in data:
                st.warning(f"⚠️ {price_tk} — could not fetch price")
                continue

            current_price = get_price(data[price_tk])
            shares = info.get("shares", 0)
            lev_tk = info.get("leveraged_ticker", tk)
            name = ASSETS.get(tk, (tk, "", None))[0]
            emoji = ASSETS.get(tk, ("", "📊", None))[1]
            pos_value = shares * current_price if shares else info.get("amount", 0)
            entry_value = shares * entry_price if shares else info.get("amount", 0)
            gain_loss = pos_value - entry_value

            # === CIRCUIT BREAKER MEASUREMENT ===
            # BASE mode: measure -10% on the underlying asset (IWM, EEM, XLK)
            #            so leverage amplification doesn't whipsaw us out
            # LEVERAGED:  measure -10% on the leveraged ETF itself (UWM, EET)
            # NONE:      no intra-week breaker
            base_entry_price = info.get("base_entry_price", entry_price)  # fallback for old saved data
            base_current_price = get_price(data[tk]) if tk in data else current_price

            if BREAKER_MODE == "BASE":
                # Compare base ticker prices. For sectors, base==leveraged so result identical.
                breaker_pct = (base_current_price - base_entry_price) / base_entry_price
                breaker_label = f"{tk} (base)"
            elif BREAKER_MODE == "LEVERAGED":
                breaker_pct = (current_price - entry_price) / entry_price
                breaker_label = f"{lev_tk} (held)"
            else:  # NONE
                breaker_pct = 0.0  # never trips
                breaker_label = "disabled"

            # Display % change is ALWAYS the leveraged-ticker change (what user sees in Fidelity)
            display_pct = (current_price - entry_price) / entry_price

            tripped = breaker_pct <= -CIRCUIT_BREAKER_PCT and BREAKER_MODE != "NONE"

            if tripped:
                any_tripped = True
                breaker_explainer = ""
                if BREAKER_MODE == "BASE" and tk != lev_tk:
                    breaker_explainer = f"<br><span style='font-size:12px;color:#ffaaaa;'>Base {tk} dropped {breaker_pct:+.1%} (your held {lev_tk}: {display_pct:+.1%}).</span>"
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
                        <span style="color: #F44336; font-weight: bold;">{display_pct:+.1%}</span>
                        &nbsp;|&nbsp; Loss: ${gain_loss:,.0f}
                        {breaker_explainer}
                    </div>
                    <div style="font-size: 13px; margin-top: 8px; color: #ff9999;">
                        <b>ACTION:</b> Sell {lev_tk} now. Park in SPAXX until next rebalance.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                gl_color = "#4CAF50" if display_pct >= 0 else "#FF9800"
                # Show stop level in the ticker user is monitoring
                if BREAKER_MODE == "BASE":
                    stop_level_str = f"Base {tk} @ ${base_entry_price * (1 - CIRCUIT_BREAKER_PCT):.2f} stop ({-CIRCUIT_BREAKER_PCT:.0%})"
                elif BREAKER_MODE == "LEVERAGED":
                    stop_level_str = f"Stop @ ${entry_price * (1 - CIRCUIT_BREAKER_PCT):.2f} ({-CIRCUIT_BREAKER_PCT:.0%})"
                else:
                    stop_level_str = "No intra-week stop (relies on weekly rotation)"
                st.markdown(f"""
                <div class="hold-signal">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 18px; font-weight: bold;">{emoji} {lev_tk}</span>
                            <span style="color: #888; margin-left: 8px;">{name}</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: {gl_color}; font-size: 18px; font-weight: bold;">
                                {display_pct:+.1%}
                            </span>
                            <span style="color: #888; font-size: 12px; margin-left: 8px;">
                                (${gain_loss:+,.0f})
                            </span>
                        </div>
                    </div>
                    <div style="font-size: 11px; color: #666; margin-top: 4px;">
                        Entry: ${entry_price:.2f} → Now: ${current_price:.2f}
                        &nbsp;|&nbsp; {stop_level_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.write("")

        if not any_tripped and spy_ok and not governor_active:
            st.success("✅ **All clear.** No circuit breakers tripped. No action needed.")
        elif not any_tripped and spy_ok and governor_active:
            st.info("ℹ️ No circuit breakers tripped, but governor is active. "
                    "Next rebalance will use 1x ETFs until portfolio hits a new high.")
        elif not any_tripped and not spy_ok:
            # Check if user is already in safe haven or needs to rotate
            held_tickers = set(current_holdings.keys()) - {CASH_PROXY}
            in_safe_haven = held_tickers.issubset(set(SAFE_HAVENS))
            if in_safe_haven or not held_tickers:
                if bear_pick:
                    st.info(f"ℹ️ Bear mode active. You should be holding **{bear_pick['ticker']}** "
                            f"({bear_pick['name']}). Check the Weekly Rebalance tab if you need to rotate.")
                else:
                    st.info("ℹ️ Bear mode active. No safe haven has momentum. Stay in SPAXX.")
            else:
                st.warning(f"⚠️ SPY below MA200 but you're holding bull-mode positions. "
                           f"Rotate to {'**' + bear_pick['ticker'] + '**' if bear_pick else '**SPAXX**'} "
                           f"on next rebalance (Friday).")

        # Emergency sell button (only for circuit breakers, not bear mode safe havens)
        if any_tripped:
            st.divider()
            st.warning("⚠️ After selling, park proceeds in **SPAXX**. Do NOT rebuy until next rebalance.")
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
                    old_amount = existing.get("amount", 0)
                    new_amount = old_amount + sold_amount
                    updated[CASH_PROXY] = {
                        "shares": new_amount,
                        "amount": new_amount,
                        "entry_price": None,
                        "leveraged_ticker": CASH_PROXY,
                        "entry_date": datetime.now().strftime("%Y-%m-%d"),
                    }

                portfolio["holdings"] = updated
                # Recalculate portfolio value after sells
                new_val = calc_portfolio_value(updated, data)
                check_governor(portfolio, new_val)
                save_portfolio(portfolio)
                log_rebalance("EMERGENCY_SELL",
                    [(tk, get_price(data[tk])) for tk in updated if tk != CASH_PROXY and tk in data],
                    spy_price, spy_ma, new_val, effective_leverage, governor_active, spy_ok)
                st.success("Updated! Sold positions parked in SPAXX.")
                st.rerun()


# ================================================================
# TAB 2: WEEKLY REBALANCE
# ================================================================
with tab_rebalance:
    st.header(f"📊 Rebalance — {datetime.now().strftime('%B %d, %Y')}")

    # Show what's driving the rebalance math
    # Use HTML directly since markdown parser mangles dollar signs + asterisks
    if current_holdings_for_value:
        st.markdown(
            f'<div style="color:#888;font-size:13px;padding:8px 0;">'
            f'💰 <b>Pilot active value: ${pilot_active_value:,.0f}</b> '
            f'(sum of current 5 positions). '
            f'Target per slot: <b>${per_position:,.0f}</b> (20% each). '
            f'Excludes FGRTX and any reserves outside the app.'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="color:#888;font-size:13px;padding:8px 0;">'
            f'💰 <b>Starting capital: ${pilot_active_value:,.0f}</b> (seed — no holdings yet). '
            f'Target per slot: <b>${per_position:,.0f}</b> (20% each).'
            f'</div>',
            unsafe_allow_html=True
        )

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
        if bear_pick:
            # Variant E: Safe haven rotation
            # In bear mode, full pilot value goes into the safe haven (1x, single position)
            st.warning(f"⚠️ SPY below MA200 — **BEAR MODE**. Rotating into safe haven: "
                       f"**{bear_pick['ticker']}** ({bear_pick['name']}) at 1x leverage.")
            st.markdown(f"""
            <div class="governor-warn">
                <div class="action-header" style="color: #FF9800;">BEAR MODE — SAFE HAVEN</div>
                <div style="font-size: 36px; margin: 8px 0;">{bear_pick['emoji']}</div>
                <div class="ticker-big">{bear_pick['ticker']}</div>
                <div style="color: #888; font-size: 14px;">{bear_pick['name']} (1x — no leverage in bear mode)</div>
                <div class="amount-big">${pilot_active_value:,.0f}</div>
                <div style="color: #888; font-size: 12px;">
                    ≈ {pilot_active_value / bear_pick['price']:.1f} shares @ ${bear_pick['price']:.2f}
                </div>
                <div style="color: #FF9800; font-size: 16px; margin-top: 8px;">
                    Momentum: {bear_pick['momentum']:.1%} | 1-month: {bear_pick['ret_1m']:+.1%}
                </div>
                <div style="color: #888; font-size: 13px; margin-top: 8px;">
                    SPY: ${spy_price:.2f} | MA200: ${spy_ma:.2f} | Market filter: BEAR
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"✅ I've bought {bear_pick['ticker']} — Save holdings", type="primary", key="bear_haven_save"):
                portfolio["holdings"] = {
                    bear_pick["ticker"]: {
                        "shares": pilot_active_value / bear_pick["price"],
                        "amount": pilot_active_value,
                        "entry_price": bear_pick["price"],
                        "base_entry_price": bear_pick["price"],
                        "leveraged_ticker": bear_pick["ticker"],
                        "entry_date": datetime.now().strftime("%Y-%m-%d"),
                    }
                }
                portfolio["last_rebalance"] = datetime.now().strftime("%Y-%m-%d")
                new_val = calc_portfolio_value(portfolio["holdings"], data)
                check_governor(portfolio, new_val)
                save_portfolio(portfolio)
                log_rebalance("BEAR_HAVEN", [(bear_pick["ticker"], bear_pick["price"])],
                    spy_price, spy_ma, new_val, effective_leverage, governor_active, spy_ok)
                st.success(f"Saved! Holding {bear_pick['ticker']}. Check back next week.")
                st.rerun()

        else:
            # No safe haven has momentum — pure cash
            st.error("🚨 SPY below MA200 and no safe haven has positive momentum. "
                     "**GO TO 100% CASH (SPAXX).**")
            st.markdown(f"""
            <div class="sell-signal">
                <div class="action-header" style="color: #F44336;">BEAR MODE — ALL CASH</div>
                <div class="amount-big">${pilot_active_value:,.0f} → SPAXX</div>
                <div style="color: #888; font-size: 13px;">
                    SPY: ${spy_price:.2f} | MA200: ${spy_ma:.2f} | GLD & TLT: no positive momentum
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("✅ I've moved everything to SPAXX", type="primary", key="bear_save"):
                portfolio["holdings"] = {
                    CASH_PROXY: {
                        "shares": pilot_active_value,
                        "amount": pilot_active_value,
                        "entry_price": None,
                        "base_entry_price": None,
                        "leveraged_ticker": CASH_PROXY,
                        "entry_date": datetime.now().strftime("%Y-%m-%d"),
                    }
                }
                portfolio["last_rebalance"] = datetime.now().strftime("%Y-%m-%d")
                new_val = calc_portfolio_value(portfolio["holdings"], data)
                check_governor(portfolio, new_val)
                save_portfolio(portfolio)
                log_rebalance("BEAR_CASH", [("SPAXX", 1.0)],
                    spy_price, spy_ma, new_val, effective_leverage, governor_active, spy_ok)
                st.success("Saved! 100% in SPAXX. Check back next week.")
                st.rerun()

    else:
        # Check if transitioning from bear to bull (holding safe haven, now in bull mode)
        held_tickers = set(current_holdings.keys()) - {CASH_PROXY}
        transitioning_from_bear = held_tickers.issubset(set(SAFE_HAVENS)) and len(held_tickers) > 0
        is_friday = datetime.now(ET).weekday() == 4  # 4 = Friday

        if transitioning_from_bear and not is_friday:
            days_until_friday = (4 - datetime.now(ET).weekday()) % 7
            if days_until_friday == 0:
                days_until_friday = 7
            st.markdown(f"""
            <div class="governor-warn">
                <div class="action-header" style="color: #FF9800; font-size: 16px;">
                    ⏳ REGIME CHANGE DETECTED — WAIT FOR FRIDAY
                </div>
                <div style="font-size: 14px; margin-top: 8px;">
                    SPY just crossed above MA200 — bull mode is activating.
                    But <b>do NOT trade today.</b> Hold your current safe haven position
                    ({', '.join(held_tickers)}) until Friday to confirm the crossover is real.
                    SPY can easily dip back below MA200 in the next {days_until_friday} day(s).
                    If it's still above on Friday, execute the trades below.
                </div>
                <div style="font-size: 13px; margin-top: 8px; color: #FF9800;">
                    SPY: ${spy_price:.2f} | MA200: ${spy_ma:.2f} |
                    Above by: {((spy_price - spy_ma) / spy_ma * 100):.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")

        # Show picks
        if not selected:
            st.warning("⚠️ No assets have positive momentum above MA200. "
                       "Recommendation: 100% CASH (SPAXX)")
        else:
            cols = st.columns(min(len(selected), 5))
            for i, pick in enumerate(selected):
                with cols[i]:
                    # Get the ACTUAL price of the ticker we're buying
                    # For 2x positions this is critical — UWM/EET/QLD etc. trade at
                    # their own prices, not the base ETF price
                    buy_tk = pick["buy_ticker"]
                    actual_buy_price = get_buy_price(buy_tk, pick["ticker"], pick["price"], data)
                    if actual_buy_price is None:
                        # Leveraged data missing — fall back to base but warn
                        actual_buy_price = pick["price"]
                        st.warning(f"⚠️ Could not load price for {buy_tk}. Using base ticker price as fallback. Verify entry in Edit Holdings after trade.")

                    pick["buy_price"] = actual_buy_price  # stash for save logic
                    shares_est = per_position / actual_buy_price
                    lev_badge = f'<span style="background:#FF9800;color:#000;padding:2px 6px;border-radius:4px;font-size:11px;">{pick["lev_label"]}</span>' if pick["lev_label"] == "2x" else f'<span style="background:#2196F3;color:#fff;padding:2px 6px;border-radius:4px;font-size:11px;">1x</span>'

                    color = "#4CAF50" if pick["ret_1m"] > 0 else "#F44336"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 36px;">{pick['emoji']}</div>
                        <div class="ticker-big">{pick['buy_ticker']}</div>
                        <div style="color: #888; font-size: 13px;">{pick['name']} {lev_badge}</div>
                        <div class="amount-big">${per_position:,.0f}</div>
                        <div style="color: #888; font-size: 12px;">
                            {pick['buy_ticker']} @ ${actual_buy_price:.2f} ≈ {shares_est:.1f} sh
                        </div>
                        <div style="color: #666; font-size: 11px;">
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
            cash_amount = pilot_active_value * cash_weight
            st.warning(f"⚠️ {cash_slots} slot(s) → **SPAXX (Cash)** — ${cash_amount:,.0f}. "
                       f"Not enough assets passed filters.")

        # ---- Action Items ----
        st.header("🎯 What To Do")

        is_rebalance_day = datetime.now(ET).weekday() == 4  # Friday

        if not is_rebalance_day and current_holdings:
            days_to_friday = (4 - datetime.now(ET).weekday()) % 7
            if days_to_friday == 0:
                days_to_friday = 7
            st.markdown(f"""
            <div class="governor-warn">
                <div class="action-header" style="color: #FF9800; font-size: 16px;">
                    📅 NOT REBALANCE DAY — DO NOT TRADE
                </div>
                <div style="font-size: 14px; margin-top: 8px;">
                    Today is {datetime.now(ET).strftime('%A')}. Rebalancing only happens on <b>Fridays</b>.
                    The picks below are a <b>preview</b> of what the portfolio would look like
                    if you rebalanced today — but rankings change daily. Wait {days_to_friday} day(s)
                    and check again Friday.
                    <br><br>
                    <b>The only mid-week actions:</b> sell if a circuit breaker trips (10% loss)
                    or if SPY drops below MA200.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.markdown("**📋 Preview of current rankings (do NOT act on these today):**")

        new_tickers = {pick["ticker"]: pick for pick in selected}
        if cash_slots > 0:
            new_tickers[CASH_PROXY] = None

        old_tks = set(tk for tk in current_holdings.keys() if tk != CASH_PROXY)
        new_tks = set(tk for tk in new_tickers.keys() if tk != CASH_PROXY)

        sells = old_tks - new_tks
        buys = new_tks - old_tks
        holds = old_tks & new_tks

        # Detect leverage swaps (e.g., GLD 1x → UGL 2x or vice versa)
        leverage_swaps = set()
        true_holds = set()
        for tk in holds:
            old_lev_tk = current_holdings.get(tk, {}).get("leveraged_ticker", tk)
            new_pick = new_tickers.get(tk)
            new_lev_tk = new_pick["buy_ticker"] if new_pick else tk
            if old_lev_tk != new_lev_tk:
                leverage_swaps.add(tk)
            else:
                true_holds.add(tk)

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
                    <div class="ticker-big">SPAXX</div>
                    <div style="color: #888;">Cash (Short-term Treasuries)</div>
                    <div class="amount-big">${pilot_active_value * cash_weight:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # We have existing holdings — show SELL / BUY / SWAP first, then TRIM/ADD for kept positions
            if not sells and not buys and not leverage_swaps:
                st.success("✅ **Same tickers as last week.** No new positions to buy or sell — but check rebalance drift below.")
                if governor_active:
                    st.info("ℹ️ Governor is active. If you're still holding 2x ETFs from before "
                            "the governor triggered, consider swapping to 1x equivalents on this rebalance.")
            else:
                action_cols = st.columns(2)
                with action_cols[0]:
                    has_sells = bool(sells) or bool(leverage_swaps)
                    if sells or leverage_swaps:
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
                        for tk in leverage_swaps:
                            name = ASSETS.get(tk, (tk, "", None))[0]
                            old_lev = current_holdings.get(tk, {}).get("leveraged_ticker", tk)
                            new_pick = new_tickers.get(tk)
                            new_lev = new_pick["buy_ticker"] if new_pick else tk
                            st.markdown(f"""
                            <div class="sell-signal">
                                <div class="action-header" style="color: #FF9800;">SWAP — SELL</div>
                                <div class="ticker-big">{old_lev} → {new_lev}</div>
                                <div style="color: #888;">{name} — switching from {old_lev} (bear mode) to {new_lev} (bull mode)</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("Nothing to sell")

                with action_cols[1]:
                    if buys or leverage_swaps:
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
                        for tk in leverage_swaps:
                            pick = new_tickers.get(tk)
                            if pick:
                                st.markdown(f"""
                                <div class="buy-signal">
                                    <div class="action-header" style="color: #FF9800;">SWAP — BUY</div>
                                    <div class="ticker-big">{pick['buy_ticker']}</div>
                                    <div style="color: #888;">{pick['name']} ({pick['lev_label']}) — replacing bear mode position</div>
                                    <div class="amount-big">${per_position:,.0f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.success("Nothing to buy")

            # ===== TRIM/ADD/HOLD logic for positions we keep =====
            # This runs in BOTH branches above — whether we have buys/sells or not.
            # Critical: a "same tickers" week still needs drift correction to maintain equal weight.
            if true_holds:
                st.markdown("**HOLD / REBALANCE TO TARGET:**")
                st.caption(f"Target per slot: ${per_position:,.0f} ({rebalance_basis_note}). "
                           f"Trim winners and top up underweights to maintain equal weight.")
                for tk in true_holds:
                    pick = new_tickers.get(tk)
                    name = ASSETS.get(tk, (tk, "", None))[0]
                    lev_tk = pick["buy_ticker"] if pick else tk
                    # Compute current value of this held position using leveraged ticker price
                    holding_info = current_holdings.get(tk, {})
                    held_shares = holding_info.get("shares", 0)
                    held_price_tk = holding_info.get("leveraged_ticker", tk)
                    if held_price_tk in data:
                        held_current_price = get_price(data[held_price_tk])
                        current_value = held_shares * held_current_price
                    else:
                        current_value = holding_info.get("amount", per_position)
                        held_current_price = 0

                    delta = per_position - current_value  # positive = need to ADD, negative = TRIM
                    drift_pct = (current_value - per_position) / per_position if per_position > 0 else 0
                    abs_delta = abs(delta)

                    DRIFT_THRESHOLD = 0.03  # 3% — don't churn on tiny drifts
                    if abs(drift_pct) < DRIFT_THRESHOLD:
                        			action_label = "HOLD"
                        			action_color = "#2196F3"
                        			action_detail = f"At target (${current_value:,.0f} of ${per_position:,.0f})"
                        			action_amount_html = ""
                    elif delta < 0:
                        action_label = "TRIM"
                        action_color = "#FF9800"
                        shares_to_sell = abs_delta / held_current_price if held_current_price > 0 else 0
                        action_detail = (f"Currently ${current_value:,.0f} ({drift_pct:+.1%} from target). "
                                         f"Sell ~{shares_to_sell:.1f} shares.")
                        action_amount_html = f'<div class="amount-big" style="color:#FF9800;">SELL ${abs_delta:,.0f}</div>'
                    else:
                        action_label = "ADD"
                        action_color = "#4CAF50"
                        shares_to_buy = abs_delta / held_current_price if held_current_price > 0 else 0
                        action_detail = (f"Currently ${current_value:,.0f} ({drift_pct:+.1%} from target). "
                                         f"Buy ~{shares_to_buy:.1f} more shares.")
                        action_amount_html = f'<div class="amount-big" style="color:#4CAF50;">BUY ${abs_delta:,.0f}</div>'

                    amount_section = action_amount_html if action_amount_html else ""
                    st.markdown(
                       				 f'<div class="hold-signal">'
                        				f'<div class="action-header" style="color: {action_color};">{action_label}</div>'
                        				f'<span class="ticker-big">{lev_tk}</span>'
                       				 f'<span style="color: #888; margin-left: 10px;">{name}</span>'
                        				f'{amount_section}'
                       				 f'<div style="color: #888; font-size: 12px; margin-top: 4px;">{action_detail}</div>'
                       				 f'</div>',
                        				unsafe_allow_html=True
                    			)

        # ---- Save Button ----
        st.divider()
        if not is_rebalance_day and current_holdings:
            st.info("💡 Rebalancing normally happens on Fridays.")
            override = st.checkbox("I missed Friday — allow me to save today's trades")
        else:
            override = True

        if (is_rebalance_day or override) and st.button("✅ I've made the trades — Save holdings", type="primary", key="save_rebal"):
            saved = {}
            for pick in selected:
                tk = pick["ticker"]              # base ticker (IWM, EEM, XLK, etc.)
                buy_tk = pick["buy_ticker"]      # what we actually hold (UWM, EET, XLK)
                # Store BOTH prices:
                #   entry_price       = leveraged ticker price (for portfolio value math)
                #   base_entry_price  = base ticker price (for BreakerOnBase calculation)
                # For 1x sectors, base_entry_price == entry_price (same ticker).
                buy_price = pick.get("buy_price")
                if buy_price is None or buy_price <= 0:
                    buy_price = pick["price"]
                base_price = pick["price"]       # always the base/ranking ticker price
                shares = per_position / buy_price
                saved[tk] = {
                    "shares": shares,
                    "amount": per_position,
                    "entry_price": buy_price,
                    "base_entry_price": base_price,
                    "leveraged_ticker": buy_tk,
                    "entry_date": datetime.now().strftime("%Y-%m-%d"),
                }
            if cash_slots > 0:
                cash_amt = pilot_active_value * cash_weight
                saved[CASH_PROXY] = {
                    "shares": cash_amt,
                    "amount": cash_amt,
                    "entry_price": None,
                    "base_entry_price": None,
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
            log_rebalance("REBALANCE",
                [(pick["buy_ticker"], pick.get("buy_price", pick["price"])) for pick in selected],
                spy_price, spy_ma, new_val, effective_leverage, governor_active, spy_ok)
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
        **APEX Momentum — Dual Momentum / Global Asset Rotation (Variant E)**

        **Schedule:** Rebalance every Friday at 3pm ET.

        **BULL MODE (SPY above MA200):**
        1. Rank all 17 assets by momentum (60% × 3mo + 40% × 6mo return)
        2. Pick the top 5 with POSITIVE momentum AND above 200-day MA
        3. Equal weight: 20% of pilot active value per position (currently **${per_position:,.0f}** each)
        4. Broad assets use 2x leveraged ETFs, sectors stay 1x

        **BEAR MODE (SPY below MA200):**
        1. Check momentum of safe havens: GLD (gold) and TLT (bonds)
        2. Buy whichever has the highest positive momentum, at 1x (no leverage)
        3. If neither has positive momentum → 100% SPAXX (cash)
        4. Recheck every Friday — safe haven can rotate (GLD → TLT or vice versa)
        5. When SPY recovers above MA200 → switch back to bull mode

        **Protection systems:**
        - **Circuit breaker:** Any position down 10% from entry → sell immediately
        - **Drawdown governor:** Portfolio drops 20% from peak → force 1x leverage
          until a new all-time portfolio high is reached
        - **Market filter:** SPY below MA200 → bear mode (safe haven rotation)

        **Backtest (1999-2026, 27 years, 7 validation tests passed):**
        - CAGR: ~22%  |  Max Drawdown: ~27%
        - During 2008: gold rotation earned +16.6% while market crashed -38%
        - Walk-forward: 4/4 windows positive, avg 19.25%
        - Monte Carlo: 99.6% of 1,000 trials beat SPY
        - Deflated Sharpe: Z-score 24.7 (selection bias corrected)
        - Parameter sensitivity: 20-24% CAGR across all 15 parameter combos
        - Transaction costs: 20.65% CAGR at realistic cost levels

        **Leveraged ETF mapping (bull mode only):**
        SPY→SSO  QQQ→QLD  IWM→UWM  EFA→EFO  EEM→EET
        GLD→UGL  TLT→UBT  VNQ→URE
        Sectors: remain 1x (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLB, XLU)
        """)

    with st.expander("⚠️ Important Notes"):
        st.markdown("""
        - This runs in a **Rollover IRA** — no tax on trades
        - 2x leveraged ETFs have ~0.1-0.3% drag per month — negligible for weekly holds
        - **Bear mode** uses 1x only on GLD or TLT — no leverage on safe havens
        - The governor automatically reduces leverage during drawdowns — no discretion needed
        - **Check the Status tab anytime you're nervous** — daily checks are fine
        - **Fridays only for buying.** Selling (circuit breakers) can happen any day.
        - Past performance does not guarantee future results
        - This is a tool for your own decision-making, not financial advice
        """)

# ================================================================
# TAB 3: EDIT HOLDINGS
# ================================================================
with tab_edit:
    st.header("✏️ Edit Holdings")
    st.caption("Use this to fix a missed rebalance, correct an entry price, or sync the app after a manual trade.")

    edit_portfolio = load_portfolio()
    edit_holdings = edit_portfolio.get("holdings", {})

    # ---- Remove a position ----
    st.subheader("🗑️ Remove a Position")
    st.write("Use this when you sold a position and need to remove it from the app.")

    positions_to_remove = [tk for tk in edit_holdings.keys() if tk != CASH_PROXY]
    if not positions_to_remove:
        st.info("No positions saved. Nothing to remove.")
    else:
        remove_labels = {}
        for tk in positions_to_remove:
            lev_tk = edit_holdings[tk].get("leveraged_ticker", tk)
            name = ASSETS.get(tk, (tk, "", None))[0]
            remove_labels[f"{lev_tk} ({name})"] = tk

        remove_choice = st.selectbox("Which position did you sell?", list(remove_labels.keys()), key="remove_select")
        if st.button("🗑️ Remove this position", key="remove_btn"):
            tk_to_remove = remove_labels[remove_choice]
            del edit_portfolio["holdings"][tk_to_remove]
            save_portfolio(edit_portfolio)
            st.success(f"Removed {remove_choice} from your holdings.")
            st.rerun()

    st.divider()

    # ---- Add a new position ----
    st.subheader("➕ Add a New Position")
    st.write("Use this when you bought a new position and need to record it in the app.")

    all_base_tickers = list(ASSETS.keys())
    existing_tickers = list(edit_holdings.keys())
    available_to_add = [tk for tk in all_base_tickers if tk not in existing_tickers]

    if not available_to_add:
        st.info("All tracked positions already saved.")
    else:
        add_labels = {}
        for tk in available_to_add:
            info = ASSETS[tk]
            lev_tk = info[2] if info[2] else tk
            add_labels[f"{lev_tk} — {info[0]}"] = tk

        add_choice = st.selectbox("What did you buy?", list(add_labels.keys()), key="add_select")
        add_base_tk = add_labels[add_choice]
        add_info = ASSETS[add_base_tk]
        add_lev_tk = add_info[2] if add_info[2] else add_base_tk

        # Show persistent success message from previous add (survives st.rerun)
        if "add_success_msg" in st.session_state and st.session_state["add_success_msg"]:
            st.success(st.session_state["add_success_msg"])
            st.session_state["add_success_msg"] = None

        add_col1, add_col2 = st.columns(2)
        with add_col1:
            # User enters the price they actually paid for the leveraged ticker (UWM, EET, etc.)
            # If they're holding a 1x sector, leveraged ticker == base ticker.
            default_lev_price = 100.0
            if add_lev_tk in data:
                default_lev_price = float(get_price(data[add_lev_tk]))
            elif add_base_tk in data:
                default_lev_price = float(get_price(data[add_base_tk]))
            # Dynamic key per ticker so values reset between selections
            add_entry_price = st.number_input(
                f"Entry price for {add_lev_tk} ($) — what you paid in Fidelity",
                min_value=0.01, max_value=100000.0,
                value=default_lev_price,
                step=0.01, key=f"add_price_{add_base_tk}"
            )
        with add_col2:
            add_amount = st.number_input(
                "Dollar amount invested ($)",
                min_value=1.0, max_value=1000000.0,
                value=float(edit_portfolio.get("account_size", 6000) / TOP_N),
                step=1.0, key=f"add_amount_{add_base_tk}"
            )

        # Base entry price (for BreakerOnBase). Auto-fetched from current data if available.
        if add_lev_tk == add_base_tk:
            add_base_price_default = add_entry_price
        elif add_base_tk in data:
            add_base_price_default = float(get_price(data[add_base_tk]))
        else:
            add_base_price_default = add_entry_price
        add_base_price = st.number_input(
            f"Base ticker entry price for {add_base_tk} ($) — used for BreakerOnBase",
            min_value=0.01, max_value=100000.0,
            value=add_base_price_default,
            step=0.01, key=f"add_base_price_{add_base_tk}",
            help=f"For 2x positions, the breaker fires when {add_base_tk} drops 10% from this price (not when {add_lev_tk} drops 10%)."
        )

        if st.button(f"➕ Add {add_lev_tk} to holdings", type="primary", key=f"add_btn_{add_base_tk}"):
            edit_portfolio["holdings"][add_base_tk] = {
                "shares": add_amount / add_entry_price,
                "amount": add_amount,
                "entry_price": add_entry_price,
                "base_entry_price": add_base_price,
                "leveraged_ticker": add_lev_tk,
                "entry_date": datetime.now().strftime("%Y-%m-%d"),
            }
            save_portfolio(edit_portfolio)
            log_rebalance("MANUAL_ADD",
                [(add_lev_tk, add_entry_price)],
                spy_price, spy_ma,
                calc_portfolio_value(edit_portfolio["holdings"], data),
                effective_leverage, governor_active, spy_ok)
            st.session_state["add_success_msg"] = (
                f"✓ Added {add_lev_tk} at ${add_entry_price:.2f} "
                f"(base {add_base_tk} @ ${add_base_price:.2f}) for ${add_amount:,.0f}."
            )
            st.rerun()

    st.divider()

    # ---- Edit an existing position's entry price ----
    st.subheader("🔧 Fix an Entry Price")
    st.write("Use this if the app recorded the wrong entry price for a position you're already holding.")

    # Show persistent success message from previous save (survives st.rerun)
    if "fix_success_msg" in st.session_state and st.session_state["fix_success_msg"]:
        st.success(st.session_state["fix_success_msg"])
        # Clear it after showing once so it doesn't linger forever
        st.session_state["fix_success_msg"] = None

    edit_positions = [tk for tk in edit_holdings.keys() if tk != CASH_PROXY]
    if not edit_positions:
        st.info("No positions saved yet.")
    else:
        fix_labels = {}
        for tk in edit_positions:
            lev_tk = edit_holdings[tk].get("leveraged_ticker", tk)
            name = ASSETS.get(tk, (tk, "", None))[0]
            current_entry = edit_holdings[tk].get("entry_price", 0)
            fix_labels[f"{lev_tk} ({name}) — entry @ ${current_entry:.2f}"] = tk

        fix_choice = st.selectbox("Which position needs fixing?", list(fix_labels.keys()), key="fix_select")
        fix_tk = fix_labels[fix_choice]
        fix_lev_tk = edit_holdings[fix_tk].get("leveraged_ticker", fix_tk)
        current_entry = edit_holdings[fix_tk].get("entry_price", 0)
        current_base_entry = edit_holdings[fix_tk].get("base_entry_price", current_entry)
        current_amount = edit_holdings[fix_tk].get("amount", 0)

        # CRITICAL: include fix_tk in widget keys so each position has its OWN state.
        # Without this, Streamlit reuses the same widget across position selections,
        # causing the previous position's typed values to "stick" when you switch.
        fix_col1, fix_col2 = st.columns(2)
        with fix_col1:
            new_entry_price = st.number_input(
                f"Correct entry price for {fix_lev_tk} ($) — what Fidelity shows",
                min_value=0.01, max_value=100000.0,
                value=float(current_entry) if current_entry else 100.0,
                step=0.01, key=f"fix_price_{fix_tk}"
            )
        with fix_col2:
            new_amount = st.number_input(
                "Correct dollar amount ($)",
                min_value=1.0, max_value=1000000.0,
                value=float(current_amount) if current_amount else 1000.0,
                step=1.0, key=f"fix_amount_{fix_tk}"
            )

        # Base entry price for BreakerOnBase. For 1x sectors, leveraged == base.
        if fix_tk == fix_lev_tk:
            new_base_entry = new_entry_price
            st.caption(f"This is a 1x sector position — base entry = entry price (${new_entry_price:.2f}).")
        else:
            base_default = float(current_base_entry) if current_base_entry else (
                float(get_price(data[fix_tk])) if fix_tk in data else 100.0
            )
            new_base_entry = st.number_input(
                f"Correct base entry price for {fix_tk} ($) — used for BreakerOnBase",
                min_value=0.01, max_value=100000.0,
                value=base_default,
                step=0.01, key=f"fix_base_price_{fix_tk}",
                help=f"For 2x positions, the breaker fires when {fix_tk} drops 10% from this price."
            )

        if st.button(f"🔧 Update {fix_lev_tk} entry price", key=f"fix_btn_{fix_tk}"):
            edit_portfolio["holdings"][fix_tk]["entry_price"] = new_entry_price
            edit_portfolio["holdings"][fix_tk]["base_entry_price"] = new_base_entry
            edit_portfolio["holdings"][fix_tk]["amount"] = new_amount
            edit_portfolio["holdings"][fix_tk]["shares"] = new_amount / new_entry_price
            save_portfolio(edit_portfolio)
            # Stash success message in session_state so it survives the rerun
            st.session_state["fix_success_msg"] = (
                f"✓ Updated {fix_lev_tk}: entry ${new_entry_price:.2f}, "
                f"base {fix_tk} @ ${new_base_entry:.2f}, amount ${new_amount:,.0f}."
            )
            st.rerun()

    st.divider()

    # ---- Current holdings summary ----
    st.subheader("📋 Current Holdings Summary")
    if edit_holdings:
        summary_data = []
        for tk, info in edit_holdings.items():
            if tk == CASH_PROXY:
                summary_data.append({
                    "Ticker": "SPAXX",
                    "Name": "Cash",
                    "Entry Price": "$1.00",
                    "Amount": f"${info.get('amount', 0):,.0f}",
                    "Entry Date": info.get("entry_date", "—"),
                })
            else:
                lev_tk = info.get("leveraged_ticker", tk)
                name = ASSETS.get(tk, (tk, "", None))[0]
                entry = info.get("entry_price", 0)
                amount = info.get("amount", 0)
                summary_data.append({
                    "Ticker": lev_tk,
                    "Name": name,
                    "Entry Price": f"${entry:.2f}" if entry else "—",
                    "Amount": f"${amount:,.0f}" if amount else "—",
                    "Entry Date": info.get("entry_date", "—"),
                })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    else:
        st.info("No holdings saved yet.")
