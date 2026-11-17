"""
AML Money Mule Network Detector — Streamlit Dashboard
Run: streamlit run app/dashboard.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import networkx as nx

st.set_page_config(page_title="AML Network Detector", page_icon="🕵️", layout="wide")

import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)  # data/ and reports/ paths are relative to the project root


def bootstrap():
    """First run (e.g. a fresh clone or the hosted demo): generate data and run detection."""
    import subprocess
    if Path("reports/risk_scores.csv").exists():
        return
    with st.spinner("First run: generating the transaction network and running detection (about 30 seconds)..."):
        subprocess.run([sys.executable, "detect.py", "--generate-data"], cwd=PROJECT_ROOT,
                       check=True, capture_output=True, text=True)


bootstrap()


@st.cache_data
def load_data():
    txns_path = Path("data/transactions.csv")
    accs_path = Path("data/accounts.csv")
    if not txns_path.exists():
        return None, None
    return pd.read_csv(txns_path), pd.read_csv(accs_path)


@st.cache_data
def load_risk_scores():
    path = Path("reports/risk_scores.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_alerts():
    path = Path("reports/structuring_alerts.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_rings():
    path = Path("reports/mule_rings.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


def render_overview(txns_df, accounts_df, risk_df, alerts_df, rings_df):
    st.title("🕵️ AML Money Mule Network Detector")
    st.markdown("**FINTRAC-aligned detection** | Structuring • Mule Rings • Network Analysis")

    if txns_df is None:
        st.warning("Run `python detect.py --generate-data` first to generate data.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accounts", f"{len(accounts_df):,}")
    c2.metric("Transactions", f"{len(txns_df):,}")

    if rings_df is not None:
        c3.metric("🔴 Mule Rings", len(rings_df))
    if alerts_df is not None:
        c4.metric("⚠️ FINTRAC Alerts", len(alerts_df))
    if risk_df is not None:
        critical = len(risk_df[risk_df["risk_tier"] == "CRITICAL"]) if "risk_tier" in risk_df.columns else 0
        c5.metric("🚨 Critical Accounts", critical)

    network_img = Path("reports/network_graph.png")
    if network_img.exists():
        st.image(str(network_img), caption="Transaction Network (node size = volume, color = risk tier)", width="stretch")
    else:
        st.info("Run `python detect.py` to generate the network graph.")


def render_rings(rings_df, txns_df):
    st.header("🔴 Detected Mule Rings")
    if rings_df is None or rings_df.empty:
        st.info("No rings detected yet. Run `python detect.py`.")
        return

    for _, row in rings_df.iterrows():
        with st.expander(f"Ring {row.get('ring_id','?')} — Score: {row.get('ring_score','?')}/100 | {row.get('ring_type','?')}"):
            st.markdown(f"**Evidence:** {row.get('evidence_summary','')}")
            st.markdown(f"**Members:** `{row.get('member_account_ids','')}`")
            col1, col2, col3 = st.columns(3)
            col1.metric("Internal Density", f"{float(row.get('internal_density',0)):.2%}")
            col2.metric("Avg Pass-Through", f"{float(row.get('avg_pass_through_ratio',0)):.2%}")
            col3.metric("Round Amount Ratio", f"{float(row.get('round_amount_ratio',0)):.2%}")

            ring_img = Path(f"reports/ring_{row.get('ring_id','?')}.png")
            if ring_img.exists():
                st.image(str(ring_img), width="stretch")


def render_structuring(alerts_df):
    st.header("⚠️ Structuring / FINTRAC Alerts")
    if alerts_df is None or alerts_df.empty:
        st.info("No structuring alerts detected.")
        return

    st.caption("PCMLTFA s.9 — Deposits structured to stay below $10,000 CAD reporting threshold")

    for col in ["risk_score"]:
        if col in alerts_df.columns:
            alerts_df[col] = pd.to_numeric(alerts_df[col], errors="coerce")

    display_cols = [c for c in ["account_id", "alert_type", "risk_score", "fintrac_rule_reference", "evidence"]
                    if c in alerts_df.columns]
    sorted_df = alerts_df[display_cols].sort_values("risk_score", ascending=False) if "risk_score" in display_cols else alerts_df[display_cols]

    def color_tier(val):
        if val >= 80:
            return "background-color: #d62728; color: white"
        elif val >= 60:
            return "background-color: #ff7f0e"
        elif val >= 40:
            return "background-color: #fdd835"
        return ""

    if "risk_score" in sorted_df.columns:
        st.dataframe(
            sorted_df.style.map(color_tier, subset=["risk_score"]),
            width="stretch",
            height=400,
        )
    else:
        st.dataframe(sorted_df, width="stretch", height=400)


def render_account_lookup(txns_df, accounts_df, risk_df):
    st.header("🔍 Account Lookup")
    if txns_df is None:
        st.info("No data loaded.")
        return

    all_accounts = sorted(set(txns_df["sender_id"].unique()) | set(txns_df["receiver_id"].unique()))
    all_accounts = [a for a in all_accounts if a != "CASH_DEPOSIT"]

    selected = st.selectbox("Search account:", all_accounts)
    if selected:
        if risk_df is not None and "account_id" in risk_df.columns:
            row = risk_df[risk_df["account_id"] == selected]
            if not row.empty:
                r = row.iloc[0]
                col1, col2, col3 = st.columns(3)
                col1.metric("Risk Score", f"{r.get('risk_score', '?')}/100")
                col2.metric("Risk Tier", str(r.get("risk_tier", "?")))
                col3.metric("Top Factors", str(r.get("top_risk_factors", ""))[:60])

        st.subheader("Transaction History")
        sent = txns_df[txns_df["sender_id"] == selected]
        received = txns_df[txns_df["receiver_id"] == selected]
        combined = pd.concat([
            sent.assign(direction="SENT"),
            received.assign(direction="RECEIVED"),
        ]).sort_values("timestamp", ascending=False)

        st.dataframe(combined[["timestamp", "direction", "sender_id", "receiver_id",
                                "amount_cad", "txn_type", "channel"]].head(50),
                     width="stretch")


def main():
    txns_df, accounts_df = load_data()
    risk_df = load_risk_scores()
    alerts_df = load_alerts()
    rings_df = load_rings()

    page = st.sidebar.radio("Navigation", [
        "🌐 Network Overview",
        "🔴 Mule Rings",
        "⚠️ Structuring Alerts",
        "🔍 Account Lookup",
    ])

    if page == "🌐 Network Overview":
        render_overview(txns_df, accounts_df, risk_df, alerts_df, rings_df)
    elif page == "🔴 Mule Rings":
        render_rings(rings_df, txns_df)
    elif page == "⚠️ Structuring Alerts":
        render_structuring(alerts_df)
    elif page == "🔍 Account Lookup":
        render_account_lookup(txns_df, accounts_df, risk_df)


if __name__ == "__main__":
    main()
