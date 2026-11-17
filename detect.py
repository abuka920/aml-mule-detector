"""
AML Money Mule Network Detector — Main CLI
Usage: python detect.py [--generate-data] [--dashboard]
"""
import sys
import csv
import json
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_generator import generate_network
from src.graph_builder import build_graph, graph_summary
from src.structuring_detector import run_all_detectors
from src.community_detector import detect_mule_rings
from src.risk_scorer import score_accounts
from src.visualizer import visualize_network, visualize_ring

REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")


@click.command()
@click.option("--generate-data", is_flag=True, default=False,
              help="Regenerate synthetic transaction data")
@click.option("--dashboard", is_flag=True, default=False,
              help="Launch Streamlit dashboard after detection")
@click.option("--no-graph", is_flag=True, default=False,
              help="Skip network visualization (faster)")
def main(generate_data, dashboard, no_graph):
    """AML Money Mule Network Detector — FINTRAC-aligned detection pipeline."""

    REPORTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    # 1. Data
    txns_path = DATA_DIR / "transactions.csv"
    accs_path = DATA_DIR / "accounts.csv"

    if generate_data or not txns_path.exists():
        print("[1/5] Generating synthetic transaction network...")
        txns_df, accounts_df = generate_network()
    else:
        print(f"[1/5] Loading existing data from {DATA_DIR}/")
        txns_df = pd.read_csv(txns_path)
        accounts_df = pd.read_csv(accs_path)

    print(f"      {len(accounts_df)} accounts | {len(txns_df)} transactions")

    # 2. Build graph
    print("\n[2/5] Building transaction network graph...")
    G = build_graph(txns_df)
    summary = graph_summary(G)
    print(f"      Nodes: {summary['nodes']} | Edges: {summary['edges']} | "
          f"Volume: ${summary['total_volume_cad']:,.0f} CAD")

    # 3. Structuring detection
    print("\n[3/5] Running structuring / FINTRAC detectors...")
    alerts = run_all_detectors(txns_df)
    structuring = [a for a in alerts if a.alert_type == "structuring"]
    pass_through = [a for a in alerts if a.alert_type == "pass_through"]
    fan_in = [a for a in alerts if a.alert_type == "fan_in"]
    print(f"      Structuring alerts : {len(structuring)}")
    print(f"      Pass-through alerts: {len(pass_through)}")
    print(f"      Fan-in alerts      : {len(fan_in)}")

    alerts_rows = []
    for a in alerts:
        alerts_rows.append({
            "account_id": a.account_id,
            "alert_type": a.alert_type,
            "risk_score": a.risk_score,
            "evidence": a.evidence,
            "fintrac_rule_reference": a.fintrac_rule_reference,
        })
    pd.DataFrame(alerts_rows).to_csv(REPORTS_DIR / "structuring_alerts.csv", index=False)

    # 4. Mule ring detection
    print("\n[4/5] Detecting mule rings (seed-and-expand network analysis)...")
    rings = detect_mule_rings(G, txns_df, alerts)
    print(f"      Detected {len(rings)} suspicious rings")

    rings_rows = []
    for r in rings:
        rings_rows.append({
            "ring_id": r.ring_id,
            "ring_score": r.ring_score,
            "ring_type": r.ring_type,
            "member_count": len(r.member_account_ids),
            "member_account_ids": json.dumps(r.member_account_ids),
            "evidence_summary": r.evidence_summary,
            "internal_density": r.internal_density,
            "avg_pass_through_ratio": r.avg_pass_through_ratio,
            "round_amount_ratio": r.round_amount_ratio,
        })
    pd.DataFrame(rings_rows).to_csv(REPORTS_DIR / "mule_rings.csv", index=False)

    # 5. Risk scoring
    print("\n[5/5] Computing composite risk scores...")
    risk_scores = score_accounts(G, txns_df, alerts, rings)
    critical = sum(1 for r in risk_scores if r.risk_tier == "CRITICAL")
    high = sum(1 for r in risk_scores if r.risk_tier == "HIGH")
    medium = sum(1 for r in risk_scores if r.risk_tier == "MEDIUM")

    risk_rows = []
    for r in risk_scores:
        risk_rows.append({
            "account_id": r.account_id,
            "risk_score": r.risk_score,
            "risk_tier": r.risk_tier,
            "top_risk_factors": " | ".join(r.top_risk_factors),
        })
    pd.DataFrame(risk_rows).to_csv(REPORTS_DIR / "risk_scores.csv", index=False)

    print(f"      CRITICAL: {critical} | HIGH: {high} | MEDIUM: {medium}")

    # Visualize
    if not no_graph:
        print("\n      Generating network visualizations...")
        visualize_network(G, risk_scores, rings)
        for ring in rings[:5]:
            visualize_ring(G, ring, out_path=f"reports/ring_{ring.ring_id}.png")

    print("\n" + "=" * 60)
    print(f"  DETECTION COMPLETE")
    print(f"  Mule rings found      : {len(rings)}")
    print(f"  FINTRAC rule alerts   : {len(alerts)}")
    print(f"  CRITICAL / HIGH accts : {critical} / {high}")
    print(f"  Reports saved to      : {REPORTS_DIR}/")
    print("=" * 60)

    if dashboard:
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"])


if __name__ == "__main__":
    main()
