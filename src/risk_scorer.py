from dataclasses import dataclass
import pandas as pd
import networkx as nx
from .structuring_detector import StructuringAlert
from .community_detector import MuleRing

WEIGHTS = {
    "structuring": 0.35,
    "pass_through": 0.25,
    "centrality": 0.20,
    "unusual_hours": 0.10,
    "round_amount": 0.10,
}

RISK_TIERS = [
    (80, "CRITICAL"),
    (60, "HIGH"),
    (40, "MEDIUM"),
    (0,  "LOW"),
]


@dataclass
class AccountRisk:
    account_id: str
    risk_score: float
    risk_tier: str
    top_risk_factors: list


def _tier(score: float) -> str:
    for threshold, label in RISK_TIERS:
        if score >= threshold:
            return label
    return "LOW"


def score_accounts(
    G: nx.DiGraph,
    txns_df: pd.DataFrame,
    alerts: list[StructuringAlert],
    rings: list[MuleRing],
) -> list[AccountRisk]:
    all_accounts = list(G.nodes())
    if not all_accounts:
        return []

    # Pre-build lookup maps
    structuring_scores: dict[str, float] = {}
    pass_through_scores: dict[str, float] = {}
    for a in alerts:
        if a.alert_type == "structuring":
            structuring_scores[a.account_id] = max(
                structuring_scores.get(a.account_id, 0), a.risk_score
            )
        elif a.alert_type == "pass_through":
            pass_through_scores[a.account_id] = max(
                pass_through_scores.get(a.account_id, 0), a.risk_score
            )

    ring_members: set[str] = set()
    for ring in rings:
        ring_members.update(ring.member_account_ids)

    # Centrality (betweenness already computed on G)
    bc_values = nx.get_node_attributes(G, "betweenness_centrality")
    max_bc = max(bc_values.values(), default=1.0) or 1.0

    # Unusual hours (fraction of transactions outside 8–20h)
    txns = txns_df[(txns_df["sender_id"] != "CASH_DEPOSIT") &
                   (txns_df["receiver_id"] != "CASH_WITHDRAWAL")].copy()
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    txns["hour"] = txns["timestamp"].dt.hour
    txns["is_unusual_hour"] = ~txns["hour"].between(8, 20)

    unusual_hour_ratio: dict[str, float] = {}
    for acc_id in all_accounts:
        sub = txns[txns["sender_id"] == acc_id]
        if not sub.empty:
            unusual_hour_ratio[acc_id] = sub["is_unusual_hour"].mean()

    # Round amount ratio per account
    round_ratio: dict[str, float] = {}
    for acc_id in all_accounts:
        sub = txns[txns["sender_id"] == acc_id]
        if not sub.empty:
            round_ratio[acc_id] = sub["amount_cad"].apply(
                lambda x: (x % 100 == 0) or (x % 1000 == 0)
            ).mean()

    fan_in_scores: dict[str, float] = {}
    for a in alerts:
        if a.alert_type == "fan_in":
            fan_in_scores[a.account_id] = max(fan_in_scores.get(a.account_id, 0), a.risk_score)

    results = []
    for acc_id in all_accounts:
        s_score = structuring_scores.get(acc_id, 0.0) / 100.0
        pt_score = pass_through_scores.get(acc_id, 0.0) / 100.0
        fi_score = fan_in_scores.get(acc_id, 0.0) / 100.0
        c_score = bc_values.get(acc_id, 0.0) / max_bc
        uh_score = unusual_hour_ratio.get(acc_id, 0.0)
        ra_score = round_ratio.get(acc_id, 0.0)

        # A confirmed rule hit drives the score; context only adds a little.
        # A weighted average would water a strong signal down so far that a
        # real mule could never reach HIGH.
        rule_signal = max(s_score, pt_score, fi_score)
        context = 0.5 * c_score + 0.25 * uh_score + 0.25 * ra_score
        composite = 70 * rule_signal + 15 * context
        if acc_id in ring_members:
            composite += 20  # part of a detected ring
        composite = min(100.0, composite)

        factors = sorted([
            ("Structuring / LCTR deposits", s_score * 100),
            ("Pass-through ratio", pt_score * 100),
            ("Fan-in (many senders)", fi_score * 100),
            ("Network centrality", c_score * 100),
            ("Unusual transaction hours", uh_score * 100),
            ("Round amount transactions", ra_score * 100),
        ], key=lambda x: -x[1])

        top_factors = [f"{name}: {val:.0f}/100" for name, val in factors if val > 5][:3]
        if acc_id in ring_members:
            top_factors.insert(0, "Member of detected mule ring")

        results.append(AccountRisk(
            account_id=acc_id,
            risk_score=round(composite, 1),
            risk_tier=_tier(composite),
            top_risk_factors=top_factors,
        ))

    return sorted(results, key=lambda r: -r.risk_score)
