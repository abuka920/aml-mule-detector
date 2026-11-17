from dataclasses import dataclass, field
import networkx as nx
import pandas as pd


@dataclass
class MuleRing:
    ring_id: str
    member_account_ids: list
    ring_score: float          # 0–100
    ring_type: str             # "layering_ring", "cash_out_ring", "mixed"
    evidence_summary: str
    internal_density: float
    avg_pass_through_ratio: float
    round_amount_ratio: float


def _pass_through_ratio(G: nx.DiGraph, node: str) -> float:
    in_vol = sum(d["total_amount"] for _, _, d in G.in_edges(node, data=True))
    out_vol = sum(d["total_amount"] for _, _, d in G.out_edges(node, data=True))
    if in_vol == 0:
        return 0.0
    return min(out_vol / in_vol, 1.0)


def _round_amount_ratio(txns_df: pd.DataFrame, account_ids: list) -> float:
    """Fraction of transactions with round amounts (multiples of 100 or 1000)."""
    sub = txns_df[
        (txns_df["sender_id"].isin(account_ids)) |
        (txns_df["receiver_id"].isin(account_ids))
    ]
    if sub.empty:
        return 0.0
    round_count = sub["amount_cad"].apply(
        lambda x: (x % 100 == 0) or (x % 1000 == 0)
    ).sum()
    return round(round_count / len(sub), 3)


def _flow_ratios(txns_df: pd.DataFrame) -> dict:
    """Lifetime money-out / money-in per account (cash withdrawals count as money out)."""
    txns = txns_df[txns_df["sender_id"] != "CASH_DEPOSIT"]
    inflow = txns.groupby("receiver_id")["amount_cad"].sum()
    outflow = txns.groupby("sender_id")["amount_cad"].sum()
    ratios = {}
    for acc, vol_in in inflow.items():
        if vol_in > 0:
            ratios[acc] = (min(outflow.get(acc, 0.0) / vol_in, 1.5), float(vol_in))
    return ratios


def detect_mule_rings(
    G: nx.DiGraph,
    txns_df: pd.DataFrame,
    alerts: list | None = None,
    min_members: int = 3,
    min_ring_score: float = 50.0,
    min_edge_amount: float = 1_000.0,
) -> list[MuleRing]:
    """
    Seed-and-expand ring detection (how bank AML teams usually work):

    1. Seeds = accounts already flagged for pass-through or fan-in.
    2. Keep only money flows of $1,000+ between accounts that both look like
       conduits: a seed, or an account that forwards 75%+ of everything it gets.
    3. Drop unflagged accounts that touch fewer than 2 flagged ones (victims).
    4. Each connected group of 3+ remaining accounts is a candidate ring.

    Running community detection on the whole bank instead just finds clusters
    of ordinary customers, which is noise.
    """
    alerts = alerts or []
    seeds = {a.account_id for a in alerts if a.alert_type in ("pass_through", "fan_in")}
    ratios = _flow_ratios(txns_df)

    def is_conduit(acc: str) -> bool:
        if acc in seeds:
            return True
        ratio, vol_in = ratios.get(acc, (0.0, 0.0))
        return ratio >= 0.75 and vol_in >= 5_000

    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d.get("total_amount", 0) >= min_edge_amount and is_conduit(u) and is_conduit(v):
            H.add_edge(u, v)

    # Prune: an account nobody flagged must link to 2+ flagged accounts to stay.
    # This drops victims, who usually pay a single collector once.
    changed = True
    while changed:
        changed = False
        for node in list(H.nodes()):
            if node not in seeds and sum(n in seeds for n in H.neighbors(node)) < 2:
                H.remove_node(node)
                changed = True

    rings = []
    for idx, component in enumerate(nx.connected_components(H)):
        members = sorted(component)
        if len(members) < min_members:
            continue
        seed_share = sum(m in seeds for m in members) / len(members)
        if seed_share < 0.3:
            continue

        subG = G.subgraph(members)
        possible_edges = len(members) * (len(members) - 1)
        actual_internal = subG.number_of_edges()
        internal_density = actual_internal / max(possible_edges, 1)

        pt_ratios = [min(ratios.get(m, (0.0, 0.0))[0], 1.0) for m in members]
        avg_pt = sum(pt_ratios) / len(pt_ratios)
        round_ratio = _round_amount_ratio(txns_df, members)

        score = min(100.0, (avg_pt * 40 + seed_share * 30 + round_ratio * 15
                            + min(internal_density * 5, 1.0) * 15))
        if score < min_ring_score:
            continue

        fan_in_members = [a.account_id for a in alerts
                          if a.alert_type == "fan_in" and a.account_id in component]
        if fan_in_members and avg_pt > 0.7:
            ring_type = "collection_and_layering"
        elif avg_pt > 0.7:
            ring_type = "layering_ring"
        else:
            ring_type = "mixed"

        evidence_summary = (
            f"{len(members)} accounts moving money between each other in {actual_internal} flows. "
            f"{seed_share:.0%} already flagged by FINTRAC rules. "
            f"Avg pass-through: {avg_pt:.0%}. Round amounts: {round_ratio:.0%}. "
            f"Collectors: {', '.join(fan_in_members) or 'none flagged'}."
        )

        rings.append(MuleRing(
            ring_id=f"RING_{len(rings):03d}",
            member_account_ids=members,
            ring_score=round(score, 1),
            ring_type=ring_type,
            evidence_summary=evidence_summary,
            internal_density=round(internal_density, 4),
            avg_pass_through_ratio=round(avg_pt, 4),
            round_amount_ratio=round(round_ratio, 4),
        ))

    rings = sorted(rings, key=lambda r: -r.ring_score)
    for i, r in enumerate(rings):
        r.ring_id = f"RING_{i:03d}"
    return rings
