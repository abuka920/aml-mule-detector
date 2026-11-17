import os
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .risk_scorer import AccountRisk
from .community_detector import MuleRing

TIER_COLORS = {
    "CRITICAL": "#D62728",
    "HIGH":     "#FF7F0E",
    "MEDIUM":   "#FDD835",
    "LOW":      "#2CA02C",
    "UNKNOWN":  "#AEC7E8",
}


def _get_node_color(acc_id: str, risk_map: dict) -> str:
    tier = risk_map.get(acc_id, {}).get("tier", "UNKNOWN")
    return TIER_COLORS.get(tier, TIER_COLORS["UNKNOWN"])


def visualize_network(
    G: nx.DiGraph,
    risk_scores: list[AccountRisk],
    rings: list[MuleRing],
    out_path: str = "reports/network_graph.png",
    max_nodes: int = 200,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    risk_map = {r.account_id: {"score": r.risk_score, "tier": r.risk_tier} for r in risk_scores}
    ring_members: set = set()
    for ring in rings:
        ring_members.update(ring.member_account_ids)

    # Subsample for readability
    nodes_to_show = list(G.nodes())[:max_nodes]
    subG = G.subgraph(nodes_to_show).copy()

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    pos = nx.spring_layout(subG, k=0.8, seed=42, weight="weight")

    node_colors = [_get_node_color(n, risk_map) for n in subG.nodes()]
    node_sizes = []
    for n in subG.nodes():
        vol = G.nodes[n].get("total_sent", 0) + G.nodes[n].get("total_received", 0)
        node_sizes.append(max(50, min(1500, vol / 1000)))

    # Draw ring member borders
    ring_nodes = [n for n in subG.nodes() if n in ring_members]
    if ring_nodes:
        ring_pos = {n: pos[n] for n in ring_nodes}
        nx.draw_networkx_nodes(
            subG, ring_pos,
            nodelist=ring_nodes,
            node_color="white",
            node_size=[s + 200 for s in [
                max(50, min(1500, (G.nodes[n].get("total_sent", 0) + G.nodes[n].get("total_received", 0)) / 1000))
                for n in ring_nodes
            ]],
            ax=ax,
        )

    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)

    edge_weights = [min(d.get("total_amount", 1) / 10_000, 3.0) for _, _, d in subG.edges(data=True)]
    nx.draw_networkx_edges(subG, pos, width=edge_weights, alpha=0.3, edge_color="#8888aa",
                           arrows=True, arrowsize=10, ax=ax)

    # Label only high-risk nodes
    high_risk_labels = {
        n: n for n in subG.nodes()
        if risk_map.get(n, {}).get("tier", "") in ("CRITICAL", "HIGH")
    }
    if high_risk_labels:
        nx.draw_networkx_labels(subG, pos, labels=high_risk_labels,
                                font_size=6, font_color="white", ax=ax)

    legend_handles = [
        mpatches.Patch(color=color, label=tier)
        for tier, color in TIER_COLORS.items() if tier != "UNKNOWN"
    ]
    legend_handles.append(
        mpatches.Patch(color="white", label="Mule Ring Member (white border)")
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9,
              facecolor="#2d2d44", labelcolor="white")

    ax.set_title(
        f"AML Transaction Network — {subG.number_of_nodes()} accounts, "
        f"{subG.number_of_edges()} flows | {len(rings)} rings detected",
        color="white", fontsize=13, pad=12,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Network graph saved to {out_path}")


def visualize_ring(
    G: nx.DiGraph,
    ring: MuleRing,
    out_path: str,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    subG = G.subgraph(ring.member_account_ids).copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    pos = nx.circular_layout(subG)
    nx.draw_networkx_nodes(subG, pos, node_color="#D62728", node_size=600, ax=ax)
    nx.draw_networkx_labels(subG, pos, font_size=7, font_color="white", ax=ax)
    nx.draw_networkx_edges(subG, pos, edge_color="#FF7F0E", arrows=True, arrowsize=15,
                           width=2, ax=ax)

    ax.set_title(
        f"Mule Ring {ring.ring_id} — Score: {ring.ring_score:.0f}/100 ({ring.ring_type})",
        color="white", fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
