import pandas as pd
import networkx as nx
from typing import Optional


def build_graph(txns_df: pd.DataFrame) -> nx.DiGraph:
    """Build directed weighted graph: nodes=accounts, edges=aggregate flows."""
    G = nx.DiGraph()

    CASH_NODES = {"CASH_DEPOSIT", "CASH_WITHDRAWAL"}
    real_txns = txns_df[
        ~txns_df["sender_id"].isin(CASH_NODES) & ~txns_df["receiver_id"].isin(CASH_NODES)
    ].copy()

    # Add all account nodes
    all_accounts = set(txns_df["sender_id"].unique()) | set(txns_df["receiver_id"].unique())
    all_accounts -= CASH_NODES
    for acc in all_accounts:
        G.add_node(acc)

    # Aggregate edges
    edge_data = (
        real_txns.groupby(["sender_id", "receiver_id"])
        .agg(
            transaction_count=("txn_id", "count"),
            total_amount=("amount_cad", "sum"),
            avg_amount=("amount_cad", "mean"),
            min_ts=("timestamp", "min"),
            max_ts=("timestamp", "max"),
        )
        .reset_index()
    )

    for _, row in edge_data.iterrows():
        try:
            ts_min = pd.Timestamp(row["min_ts"])
            ts_max = pd.Timestamp(row["max_ts"])
            span_hours = (ts_max - ts_min).total_seconds() / 3600
        except Exception:
            span_hours = 0.0

        G.add_edge(
            row["sender_id"],
            row["receiver_id"],
            weight=float(row["total_amount"]),
            transaction_count=int(row["transaction_count"]),
            total_amount=float(row["total_amount"]),
            avg_amount=float(row["avg_amount"]),
            time_span_hours=round(span_hours, 2),
        )

    # Node attributes
    for node in G.nodes():
        sent = sum(d["total_amount"] for _, _, d in G.out_edges(node, data=True))
        received = sum(d["total_amount"] for _, _, d in G.in_edges(node, data=True))
        G.nodes[node]["total_sent"] = round(sent, 2)
        G.nodes[node]["total_received"] = round(received, 2)
        G.nodes[node]["in_degree"] = G.in_degree(node)
        G.nodes[node]["out_degree"] = G.out_degree(node)

    # Betweenness centrality (expensive — use approximate for large graphs)
    if G.number_of_nodes() < 2000:
        try:
            bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
            nx.set_node_attributes(G, bc, "betweenness_centrality")
        except Exception:
            nx.set_node_attributes(G, 0.0, "betweenness_centrality")
    else:
        nx.set_node_attributes(G, 0.0, "betweenness_centrality")

    return G


def get_subgraph(G: nx.DiGraph, account_ids: list) -> nx.DiGraph:
    return G.subgraph(account_ids).copy()


def get_ego_network(G: nx.DiGraph, account_id: str, radius: int = 2) -> nx.DiGraph:
    """Return the ego network of `account_id` up to `radius` hops."""
    undirected = G.to_undirected()
    try:
        ego_nodes = nx.ego_graph(undirected, account_id, radius=radius).nodes()
        return G.subgraph(ego_nodes).copy()
    except nx.NodeNotFound:
        return nx.DiGraph()


def graph_summary(G: nx.DiGraph) -> dict:
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "total_volume_cad": round(
            sum(d["total_amount"] for _, _, d in G.edges(data=True)), 2
        ),
        "avg_degree": round(
            sum(d for _, d in G.degree()) / max(G.number_of_nodes(), 1), 2
        ),
    }
