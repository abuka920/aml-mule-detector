# AML Money Mule Network Detector

A graph-based Anti-Money Laundering detection system that identifies mule account rings and structuring (smurfing) behaviour in synthetic banking transaction data, aligned with Canadian **FINTRAC** regulations.

## What It Detects

| Detection | Method | FINTRAC Reference |
|-----------|--------|-------------------|
| Structuring / Smurfing | Sliding-window cash deposit analysis | PCMLTFA s.9 — LCTR |
| Rapid pass-through (layering) | Re-sends 80–110% of incoming funds within 24h, 2+ times | PCMLTFA s.7 — STR |
| Fan-in collector accounts | 10+ unique senders in a rolling 30-day window | STR typology |
| Mule rings | Seed-and-expand network analysis from flagged accounts | FINTRAC ML indicators |

## Why Graph Analysis?

Rule-based systems flag individual accounts. Network analysis catches what they miss: **coordinated rings** where each account looks normal in isolation, but the collective flow pattern reveals layering behaviour.

A mule ring shows:
- High internal transaction density (money circulates within the ring)
- Low external connectivity (little legitimate economic activity)
- High pass-through ratios (collectors → layerers → cashouts)
- Round amounts and burst timing (coordinated movement)

## What Is Structuring?

Under Canada's **Proceeds of Crime (Money Laundering) and Terrorist Financing Act (PCMLTFA)**, financial institutions must file a **Large Cash Transaction Report (LCTR)** for cash transactions ≥ **$10,000 CAD**.

Structuring (smurfing) means deliberately breaking deposits into amounts just below $10,000 to avoid this reporting requirement. This tool detects accounts with 3+ such deposits within 10 days.

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic data + run full detection pipeline
python detect.py --generate-data

# Launch interactive dashboard
python detect.py --dashboard

# Or launch dashboard separately
streamlit run app/dashboard.py
```

## Detection Methodology

```
Transactions → NetworkX Graph → {
  Rule Detectors   → structuring, pass-through, fan-in alerts
  Ring Detector    → seed-and-expand from flagged accounts → Mule Rings
  Risk Scorer      → rule hit + network context + ring membership → Risk Tier
} → Streamlit Dashboard + CSV Reports
```

### Why seed-and-expand instead of plain community detection?

My first version ran community detection (greedy modularity) over the whole bank. It found
clusters, but they were clusters of ordinary customers, and **none of the reported rings
contained a real mule**. Most of a bank is normal customers, so whole-network clustering
mostly describes normal behaviour.

The fix mirrors how AML investigators actually work:

1. **Seeds** — accounts already flagged by a rule (pass-through or fan-in)
2. **Expand** — follow $1,000+ flows to other accounts that also forward most of what they receive
3. **Prune** — drop unflagged accounts linked to fewer than 2 flagged ones (usually victims)
4. **Ring** — every connected group of 3+ accounts left is a candidate ring

## Results on synthetic data

Ground truth is known for the synthetic data, so detection can be measured. Across 4 random seeds:

| Metric | Result |
|--------|--------|
| Mule rings found | 11 of 12 (one pair of rings merged into one) |
| Real mules inside detected rings | 100% |
| Innocent accounts pulled into rings | 0–2 per run |
| HIGH/CRITICAL accounts that are truly bad | 97–100% |
| Structuring accounts caught | 10 of 10 every run |

Synthetic data is cleaner than real banking data, so real-world numbers would be lower.
The point is the method, and that it is measured instead of assumed.

## Dashboard Pages

| Page | Description |
|------|-------------|
| Network Overview | Full transaction graph, colour-coded by risk tier |
| Mule Rings | Detected rings with subgraph visualizations |
| Structuring Alerts | FINTRAC-style alert table sortable by risk score |
| Account Lookup | Search any account, see ego network + history |

## Tech Stack

- **NetworkX** — graph construction and community detection
- **Pandas** — transaction data manipulation
- **Matplotlib** — network visualization
- **Streamlit** — analyst dashboard
- **Greedy Modularity Communities** — `networkx.algorithms.community`
