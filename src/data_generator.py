import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

FINTRAC_THRESHOLD = 10_000.0
SEED = 42


def generate_network(n_normal=450, n_mule_rings=3, n_structuring=10, n_txns=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    accounts = []
    account_id = 1

    for _ in range(n_normal):
        accounts.append({
            "account_id": f"ACC{account_id:05d}",
            "account_type": "normal",
            "ring_id": None,
            "role": None,
        })
        account_id += 1

    rings = []
    for ring_num in range(n_mule_rings):
        ring_size = int(rng.integers(6, 13))
        ring_members = []
        n_collect = 1 if ring_size < 9 else 2
        n_cash = 1 if ring_size < 8 else 2
        roles = (["collector"] * n_collect + ["layerer"] * (ring_size - n_collect - n_cash)
                 + ["cashout"] * n_cash)
        for role in roles:
            accounts.append({
                "account_id": f"ACC{account_id:05d}",
                "account_type": "mule",
                "ring_id": f"RING{ring_num:02d}",
                "role": role,
            })
            ring_members.append(f"ACC{account_id:05d}")
            account_id += 1
        rings.append(ring_members)

    struct_accounts = []
    for _ in range(n_structuring):
        accounts.append({
            "account_id": f"ACC{account_id:05d}",
            "account_type": "structuring",
            "ring_id": None,
            "role": "structurer",
        })
        struct_accounts.append(f"ACC{account_id:05d}")
        account_id += 1

    accounts_df = pd.DataFrame(accounts)
    normal_ids = accounts_df[accounts_df["account_type"] == "normal"]["account_id"].tolist()

    transactions = []
    txn_id = 1
    base_time = datetime(2024, 1, 1, 8, 0, 0)

    n_normal_txns = int(n_txns * 0.60)
    for _ in range(n_normal_txns):
        sender = random.choice(normal_ids)
        receiver = random.choice(normal_ids)
        while receiver == sender:
            receiver = random.choice(normal_ids)
        amount = float(rng.lognormal(mean=5.5, sigma=1.2))
        amount = round(min(amount, 50_000), 2)
        offset = int(rng.integers(0, 60 * 24 * 90))
        ts = base_time + timedelta(minutes=offset)
        transactions.append({
            "txn_id": f"TXN{txn_id:07d}",
            "timestamp": ts,
            "sender_id": sender,
            "receiver_id": receiver,
            "amount_cad": amount,
            "txn_type": random.choice(["e-transfer", "wire", "cheque", "bill-payment"]),
            "channel": random.choice(["online", "branch", "atm"]),
            "is_suspicious": False,
        })
        txn_id += 1

    # Mule rings. Each ring runs a short burst (about 3 weeks):
    #   victims -> collector -> layerers (sometimes layerer -> layerer) -> cash-out -> ATM cash
    # Mules forward 85-98% of what they get within hours and often use round amounts.
    def add_txn(ts, sender, receiver, amount, txn_type, channel):
        nonlocal txn_id
        transactions.append({
            "txn_id": f"TXN{txn_id:07d}",
            "timestamp": ts,
            "sender_id": sender,
            "receiver_id": receiver,
            "amount_cad": amount,
            "txn_type": txn_type,
            "channel": channel,
            "is_suspicious": True,
        })
        txn_id += 1

    for ring_members in rings:
        ring_df = accounts_df[accounts_df["account_id"].isin(ring_members)]
        collectors = ring_df[ring_df["role"] == "collector"]["account_id"].tolist()
        layerers = ring_df[ring_df["role"] == "layerer"]["account_id"].tolist()
        cashouts = ring_df[ring_df["role"] == "cashout"]["account_id"].tolist()

        burst_start = int(rng.integers(0, 60 * 24 * 65))
        victims = random.sample(normal_ids, int(rng.integers(12, 26)))
        for victim in victims:
            ts = base_time + timedelta(minutes=burst_start + int(rng.integers(0, 60 * 24 * 21)))
            amount = round(float(rng.uniform(2_000, 15_000)), 2)
            collector = random.choice(collectors)
            add_txn(ts, victim, collector, amount, random.choice(["e-transfer", "wire"]), "online")

            # Collector splits the money across 2-3 layerers, round amounts
            splits = rng.dirichlet(np.ones(int(rng.integers(2, 4))))
            forwarded = amount * float(rng.uniform(0.88, 0.97))
            for share in splits:
                layerer = random.choice(layerers)
                hop_amount = max(100.0, round(forwarded * float(share), -2))
                ts2 = ts + timedelta(minutes=int(rng.integers(20, 6 * 60)))
                add_txn(ts2, collector, layerer, hop_amount, "e-transfer", "online")

                # Sometimes an extra layering hop between layerers
                if len(layerers) > 1 and rng.random() < 0.35:
                    nxt = random.choice([l for l in layerers if l != layerer])
                    ts2 = ts2 + timedelta(minutes=int(rng.integers(20, 4 * 60)))
                    hop_amount = round(hop_amount * float(rng.uniform(0.92, 0.99)), -2) or 100.0
                    add_txn(ts2, layerer, nxt, hop_amount, "e-transfer", "online")
                    layerer = nxt

                cashout = random.choice(cashouts)
                ts3 = ts2 + timedelta(minutes=int(rng.integers(30, 8 * 60)))
                out_amount = round(hop_amount * float(rng.uniform(0.92, 0.99)), -2) or 100.0
                add_txn(ts3, layerer, cashout, out_amount, "e-transfer", "online")

                # Cash-out mule pulls the money out at ATMs, often at night
                ts4 = ts3 + timedelta(minutes=int(rng.integers(30, 10 * 60)))
                add_txn(ts4, cashout, "CASH_WITHDRAWAL", round(out_amount * 0.98, -2) or 100.0,
                        "cash", "atm")

    for acc in struct_accounts:
        n_deposits = int(rng.integers(4, 15))
        offset_base = int(rng.integers(0, 60 * 24 * 60))
        for i in range(n_deposits):
            amount = round(float(rng.uniform(8_000, 9_900)), 2)
            ts = base_time + timedelta(minutes=offset_base + i * int(rng.integers(60 * 8, 60 * 48)))
            transactions.append({
                "txn_id": f"TXN{txn_id:07d}",
                "timestamp": ts,
                "sender_id": "CASH_DEPOSIT",
                "receiver_id": acc,
                "amount_cad": amount,
                "txn_type": "cash",
                "channel": "branch",
                "is_suspicious": True,
            })
            txn_id += 1

    txns_df = pd.DataFrame(transactions).sort_values("timestamp").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    txns_df.to_csv("data/transactions.csv", index=False)
    accounts_df.to_csv("data/accounts.csv", index=False)

    print(f"Generated {len(accounts_df)} accounts and {len(txns_df)} transactions")
    print(f"  Mule ring accounts : {len(accounts_df[accounts_df['account_type']=='mule'])}")
    print(f"  Structuring accounts: {len(struct_accounts)}")
    print(f"  Suspicious txns    : {txns_df['is_suspicious'].sum()}")

    return txns_df, accounts_df


if __name__ == "__main__":
    generate_network()
