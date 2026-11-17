from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

FINTRAC_THRESHOLD = 10_000.0
STRUCTURING_WINDOW_DAYS = 10
STRUCTURING_MIN_COUNT = 3
STRUCTURING_LOWER_BOUND = 8_000.0
PASS_THROUGH_RATIO_THRESHOLD = 0.80
PASS_THROUGH_RATIO_MAX = 1.10          # sending far MORE than you got is just spending
PASS_THROUGH_MIN_AMOUNT = 1_000.0      # ignore small everyday transfers
PASS_THROUGH_MIN_EVENTS = 2            # one coincidence is noise, a pattern is a signal
PASS_THROUGH_WINDOW_HOURS = 24
FAN_IN_MIN_SOURCES = 10
FAN_IN_WINDOW_DAYS = 30


@dataclass
class StructuringAlert:
    account_id: str
    alert_type: str          # "structuring", "pass_through", "fan_in"
    risk_score: float        # 0–100
    evidence: str
    fintrac_rule_reference: str
    deposit_amounts: list = field(default_factory=list)
    deposit_timestamps: list = field(default_factory=list)


def detect_structuring(txns_df: pd.DataFrame) -> list[StructuringAlert]:
    """
    Detect repeated cash deposits just below the FINTRAC $10,000 CAD reporting threshold.
    FINTRAC PCMLTFA s.9: Mandatory reporting of cash transactions >= $10,000 CAD.
    """
    alerts = []
    cash_txns = txns_df[
        (txns_df["txn_type"] == "cash") &
        (txns_df["amount_cad"] >= STRUCTURING_LOWER_BOUND) &
        (txns_df["amount_cad"] < FINTRAC_THRESHOLD) &
        (txns_df["sender_id"] == "CASH_DEPOSIT")
    ].copy()

    if cash_txns.empty:
        return alerts

    cash_txns["timestamp"] = pd.to_datetime(cash_txns["timestamp"])

    for acc_id, group in cash_txns.groupby("receiver_id"):
        group = group.sort_values("timestamp")
        # Sliding 10-day window
        for i in range(len(group)):
            window_start = group.iloc[i]["timestamp"]
            window_end = window_start + pd.Timedelta(days=STRUCTURING_WINDOW_DAYS)
            window_txns = group[
                (group["timestamp"] >= window_start) &
                (group["timestamp"] <= window_end)
            ]
            if len(window_txns) >= STRUCTURING_MIN_COUNT:
                total = window_txns["amount_cad"].sum()
                amounts = window_txns["amount_cad"].tolist()
                risk_score = min(100.0, 50 + len(window_txns) * 5 + (total / FINTRAC_THRESHOLD) * 10)
                alerts.append(StructuringAlert(
                    account_id=acc_id,
                    alert_type="structuring",
                    risk_score=round(risk_score, 1),
                    evidence=(
                        f"{len(window_txns)} cash deposits totalling ${total:,.2f} CAD "
                        f"in {STRUCTURING_WINDOW_DAYS} days "
                        f"(all between ${STRUCTURING_LOWER_BOUND:,.0f}–${FINTRAC_THRESHOLD:,.0f}). "
                        f"Deposits: {[f'${a:,.2f}' for a in amounts]}"
                    ),
                    fintrac_rule_reference="PCMLTFA s.9 — Large Cash Transaction Report (LCTR)",
                    deposit_amounts=amounts,
                    deposit_timestamps=window_txns["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
                ))
                break  # one alert per account

    return alerts


def detect_pass_through(txns_df: pd.DataFrame) -> list[StructuringAlert]:
    """
    Detect layering: an account receives money and sends 80-110% of it back out
    within 24 hours, at least twice. Outbound includes cash withdrawals.
    """
    alerts = []
    txns = txns_df[txns_df["sender_id"] != "CASH_DEPOSIT"].copy()
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    window = pd.Timedelta(hours=PASS_THROUGH_WINDOW_HOURS)

    inbound_by_acc = {a: g for a, g in txns.groupby("receiver_id")}
    outbound_by_acc = {a: g for a, g in txns.groupby("sender_id")}

    for acc_id, inbound in inbound_by_acc.items():
        if acc_id == "CASH_WITHDRAWAL" or acc_id not in outbound_by_acc:
            continue
        outbound = outbound_by_acc[acc_id]
        events = []
        for _, in_row in inbound[inbound["amount_cad"] >= PASS_THROUGH_MIN_AMOUNT].iterrows():
            t0, received = in_row["timestamp"], in_row["amount_cad"]
            sent = outbound[(outbound["timestamp"] >= t0) &
                            (outbound["timestamp"] <= t0 + window)]["amount_cad"].sum()
            ratio = sent / received
            if PASS_THROUGH_RATIO_THRESHOLD <= ratio <= PASS_THROUGH_RATIO_MAX:
                events.append((received, sent, ratio))

        if len(events) >= PASS_THROUGH_MIN_EVENTS:
            total_in = sum(e[0] for e in events)
            total_out = sum(e[1] for e in events)
            avg_ratio = total_out / total_in
            risk_score = min(100.0, 55 + len(events) * 4 + avg_ratio * 15)
            alerts.append(StructuringAlert(
                account_id=acc_id,
                alert_type="pass_through",
                risk_score=round(risk_score, 1),
                evidence=(
                    f"{len(events)} times received money and re-sent {avg_ratio:.0%} of it "
                    f"within {PASS_THROUGH_WINDOW_HOURS}h (${total_in:,.0f} in, ${total_out:,.0f} out). "
                    f"Classic layering pattern."
                ),
                fintrac_rule_reference="PCMLTFA s.7 — Suspicious Transaction Report (STR): Rapid pass-through",
            ))

    return alerts


def detect_fan_in(txns_df: pd.DataFrame) -> list[StructuringAlert]:
    """
    Detect collector accounts: money arriving from many different senders
    inside any rolling 30-day window.
    """
    alerts = []
    txns = txns_df[~txns_df["sender_id"].isin(["CASH_DEPOSIT"]) &
                   (txns_df["receiver_id"] != "CASH_WITHDRAWAL")].copy()
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    window = pd.Timedelta(days=FAN_IN_WINDOW_DAYS)

    for acc_id, group in txns.groupby("receiver_id"):
        if group["sender_id"].nunique() < FAN_IN_MIN_SOURCES:
            continue
        group = group.sort_values("timestamp")
        best, best_total = 0, 0.0
        for t0 in group["timestamp"]:
            w = group[(group["timestamp"] >= t0) & (group["timestamp"] < t0 + window)]
            n = w["sender_id"].nunique()
            if n > best:
                best, best_total = n, w["amount_cad"].sum()
        if best >= FAN_IN_MIN_SOURCES:
            risk_score = min(100.0, 45 + best * 3)
            alerts.append(StructuringAlert(
                account_id=acc_id,
                alert_type="fan_in",
                risk_score=round(risk_score, 1),
                evidence=(
                    f"Received funds from {best} different accounts "
                    f"totalling ${best_total:,.2f} CAD within {FAN_IN_WINDOW_DAYS} days. "
                    f"Collector node pattern."
                ),
                fintrac_rule_reference="FINTRAC money laundering indicators: third-party deposits and mule accounts",
            ))

    return alerts


def run_all_detectors(txns_df: pd.DataFrame) -> list[StructuringAlert]:
    structuring = detect_structuring(txns_df)
    pass_through = detect_pass_through(txns_df)
    fan_in = detect_fan_in(txns_df)

    seen = set()
    deduped = []
    for alert in structuring + pass_through + fan_in:
        key = (alert.account_id, alert.alert_type)
        if key not in seen:
            seen.add(key)
            deduped.append(alert)

    return sorted(deduped, key=lambda a: -a.risk_score)
