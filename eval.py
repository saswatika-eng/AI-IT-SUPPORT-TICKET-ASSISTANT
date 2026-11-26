import os
import time
import pandas as pd
from classifier import analyze_ticket
from retriever import search_kb

THRESHOLD = 70
OUT = "data/results.csv"

UNKNOWN_IDS = {"T047", "T048", "T049", "T050"}

tickets = pd.read_csv("data/tickets.csv")
rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
done_ids = {r["ticket_id"] for r in rows}

print(
    f"Already done: {len(done_ids)} tickets. Remaining: {len(tickets) - len(done_ids)}")

for t in tickets.itertuples():
    if t.ticket_id in done_ids:
        continue
    try:
        articles = search_kb(t.text, 5)
        r = analyze_ticket(t.text, articles)
    except Exception as e:
        print(f"\nStopped at {t.ticket_id}: {str(e)[:200]}")
        print("Quota khatam ho gaya hoga. Baad mein dobara `python eval.py` chalao, wahin se continue hoga.")
        break
    rows.append({
        "ticket_id": t.ticket_id,
        "pred_category": r["category"],
        "pred_priority": r["priority"],
        "kb_id": r["kb_id"],
        "confidence": int(float(r["confidence"])),
    })
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{t.ticket_id} done")
    time.sleep(5) 

if rows:
    df = pd.DataFrame(rows).merge(tickets, on="ticket_id")
    df["escalated"] = (df["kb_id"] == "NONE") | (df["confidence"] < THRESHOLD)
    df["should_escalate"] = df["ticket_id"].isin(UNKNOWN_IDS)
    print(f"\nTickets evaluated: {len(df)} of {len(tickets)}")
    print(
        f"Category accuracy:   {(df.pred_category == df.true_category).mean():.0%}")
    print(
        f"Priority accuracy:   {(df.pred_priority == df.true_priority).mean():.0%}")
    print(
        f"Escalation accuracy: {(df.escalated == df.should_escalate).mean():.0%}")
