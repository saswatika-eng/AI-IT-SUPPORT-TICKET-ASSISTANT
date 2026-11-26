import html
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from classifier import analyze_ticket
from retriever import search_kb

CACHE_FILE = "data/cache.json"
HISTORY_FILE = "data/history.csv"
RESULTS_FILE = "data/results.csv"
UNKNOWN_IDS = {"T047", "T048", "T049", "T050"}  # tickets with no KB article

# Which team a ticket would go to (used only to suggest a default in the demo)
TEAMS = [
    "Network team",
    "Identity & Access team",
    "Messaging team",
    "Hardware support",
    "Application support",
    "Security team",
]
CATEGORY_TO_TEAM = {
    "Network/VPN": "Network team",
    "Account/Login": "Identity & Access team",
    "Access Request": "Identity & Access team",
    "Email": "Messaging team",
    "Hardware": "Hardware support",
    "Software": "Application support",
    "Security": "Security team",
}

# Colours chosen to be readable on the dark theme
BLUE = "#4C8DFF"
RED = "#FF6B6B"
GREEN = "#4CC38A"
PRIORITY_COLOR = {"High": "#FF6B6B", "Medium": "#FFB020", "Low": "#4CC38A"}

st.set_page_config(page_title="SafeDesk AI", page_icon="🛡️", layout="wide")

st.title("🛡️ SafeDesk AI")
st.caption(
    "AI IT Support Ticket Assistant: answers when sure, "
    "hands over to a human engineer when unsure or when the risk is high."
)

threshold = st.sidebar.slider(
    "Confidence threshold for auto-resolve (%)", 0, 100, 70)
minutes_saved = st.sidebar.number_input(
    "Minutes saved per auto-resolved ticket (assumption)", 1, 120, 15
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works**\n\n"
    "1. Search the knowledge base\n\n"
    "2. Gemini drafts the answer\n\n"
    "3. Rules decide: auto-resolve or hand over to an engineer"
)


# ---------------- helpers ----------------
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_analysis(ticket, articles):
    """Use the saved result if we have one. Otherwise call the AI."""
    cache = load_cache()
    key = ticket.strip().lower()
    if key in cache:
        return cache[key], True
    result = analyze_ticket(ticket, articles)
    cache[key] = result
    save_cache(cache)
    return result, False


def decide(result, threshold):
    conf = int(float(result["confidence"]))
    if result["kb_id"] == "NONE":
        return True, "No matching knowledge base article was found."
    if conf < threshold:
        return True, f"Confidence ({conf}%) is below the threshold ({threshold}%)."
    if result["category"] == "Security":
        return True, "Security issue needs the security team."
    return False, "A matching article was found with high confidence."


def log_history(ticket, result, escalate):
    row = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticket": ticket,
        "category": result["category"],
        "priority": result["priority"],
        "kb_id": result["kb_id"],
        "confidence": int(float(result["confidence"])),
        "escalated": escalate,
    }])
    row.to_csv(HISTORY_FILE, mode="a",
               header=not os.path.exists(HISTORY_FILE), index=False)


def card(label, value, color):
    """A small coloured card shown as HTML."""
    return (
        f'<div style="border-left:6px solid {color}; background:rgba(128,128,128,0.10);'
        f' padding:12px 16px; border-radius:8px;">'
        f'<div style="font-size:0.8rem; opacity:0.75;">{html.escape(label)}</div>'
        f'<div style="font-size:1.4rem; font-weight:700; color:{color};">'
        f'{html.escape(str(value))}</div></div>'
    )


tab1, tab2, tab3 = st.tabs(["Analyze Ticket", "Dashboard", "Accuracy Report"])

# ---------------- TAB 1: Analyze ----------------
with tab1:
    samples = pd.read_csv("data/tickets.csv")
    choice = st.selectbox("Pick a sample ticket (optional)", [
                          ""] + samples["text"].tolist())
    ticket = st.text_area("Or paste a support ticket here", value=choice)

    if st.button("Analyze"):
        if not ticket.strip():
            st.warning("Please enter a ticket first.")
        else:
            try:
                articles = search_kb(ticket, top_k=5)
                result, from_cache = get_analysis(ticket, articles)
                escalate, _ = decide(result, threshold)
                log_history(ticket, result, escalate)
                # Keep the result so it stays on screen when other buttons are clicked
                st.session_state["last"] = {
                    "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
                    "ticket": ticket,
                    "result": result,
                    "articles": articles,
                    "from_cache": from_cache,
                }
            except Exception as e:
                st.session_state.pop("last", None)
                st.error(f"Something went wrong: {e}")

    last = st.session_state.get("last")
    if last:
        result = last["result"]
        articles = last["articles"]
        escalate, why = decide(result, threshold)
        conf = int(float(result["confidence"]))

        if last["from_cache"]:
            st.caption("Loaded from saved results (no AI request used).")

        st.subheader("Result")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            card("Category", result["category"], BLUE), unsafe_allow_html=True)
        c2.markdown(
            card("Priority", result["priority"], PRIORITY_COLOR.get(
                result["priority"], BLUE)),
            unsafe_allow_html=True,
        )
        c3.markdown(
            card("Confidence", f"{conf}%",
                 GREEN if conf >= threshold else RED),
            unsafe_allow_html=True,
        )
        c4.markdown(
            card(
                "Decision",
                "ESCALATED" if escalate else "AUTO-RESOLVED",
                RED if escalate else GREEN,
            ),
            unsafe_allow_html=True,
        )

        st.write("")
        st.write("**Why:**", why)
        st.write("**Article used:**", result["kb_id"])
        st.write("**AI notes:**", result["reason"])

        st.subheader("Reply to employee")
        st.info(result["reply"])

        if escalate:
            st.subheader("Summary for the engineer")
            summary = (
                f"Ticket: {last['ticket']}\n"
                f"Category: {result['category']}\n"
                f"Priority: {result['priority']}\n"
                f"AI notes: {result['reason']}\n"
                f"Closest article: {result['kb_id']} ({conf}% confidence)\n"
                f"Escalation reason: {why}"
            )
            st.code(summary, language=None)

            st.subheader("Hand over to a human")
            default_team = CATEGORY_TO_TEAM.get(result["category"], TEAMS[0])
            team = st.selectbox(
                "Assign to",
                TEAMS,
                index=TEAMS.index(default_team),
                key=f"team_{last['id']}",
            )
            assigned = st.session_state.setdefault("assigned", {})
            if st.button("Assign to engineer", key=f"assign_{last['id']}"):
                assigned[last["id"]] = team
            if last["id"] in assigned:
                st.success(
                    f"Ticket assigned to {assigned[last['id']]} "
                    "(demo only - no real ticketing system is connected)."
                )

        with st.expander("Articles found by search"):
            st.dataframe(articles[["id", "title", "score"]])

# ---------------- TAB 2: Dashboard ----------------
with tab2:
    if os.path.exists(HISTORY_FILE):
        h = pd.read_csv(HISTORY_FILE)
        total = len(h)
        esc = int(h["escalated"].sum())
        auto = total - esc
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickets handled", total)
        c2.metric("Auto-resolved", auto)
        c3.metric("Escalated", esc)
        c4.metric("Est. time saved (min)", auto * minutes_saved)
        st.subheader("Tickets by category")
        st.bar_chart(h["category"].value_counts())
        st.subheader("Tickets by priority")
        st.bar_chart(h["priority"].value_counts())
        st.subheader("Recent tickets")
        st.dataframe(h.tail(10), use_container_width=True)
    else:
        st.info("No tickets analyzed yet. Analyze a ticket in the first tab.")

# ---------------- TAB 3: Accuracy Report ----------------
with tab3:
    if os.path.exists(RESULTS_FILE):
        res = pd.read_csv(RESULTS_FILE)
        tk = pd.read_csv("data/tickets.csv")
        df = res.merge(tk, on="ticket_id")
        df["escalated"] = (df["kb_id"] == "NONE") | (
            df["confidence"] < threshold)
        df["should_escalate"] = df["ticket_id"].isin(UNKNOWN_IDS)
        cat_acc = (df["pred_category"] == df["true_category"]).mean()
        pri_acc = (df["pred_priority"] == df["true_priority"]).mean()
        esc_acc = (df["escalated"] == df["should_escalate"]).mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickets tested", f"{len(df)} / {len(tk)}")
        c2.metric("Category accuracy", f"{cat_acc:.0%}")
        c3.metric("Priority accuracy", f"{pri_acc:.0%}")
        c4.metric("Escalation accuracy", f"{esc_acc:.0%}")
        st.subheader("Mistakes (where the AI disagreed with our labels)")
        wrong = df[
            (df["pred_category"] != df["true_category"])
            | (df["pred_priority"] != df["true_priority"])
        ]
        st.dataframe(
            wrong[["ticket_id", "text", "true_category", "pred_category",
                   "true_priority", "pred_priority"]],
            use_container_width=True,
        )
    else:
        st.info("Run `python eval.py` first to generate the accuracy report.")
