import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def call_model(prompt, retries=5):
    
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            msg = str(e).lower()
            temporary = (
                "503" in msg
                or "disconnected" in msg
                or "timed out" in msg
                or "timeout" in msg
                or "connection" in msg
            )
            if temporary and attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            raise


INSTRUCTIONS = """You are an IT support assistant. Read the ticket and the knowledge base articles below.
Reply with JSON only, in this exact format:
{"category": "...", "priority": "...", "reason": "...", "kb_id": "...", "confidence": 0, "reply": "..."}

Allowed categories: Network/VPN, Account/Login, Email, Hardware, Software, Security, Access Request

Priority rules (decide using these steps):
- High: the user is completely blocked from core work with no workaround, OR many people are affected, OR a security incident has already happened (user clicked a link, entered a password, or malware was detected), OR a hard deadline today or tomorrow is at risk.
- Medium: one user is affected and work is slowed or partly blocked, OR a workaround exists, OR a suspicious message was reported but nothing was clicked, OR a request is needed for ongoing work.
- Low: no real impact on current work, such as minor or cosmetic issues, routine requests without a deadline, and general questions.

Other rules:
- reason: one short sentence explaining the category and priority
- Use ONLY the knowledge base articles below to answer. Do not invent steps.
- kb_id: the id of the article you used, or "NONE" if no article fits the ticket
- confidence: a number from 0 to 100 showing how sure you are that the article solves this ticket
- reply: a polite, short reply to the employee with the fix steps. If no article fits, say the ticket is being passed to an engineer.

"""


def analyze_ticket(ticket, articles):
    kb_text = "\n\n".join(
        f"[{r.id}] {r.title}: {r.solution}" for r in articles.itertuples()
    )
    prompt = (
        INSTRUCTIONS
        + "Knowledge base articles:\n"
        + kb_text
        + "\n\nTicket:\n"
        + ticket
    )
    return call_model(prompt)
