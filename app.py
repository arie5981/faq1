import streamlit as st
import os
import re
import unicodedata
import streamlit as st
import copy
from dataclasses import dataclass
from typing import List, Optional
from rapidfuzz import fuzz
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import openai

# ========== הגדרות ==========
st.set_page_config(page_title="עוזר אתר מייצגים", layout="wide")
st.title("🟦 עוזר אתר מייצגים – גרסת דמו אינטרנטית")

# קלט API key מצד המשתמש
# api_key = st.text_input("🔑 הכנס מפתח OpenAI:", type="password")

# טעינת המפתח מתוך Streamlit Secrets
api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key

if not api_key:
    st.info("הכנס מפתח API כדי להתחיל.")
    st.stop()

openai.api_key = api_key
os.environ["OPENAI_API_KEY"] = api_key

# ========== העלאת קובץ FAQ ==========
st.subheader("📄 העלה קובץ FAQ (טקסט בפורמט UTF-8):")
uploaded_file = st.file_uploader("בחר קובץ faq.txt", type=["txt"])

if not uploaded_file:
    st.warning("יש להעלות קובץ FAQ כדי להמשיך.")
    st.stop()

raw_faq = uploaded_file.read().decode("utf-8")

# ========== פונקציות נורמליזציה ==========
def normalize_he(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\u200e\u200f]", "", s)
    s = re.sub(r"[^\w\s\u0590-\u05FF]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

# ========== מבנה FAQ ==========
@dataclass
class FAQItem:
    question: str
    variants: List[str]
    answer: str

# ========== פירוק ה-FAQ ==========
def parse_faq(text: str) -> List[FAQItem]:
    items = []
    blocks = re.split(r"(?=שאלה\s*:)", text)
    for b in blocks:
        b = b.strip()
        if not b:
            continue

        q_match = re.search(r"שאלה\s*:\s*(.+)", b)
        a_match = re.search(r"(?s)תשובה\s*:\s*(.+?)(?:\nהוראה\s*:|\Z)", b)
        v_match = re.search(r"(?s)ניסוחים דומים\s*:\s*(.+?)(?:\nתשובה\s*:|\Z)", b)

        question = q_match.group(1).strip() if q_match else ""
        answer = a_match.group(1).strip() if a_match else ""
        variants = []

        if v_match:
            raw = v_match.group(1)
            variants = [s.strip(" -\t") for s in raw.split("\n") if s.strip()]

        items.append(FAQItem(question, variants, answer))

    return items

faq_items = parse_faq(raw_faq)
st.success(f"נטענו {len(faq_items)} שאלות מה-FAQ")

# ========== Embeddings ==========
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = []
for i, item in enumerate(faq_items):
    merged = " | ".join([item.question] + item.variants)
    docs.append(Document(page_content=merged, metadata={"idx": i}))

faq_store = FAISS.from_documents(docs, embeddings)

# ========== מנוע חיפוש ==========
def search_faq(query: str) -> Optional[str]:
    nq = normalize_he(query)

    verbs = {
        "add": ["הוסף", "להוסיף", "הוספה", "מוסיף", "מוסיפים", "לצרף", "צירוף", "פתיחה", "פתיחת", "רישום", "להירשם"],
        "delete": ["מחק", "מחיקה", "להסיר", "הסר", "הסרה", "ביטול", "לבטל", "סגור", "לסגור"],
        "update": ["עדכן", "לעדכן", "עדכון", "שינוי", "לשנות", "עריכה", "לתקן"]
    }

    intent = None
    for k, words in verbs.items():
        if any(w in nq for w in words):
            intent = k
            break

    scored = []
    for i, item in enumerate(faq_items):
        all_texts = [item.question] + item.variants

        for t in all_texts:
            score = fuzz.token_sort_ratio(nq, normalize_he(t))

            t_intent = None
            for k, words in verbs.items():
                if any(w in t for w in words):
                    t_intent = k
                    break

            if intent and t_intent and intent != t_intent:
                score -= 50
            if intent and t_intent and intent == t_intent:
                score += 25

            scored.append((score, i, t))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:5]

    best_score = top[0][0]
    best_index = top[0][1]

    if best_score >= 55:
        return faq_items[best_index]

    hits = faq_store.similarity_search_with_score(nq, k=4)
    hits = sorted(hits, key=lambda x: x[1])

    if hits and hits[0][1] < 1.2:
        idx = hits[0][0].metadata["idx"]
        return faq_items[idx]

    return None

# ========== ממשק משתמש ==========
st.subheader("❓ שאל שאלה")

query = st.text_input("הקלד שאלה כאן:")
submit = st.button("שלח")

if submit and query:
    result = search_faq(query)

    if not result:
        st.error("לא נמצאה תשובה. נסה לנסח אחרת.")
    else:
        st.success("✓ נמצאה תשובה")
        st.write(result.answer)
        st.caption(f"🔹 שאלה מזוהה: {result.question}")




