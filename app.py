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

# ====== הגדרת API KEY ======
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

st.subheader("🔑 הגדרת מפתח OpenAI")

api_key_input = st.text_input("הכנס מפתח OpenAI:", type="password")

if api_key_input:
    st.session_state.api_key = api_key_input

if not st.session_state.api_key:
    st.warning("יש להזין מפתח API כדי להמשיך")
    st.stop()

openai.api_key = st.session_state.api_key
os.environ["OPENAI_API_KEY"] = api_key

# ========== טעינת קובץ FAQ מתוך הריפו ==========
FAQ_PATH = "faq.txt"

try:
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        raw_faq = f.read()
except FileNotFoundError:
    st.error("❌ הקובץ faq.txt לא נמצא בריפו. ודא שהוא נמצא באותה תיקייה כמו app.py.")
    st.stop()

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

# ========== RTL + עיצוב כללי ==========
st.markdown("""
<style>
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
    font-family: "Alef", "Segoe UI", sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 צ'אט התמיכה למייצגים")

# ========== שמירת היסטוריה ==========
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ========== תצוגת כל ההתכתבות ==========
st.markdown("### 💬 היסטוריית השיחה")

for role, msg in st.session_state["messages"]:
    if role == "user":
        st.markdown(f"""
        <div style='background:#e8f0fe;padding:12px;border-radius:10px;margin:6px 0;text-align:right'>
            <b>🧑 שאלה:</b><br>{msg}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#f1f1f1;padding:12px;border-radius:10px;margin:6px 0;text-align:right'>
            <b>🤖 תשובה:</b><br>{msg}
        </div>
        """, unsafe_allow_html=True)

# ========== שדה שאלה ==========
query = st.text_input("✏️ הקלד שאלה חדשה כאן:")

if st.button("📨 שלח"):
    if query.strip():
        # תשובה מהמערכת
        result = search_faq(query)
        if not result:
            answer_text = "לא נמצאה תשובה. נסה לנסח אחרת."
        else:
            answer_text = f"{result.answer}\n\n🔹 שאלה מזוהה: {result.question}"

        # שמירה להיסטוריה
        st.session_state["messages"].append(("user", query))
        st.session_state["messages"].append(("assistant", answer_text))

        # רענון מיידי של הדף כדי להציג את ההודעה
        st.session_state.clear()











