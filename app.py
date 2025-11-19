import streamlit as st
import os
import openai
import unicodedata
import re
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rapidfuzz import fuzz
from dataclasses import dataclass
from typing import List, Optional

# הגדרת API KEY (כדי לשלוף את המפתח OpenAI)
openai.api_key = os.getenv("OPENAI_API_KEY")

# הגדרת עיצוב האתר
st.set_page_config(
    page_title="תמיכה לאתר מייצגים בגבייה",
    page_icon="💬",
    layout="wide",
)

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        direction: rtl;
        text-align: right;
        font-family: "Alef", "Heebo", "Arial", sans-serif;
        background-color: #0e1117;
        color: #ffffff;
    }

    /* כותרת עליונה – יישור מלא לימין */
    .header-container {
        display: flex;
        flex-direction: row-reverse;
        align-items: center;
        justify-content: flex-end;
        gap: 14px;
        margin-bottom: 20px;
    }

    .header-text-main {
        font-size: 26px;
        font-weight: 700;
        color: #1f9cf0;
        line-height: 1.1;
    }

    .header-text-sub {
        font-size: 16px;
        font-weight: 500;
        color: #4fd1ff;
        line-height: 1.1;
    }

    /* שאלות נפוצות */
    .faq-box {
        background-color: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 16px 18px;
        font-size: 16px;
        margin-bottom: 20px;
        color: white;
    }

    .faq-box li {
        margin-bottom: 6px;
        color: white;
    }

    /* בועות צ'אט */
    .chat-bubble-question {
        background-color: #e5e7eb; /* אפור בהיר */
        color: #111111;
        border-radius: 16px;
        padding: 10px 14px;
        margin-bottom: 6px;
        max-width: 80%;
        margin-left: auto;
    }

    .chat-bubble-answer {
        background-color: transparent;
        border-radius: 16px;
        padding: 10px 14px;
        margin-bottom: 18px;
        max-width: 95%;
        margin-right: auto;
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
    }

    /* תיבת השאלה */
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
        border-radius: 999px;
        border: 1px solid #d1d5db;
        padding-right: 14px;
        padding-left: 40px;
        background-color: white;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# פונקציות נורמליזציה ועיבוד טקסט
# ===========================
def normalize_he(s: str) -> str:
    """מנקה, מאחד, ומרחיב ניסוחים חלקיים לשפה טבעית."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\u200e\u200f]", "", s)
    s = re.sub(r"[^\w\s\u0590-\u05FF]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

@dataclass
class FAQItem:
    question: str
    variants: List[str]
    answer: str

# ============================
# שלב 2: קריאת קובץ ה-FAQ מ-GitHub
# ============================
faq_url = "https://raw.githubusercontent.com/arie5981/faq1/main/faq.txt"

def read_faq_from_url(url: str) -> str:
    import requests
    response = requests.get(url)
    return response.text

raw_faq = read_faq_from_url(faq_url)

# ============================
# שלב 3: Parsing לקובץ ה-FAQ
# ============================
def parse_faq_new(text: str) -> List[FAQItem]:
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

faq_items = parse_faq_new(raw_faq)

# ============================
# שלב 4: יצירת אינדקס Embeddings על כל השאלות והניסוחים
# ============================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = []
for i, item in enumerate(faq_items):
    all_variants = [item.question] + item.variants
    merged_text = " | ".join(all_variants)
    docs.append(Document(page_content=merged_text, metadata={"idx": i}))

faq_store = FAISS.from_documents(docs, embeddings)

# ============================
# שלב 5: חיפוש פאזי + סמנטי + ניתוח כוונה
# ============================
def search_faq(query: str) -> Optional[str]:
    nq = normalize_he(query)

    # זיהוי כוונה (intent)
    verbs = {
        "add": ["הוסף", "להוסיף", "הוספה", "מוסיף", "מוסיפים", "לצרף", "צירוף", "פתיחה", "פתיחת", "רישום", "להירשם"],
        "delete": ["מחק", "מחיקה", "להסיר", "הסר", "הסרה", "ביטול", "לבטל", "סגור", "לסגור", "ביטול משתמש"],
        "update": ["עדכן", "לעדכן", "עדכון", "שינוי", "לשנות", "עריכה", "ערוך", "לתקן", "תיקון"]
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

            scored.append((score, i, t.strip(), t_intent))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:5]

    best = top[0]
    best_fuzzy_score = best[0]
    result_item = None

    # ============================
    # Embeddings
    # ============================
    hits = faq_store.similarity_search_with_score(nq, k=8)
    boosted_hits = []

    for doc, score in hits:
        idx = doc.metadata["idx"]
        question_text = faq_items[idx].question
        text_norm = normalize_he(question_text + " " + " ".join(faq_items[idx].variants))

        boosted_hits.append((doc, score))

    boosted_hits.sort(key=lambda x: x[1])

    # נבדוק את הציון הטוב ביותר
    best_embed_score = boosted_hits[0][1] if boosted_hits else 999

    if best_fuzzy_score < 55 and best_embed_score > 1.2:
        return "לא נמצאה תשובה, נסה לנסח את השאלה מחדש"

    if best_fuzzy_score >= 55:
        result_item = faq_items[best[1]]
    elif boosted_hits:
        result_item = faq_items[boosted_hits[0][0].metadata["idx"]]

    if result_item:
        answer_text = result_item.answer.strip()
        return answer_text

    return "לא נמצאה תשובה, נסה לנסח את השאלה מחדש"


# ============================
# שלב 6: ממשק שורת פקודה עם שאלות נפוצות
# ============================
query = st.text_input("שאל שאלה והקש Enter", "")

if query:
    st.write("תשובה: התשובה המתקבלת לשאלה שלך") # תשובה ממאגר FAQ או חיפוש אחר



