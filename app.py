import streamlit as st
import os
import re
import unicodedata
from rapidfuzz import fuzz
from dataclasses import dataclass
from typing import List
import requests

# =========================================================
#   הגדרות עיצוב – RTL + עיצוב כמו ChatGPT
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"]  {
    direction: rtl;
    text-align: right;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* כותרת עליונה */
.header-bar {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 12px;
    padding: 0.8rem 1.2rem;
    border-bottom: 1px solid #333;
}

.header-logo {
    height: 50px;
}

.header-text-main {
    font-weight: 700;
    font-size: 1.2rem;
    color: #3b82f6;
}

.header-text-sub {
    font-weight: 500;
    font-size: 0.95rem;
    color: #38bdf8;
}

/* בועת שאלה */
.user-bubble {
    background-color: #e5e7eb;
    color: #111;
    padding: 0.8rem 1rem;
    border-radius: 18px;
    display: inline-block;
    margin: 0.2rem 0 0.4rem 0;
}

/* תשובה */
.assistant-text {
    margin: 0.2rem 0 1rem 0;
    color: white;
}

/* שורת שאלה בתחתית */
.question-box {
    position: fixed;
    bottom: 0;
    right: 0;
    left: 0;
    padding: 1rem;
    background: #111;
    border-top: 1px solid #333;
}

/* עיצוב תיבת קלט */
input[type="text"] {
    border-radius: 16px !important;
    padding: 0.6rem 1rem !important;
    font-size: 1rem !important;
}

/* הסתרת כפתור "שלח" */
div.stButton > button {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
#   כותרת עם לוגו
# =========================================================

logo_url = "https://raw.githubusercontent.com/arie5981/faq1/main/logobtl.png"

st.markdown(f"""
<div class="header-bar">
    <div>
        <div class="header-text-main">הביטוח הלאומי</div>
        <div class="header-text-sub">תמיכה לאתר מייצגים בגבייה</div>
    </div>
    <img class="header-logo" src="{logo_url}">
</div>
""", unsafe_allow_html=True)

# =========================================================
#   טעינת FAQ מה-GitHub
# =========================================================

FAQ_URL = "https://raw.githubusercontent.com/arie5981/faq1/main/faq.txt"

@st.cache_data
def load_faq():
    text = requests.get(FAQ_URL).text
    return text

faq_raw = load_faq()

# =========================================================
#   מודל נתונים לשאלה
# =========================================================

@dataclass
class FAQItem:
    question: str
    variants: List[str]
    answer: str

# =========================================================
#   פירוק קובץ ה-FAQ
# =========================================================

def parse_faq(text: str) -> List[FAQItem]:
    items = []
    blocks = re.split(r"(?=שאלה\s*:)", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        q = re.search(r"שאלה\s*:\s*(.+)", block)
        a = re.search(r"(?s)תשובה\s*:\s*(.+)", block)
        v = re.search(r"(?s)ניסוחים דומים\s*:\s*(.+?)(?:\nתשובה|\Z)", block)

        question = q.group(1).strip() if q else ""
        answer = a.group(1).strip() if a else ""
        variants = []

        if v:
            variants = [
                x.strip(" -\t")
                for x in v.group(1).split("\n")
                if x.strip()
            ]

        items.append(FAQItem(question, variants, answer))
    return items

faq_items = parse_faq(faq_raw)

# =========================================================
#   פונקציה לנורמליזציה
# =========================================================

def normalize_he(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[^\w\s\u0590-\u05FF]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

# =========================================================
#   מנוע חיפוש פשוט
# =========================================================

def search_faq(query: str):
    nq = normalize_he(query)
    best_score = -1
    best_item = None

    for item in faq_items:
        texts = [item.question] + item.variants
        for t in texts:
            score = fuzz.token_sort_ratio(nq, normalize_he(t))
            if score > best_score:
                best_score = score
                best_item = item

    if best_score < 55:
        return None

    return best_item

# =========================================================
#   שמירת היסטוריה
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
#   הצגת היסטוריה
# =========================================================

for q, a in st.session_state.history:
    st.markdown(f'<div class="user-bubble">{q}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="assistant-text">{a}</div>', unsafe_allow_html=True)

# =========================================================
#   תיבת השאלה בתחתית המסך
# =========================================================

st.markdown('<div class="question-box">', unsafe_allow_html=True)

query = st.text_input("🔍 שאל שאלה והקש Enter")

if query:
    result = search_faq(query)
    if result:
        st.session_state.history.append(
            (query, result.answer)
        )
        st.experimental_rerun()
    else:
        st.session_state.history.append((query, "לא נמצאה תשובה"))
        st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)
