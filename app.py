import os
import re
import openai
import streamlit as st
# ============================
# שלב 1: הגדרת API KEY
# ============================
# תיקון: הייבוא של OpenAIEmbeddings עבר לחבילה langchain_openai.
from langchain_openai import OpenAIEmbeddings
# תיקון: הייבוא של FAISS עבר לחבילה langchain_community.
from langchain_community.vectorstores import FAISS
# תיקון: הייבוא של Document עבר לחבילה langchain_core.
from langchain_core.documents import Document

from rapidfuzz import fuzz
import requests
import unicodedata

openai.api_key = os.getenv('OPENAI_API_KEY')

# ============================
# שלב 2: הגדרת דף האינטרנט
# ============================
st.set_page_config(
    page_title="תמיכה לאתר מייצגים בגבייה",
    page_icon="💬",
    layout="wide",
)

# ============================
# שלב 3: הגדרות CSS
# ============================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
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
        color: black !important;    /* תיקון צבע */
        text-align: right;
    }

    .faq-box li {
        margin-bottom: 6px;
        color: black !important;
    }

    /* בועות צ'אט */
    .chat-bubble-question {
        background-color: #e5e7eb;      /* אפור בהיר */
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

        background-color: white !important;    /* לבן ✔ */
        color: black !important;               /* טקסט שחור ✔ */
    }

    .stTextInput input::placeholder {
        color: #888 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# שלב 4: טעינת FAQ אוטומטית מ-GitHub
# ============================
FAQ_URL = "https://raw.githubusercontent.com/arie5981/faq1/main/faq.txt"
faq_text = requests.get(FAQ_URL).text

# ============================
# שלב 5: נורמליזציה של טקסט
# ============================
def normalize_he(s: str) -> str:
    """מנקה ומנרמל טקסט לעברית"""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\u200e\u200f]", "", s)
    s = re.sub(r"[^\w\s\u0590-\u05FF]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

# ============================
# שלב 6: יצירת אינדקס Embedding
# ============================
def create_faq_index(faq_text):
    faq_items = []
    blocks = re.split(r"(?=שאלה\s*:)", faq_text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        q_match = re.search(r"שאלה\s*:\s*(.+)", block)
        a_match = re.search(r"(?s)תשובה\s*:\s*(.+?)(?:\nהוראה\s*:|\Z)", block)
        v_match = re.search(r"(?s)ניסוחים דומים\s*:\s*(.+?)(?:\nתשובה\s*:|\Z)", block)

        question = q_match.group(1).strip() if q_match else ""
        answer = a_match.group(1).strip() if a_match else ""
        variants = [s.strip(" -\t") for s in v_match.group(1).split("\n") if s.strip()] if v_match else []

        faq_items.append({"question": question, "answer": answer, "variants": variants})
    return faq_items

faq_items = create_faq_index(faq_text)

# ============================
# שלב 7: חיפוש פאזי + סמנטי
# ============================
def search_faq(query: str):
    query = normalize_he(query)
    scored = []
    
    for item in faq_items:
        all_texts = [item['question']] + item['variants']
        for text in all_texts:
            score = fuzz.token_sort_ratio(query, normalize_he(text))
            scored.append((score, item))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:5]
    
    if top[0][0] >= 55:
        return top[0][1]['answer']
    else:
        return "לא נמצאה תשובה. נסה לנסח אחרת."

# ============================
# שלב 8: ממשק משתמש
# ============================
st.title("תמיכה לאתר מייצגים בגבייה")

# כותרת עליונה
st.markdown(
    """
    <div class="header-container">
      <div>
        <div class="header-text-main">הביטוח הלאומי</div>
        <div class="header-text-sub">תמיכה לאתר מייצגים בגבייה</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# שאלות נפוצות
st.subheader("שאלות נפוצות:")
st.write("1. איך מוסיפים משתמש חדש באתר מייצגים.")
st.write("2. מקבל הודעה שאחד או יותר מנתוני ההזדהות שגויים.")
st.write("3. איך יוצרים קיצור דרך לאתר מייצגים על שולחן העבודה.")
st.write("4. רוצה לקבל את הקוד החד פעמי לדואר אלקטרוני.")

# תיבת שאלה
question = st.text_input("שאל שאלה והקש Enter")

if question:
    answer = search_faq(question)
    st.write(f"**תשובה:** {answer}")
