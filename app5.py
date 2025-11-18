# app.py
import os
import re
import unicodedata
import copy
from dataclasses import dataclass
from typing import List, Optional

import requests
import streamlit as st
from rapidfuzz import fuzz

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# =========================
# הגדרות כלליות ו־CSS
# =========================
st.set_page_config(
    page_title="תמיכה לאתר מייצגים בגבייה",
    page_icon="💬",
    layout="wide",
)

# CSS לעיצוב ול־RTL
st.markdown(
    """
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
        text-align: right;
    }

    .header-text-sub {
        font-size: 16px;
        font-weight: 500;
        color: #4fd1ff;
        line-height: 1.1;
        text-align: right;
    }

    /* שאלות נפוצות */
    .faq-box {
        background-color: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 16px 18px;
        font-size: 16px;
        margin-bottom: 20px;
        color: black !important;   /* תיקן לכח */
        text-align: right;
    }

    .faq-box li {
        margin-bottom: 6px;
        color: black !important;
    }

    /* בועות צ'אט */
    .chat-bubble-question {
        background-color: #e5e7eb;     /* אפור בהיר */
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

        background-color: white !important;   /* לבן ✔ */
        color: black !important;               /* טקסט שחור ✔ */
    }

    .stTextInput input::placeholder {
        color: #888 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# לוגו + כותרת עליונה
# =========================
with st.container():
    col_logo, col_title = st.columns([1, 5])

    with col_logo:
        st.image(
            "https://raw.githubusercontent.com/arie5981/faq1/main/logobtl.png",
            width=70,
        )

    with col_title:
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


# =========================
# בדיקת מפתח OpenAI מתוך secrets
# =========================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ חסר מפתח OPENAI_API_KEY ב־Streamlit secrets.")
    st.stop()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# =========================
# טעינת faq.txt מ-GitHub
# =========================
FAQ_URL = "https://raw.githubusercontent.com/arie5981/faq1/main/faq.txt"


@st.cache_data(show_spinner="טוען את קובץ ה־FAQ מ-GitHub...")
def load_faq_from_github(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


raw_faq = load_faq_from_github(FAQ_URL)

# =========================
# נורמליזציה לעברית
# =========================
def normalize_he(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\u200e\u200f]", "", s)
    s = re.sub(r"[^\w\s\u0590-\u05FF]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()

    # הוספת 'איך' לניסוחים חלקיים
    patterns = [
        (r"^רוצה ", "איך "),
        (r"^אני רוצה ", "איך "),
        (r"^אפשר ", "איך "),
        (r"^בא לי ", "איך "),
        (r"^מבקש ", "איך "),
    ]
    for p, repl in patterns:
        if re.match(p, s):
            s = re.sub(p, repl, s)
            break

    # התאמות ספציפיות למייצגים
    s = s.replace("עמדה", "עמדה למחשב")
    s = s.replace("להוסיף עמדה", "להוסיף משתמש חדש")
    return s


# =========================
# מבנה הנתונים של שאלה/תשובה
# =========================
@dataclass
class FAQItem:
    question: str
    variants: List[str]
    answer: str


# =========================
# פירוק קובץ ה-FAQ
# =========================
def parse_faq_new(text: str) -> List[FAQItem]:
    """
    מצפה למבנה:
    שאלה: ...
    ניסוחים דומים:
    - ...
    - ...
    תשובה: ...
    הוראה: ...   (לא חובה ולא בשימוש כרגע)
    """
    items: List[FAQItem] = []
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
        variants: List[str] = []

        if v_match:
            raw = v_match.group(1)
            variants = [s.strip(" -\t") for s in raw.split("\n") if s.strip()]

        if question or answer:
            items.append(FAQItem(question, variants, answer))

    return items


# =========================
# יצירת אינדקס Embeddings
# =========================
@st.cache_resource(show_spinner="יוצר אינדקס Embeddings...")
def build_faq_index(faq_text: str, api_key: str):
    items = parse_faq_new(faq_text)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    docs: List[Document] = []
    for i, item in enumerate(items):
        all_variants = [item.question] + item.variants
        merged_text = " | ".join(all_variants)
        docs.append(Document(page_content=merged_text, metadata={"idx": i}))

    store = FAISS.from_documents(docs, embeddings)
    return items, store


faq_items, faq_store = build_faq_index(raw_faq, OPENAI_API_KEY)


# =========================
# חיפוש פאזי + Embeddings
# =========================
def search_faq(query: str) -> Optional[str]:
    """
    מחזיר טקסט מוכן להצגה (תשובה + מקור + שאלה מזוהה),
    או הודעת שגיאה ידידותית.
    """
    nq = normalize_he(query)

    # זיהוי כוונה (add/delete/update)
    verbs = {
        "add": ["הוסף", "להוסיף", "הוספה", "מוסיף", "מוסיפים", "לצרף", "צירוף", "פתיחה", "פתיחת", "רישום", "להירשם"],
        "delete": ["מחק", "מחיקה", "להסיר", "הסר", "הסרה", "ביטול", "לבטל", "סגור", "לסגור", "ביטול משתמש"],
        "update": ["עדכן", "לעדכן", "עדכון", "שינוי", "לשנות", "עריכה", "ערוך", "לתקן", "תיקון"],
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
    best_fuzzy_score, best_fuzzy_idx, _, _ = top[0]

    # Embeddings
    hits = faq_store.similarity_search_with_score(nq, k=8)

    # ניקוד מבוסס מילות מפתח בתחום מייצגים
    key_words = ["יפוי", "כוח", "הרשאה", "ייצוג", "מייצג", "מעסיק", "מבוטח"]
    boosted_hits = []
    for doc, score in hits:
        idx = doc.metadata["idx"]
        text_norm = normalize_he(
            faq_items[idx].question + " " + " ".join(faq_items[idx].variants)
        )
        for kw in key_words:
            if kw in nq and kw in text_norm:
                score -= 0.15
        boosted_hits.append((doc, score))

    boosted_hits.sort(key=lambda x: x[1])
    best_embed_score = boosted_hits[0][1] if boosted_hits else 999.0

    # תנאי סף – אם גם פאזי חלש וגם Embeddings רחוקים
    if best_fuzzy_score < 55 and best_embed_score > 1.2:
        return "לא נמצאה תשובה, נסה לנסח את השאלה מחדש."

    # בחירת תשובה סופית
    if best_fuzzy_score >= 55:
        result_item = copy.deepcopy(faq_items[best_fuzzy_idx])
    else:
        best_idx = boosted_hits[0][0].metadata["idx"]
        result_item = copy.deepcopy(faq_items[best_idx])

    # שאלות קשורות (מבוסס Embeddings)
    similar_questions: List[str] = []
    for doc, score in boosted_hits[1:4]:
        idx = doc.metadata["idx"]
        q_txt = faq_items[idx].question.strip()
        if q_txt == result_item.question.strip():
            continue
        if score <= 1.3:
            similar_questions.append(q_txt)

    answer_text = result_item.answer.strip()
    if similar_questions:
        answer_text += "\n\nשאלות קשורות:\n" + "\n".join(similar_questions)

    answer_text += f"\n\nמקור: faq\nשאלה מזוהה: {result_item.question}"
    return answer_text


# =========================
# ניהול היסטוריית צ'אט
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # כל איבר: {"question": str, "answer": str}


# =========================
# גוף הדף – כמו ChatGPT
# =========================
st.markdown(
    """
    <div class="chat-container">
    """,
    unsafe_allow_html=True,
)

# דף ראשון – שאלות נפוצות
if not st.session_state.chat_history:
    st.markdown(
        """
        <div class="faq-box">
          <strong>שאלות נפוצות:</strong>
          <ol>
            <li>איך מוסיפים משתמש חדש באתר מייצגים.</li>
            <li>מקבל הודעה שאחד או יותר מנתוני ההזדהות שגויים.</li>
            <li>איך יוצרים קיצור דרך לאתר מייצגים על שולחן העבודה.</li>
            <li>רוצה לקבל את הקוד החד פעמי לדואר אלקטרוני.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

# הצגת היסטוריית השיחה
for turn in st.session_state.chat_history:
    st.markdown(
        f"""
        <div class="chat-bubble-question">
          <strong>שאלה:</strong> {turn["question"]}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="chat-bubble-answer">
          <strong>תשובה:</strong><br>
          {turn["answer"].replace("\n", "<br>")}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)  # סוף chat-container

# =========================
# טופס השאלה בתחתית
# =========================
with st.form(key="question_form", clear_on_submit=True):
    query = st.text_input("איך אפשר לעזור?", placeholder="שאל שאלה והקש Enter")
    submitted = st.form_submit_button("שלח")

    if submitted and query.strip():
        answer = search_faq(query.strip())
        st.session_state.chat_history.append(
            {"question": query.strip(), "answer": answer}
        )
        # אין צורך ב־experimental_rerun – Streamlit מרנדר מחדש לבד



