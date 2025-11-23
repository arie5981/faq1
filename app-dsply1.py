# ============================================
#   עוזר אתר מייצגים – גרסה ל-Streamlit
#   קורא faq.txt מהריפו, מציג צ'אט בסגנון ChatGPT
# ============================================

import streamlit as st
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import List
from rapidfuzz import fuzz

import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ============================================
#   הגדרת מפתח OpenAI מ־Streamlit Secrets
# ============================================
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("❌ חסר מפתח OPENAI_API_KEY ב־Streamlit Secrets.\nיש להיכנס ל־Manage app → Settings → Secrets ולהוסיף:\nOPENAI_API_KEY = \"...\"")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# ============================================
#   הגדרות עמוד ו־CSS ל־RTL + עיצוב עדין
# ============================================
st.set_page_config(page_title="תמיכה לאתר מייצגים", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"]  {
    direction: rtl;
    text-align: right;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* שורת כותרת עליונה */
.header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.8rem 1.2rem;
    border-bottom: 1px solid #333;
}

/* לוגו */
.header-logo {
    height: 48px;
}

/* טקסט ליד הלוגו */
.header-text-main {
    font-weight: 700;
    font-size: 1.2rem;
    color: #3b82f6; /* כחול */
}
.header-text-sub {
    font-weight: 500;
    font-size: 0.95rem;
    color: #38bdf8; /* תכלת */
}

/* בועת שאלה */
.user-bubble {
    background-color: #e5e5e5;
    padding: 0.8rem 1rem;
    border-radius: 18px;
    margin: 0.2rem 0 0.4rem 0;
    display: inline-block;
}

/* טקסט תשובה */
.assistant-text {
    margin: 0.2rem 0 0.8rem 0;
}

/* תיבת הקלט בתחתית */
.question-box {
    position: relative;
    margin-top: 1rem;
    padding-top: 0.5rem;
    border-top: 1px solid #333;
}

/* הסתרת כפתור "שלח" של הטופס – שולחים עם Enter */
div.stButton > button {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ============================================
#   כותרת עליונה עם לוגו וטקסט
# ============================================
# אם יש לך קובץ לוגו בתיקייה, אפשר להשתמש: "logo.png"
# כרגע נשים URL כללי, אפשר להחליף אחר כך.
logo_url = "https://raw.githubusercontent.com/arie5981/faq1/main/logobtl.png"

st.markdown(
    f"""
<div class="header-bar">
  <div style="display:flex; align-items:center; gap:0.6rem;">
    <img src="{logo_url}" class="header-logo" alt="לוגו הביטוח הלאומי" />
    <div style="display:flex; flex-direction:column;">
      <span class="header-text-main">הביטוח הלאומי</span>
      <span class="header-text-sub">תמיכה לאתר מייצגים בגבייה</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================
#   קריאת קובץ faq.txt מתוך הריפו
# ============================================
FAQ_PATH = "faq.txt"

def read_txt_utf8(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

raw_faq = read_txt_utf8(FAQ_PATH)

# ============================================
#   עיבוד ה-FAQ (כמו בקוד שלך)
# ============================================
def normalize_he(s: str) -> str:
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

# === יצירת Embeddings + FAISS ===
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

docs = []
for i, item in enumerate(faq_items):
    merged = " | ".join([item.question] + item.variants)
    docs.append(Document(page_content=merged, metadata={"idx": i}))

faq_store = FAISS.from_documents(docs, embeddings)

# ============================================
#   חיפוש FAQ – fuzzy + embeddings
# ============================================
def search_faq(query: str) -> str:
    nq = normalize_he(query)

    # --- חיפוש פאזי על שאלות וניסוחים ---
    scored = []
    for i, item in enumerate(faq_items):
        all_texts = [item.question] + item.variants
        for t in all_texts:
            score = fuzz.token_sort_ratio(nq, normalize_he(t))
            scored.append((score, i, t))

    scored.sort(reverse=True, key=lambda x: x[0])
    best_score, best_idx, _ = scored[0]

    if best_score >= 60:
        item = faq_items[best_idx]
        return f"{item.answer}\n\nמקור: faq\nשאלה מזוהה: {item.question}"

    # --- fallback: embeddings ---
    hits = faq_store.similarity_search_with_score(query, k=3)
    best_doc, best_dist = hits[0]

    if best_dist < 1.1:
        idx = best_doc.metadata["idx"]
        item = faq_items[idx]
        return f"{item.answer}\n\nמקור: faq\nשאלה מזוהה (סמנטי): {item.question}"

    return "לא נמצאה תשובה, נסה לנסח את השאלה מחדש."

# ============================================
#   ניהול שיחה כמו ChatGPT
# ============================================
if "messages" not in st.session_state:
    # כל הודעה היא מילון: {"role": "user"/"assistant", "content": "..."}
    st.session_state.messages = []

# שאלות נפוצות למסך הראשון
POPULAR_QUESTIONS = [
    "איך מוסיפים משתמש חדש באתר מייצגים.",
    "מקבל הודעה שאחד או יותר מנתוני ההזדהות שגויים.",
    "איך יוצרים קיצור דרך לאתר מייצגים על שולחן העבודה.",
    "רוצה לקבל את הקוד החד פעמי לדואר אלקטרוני.",
]

st.markdown("")

# אם עדיין אין שיחה – מסך פתיחה עם שאלות נפוצות
if len(st.session_state.messages) == 0:
    st.markdown("### שאלות נפוצות:")
    for i, q in enumerate(POPULAR_QUESTIONS, start=1):
        st.markdown(f"{i}. {q}")

    st.markdown("## איך אפשר לעזור?")
    st.markdown("")

# הצגת היסטוריית שיחה (שאלה = בועה אפורה, תשובה = טקסט לבן)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
<div class="user-bubble">
<strong>שאלה:</strong> {msg['content']}
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="assistant-text">
<strong>תשובה:</strong> {msg['content']}
</div>
""", unsafe_allow_html=True)

# תיבת השאלה בתחתית (Enter שולח, בלי כפתור)
st.markdown('<div class="question-box"></div>', unsafe_allow_html=True)

with st.form("ask_form", clear_on_submit=True):
    query = st.text_input(" ", placeholder="שאל שאלה והקש Enter")
    submitted = st.form_submit_button("שלח")  # מוסתר ב-CSS

if submitted and query:
    # הוספת השאלה להיסטוריה
    st.session_state.messages.append({"role": "user", "content": query})
    # הפעלת מנוע ה־FAQ
    answer = search_faq(query)
    # הוספת תשובה להיסטוריה
    st.session_state.messages.append({"role": "assistant", "content": answer})

