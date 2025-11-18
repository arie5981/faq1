import streamlit as st
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional
from rapidfuzz import fuzz
import requests

# =========================================================
#   עיצוב בסיסי – RTL + בועות בסגנון ChatGPT
# =========================================================
st.set_page_config(page_title="תמיכה לאתר מייצגים", layout="wide")

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
    height: 48px;
}

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
    background-color: #e5e7eb;  /* אפור בהיר */
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

/* קו מפריד ותיבת שאלה בתחתית האזור */
.question-box {
    margin-top: 1.5rem;
    padding-top: 0.8rem;
    border-top: 1px solid #333;
}

/* הסתרת כפתור "שלח" של הטופס – שולחים עם Enter */
div.stButton > button {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
#   כותרת עם לוגו
# =========================================================

# הלוגו צריך להיות קובץ logobtl.png בריפו:
# https://raw.githubusercontent.com/arie5981/faq1/main/logobtl.png
logo_url = "https://raw.githubusercontent.com/arie5981/faq1/main/logobtl.png"

st.markdown(f"""
<div class="header-bar">
    <div>
        <div class="header-text-main">הביטוח הלאומי</div>
        <div class="header-text-sub">תמיכה לאתר מייצגים בגבייה</div>
    </div>
    <img class="header-logo" src="{logo_url}" alt="לוגו הביטוח הלאומי" />
</div>
""", unsafe_allow_html=True)

# =========================================================
#   טעינת FAQ מה-GitHub
# =========================================================

FAQ_URL = "https://raw.githubusercontent.com/arie5981/faq1/main/faq.txt"

@st.cache_data
def load_faq_text(url: str) -> str:
    resp = requests.get(url)
    resp.encoding = "utf-8"
    return resp.text

raw_faq = load_faq_text(FAQ_URL)

# =========================================================
#   מודל הנתונים + Parser ל-FAQ
# =========================================================

@dataclass
class FAQItem:
    question: str
    variants: List[str]
    answer: str

def normalize_he(s: str) -> str:
    """ניקוי טקסט לעברית להשוואה פאזית"""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\u200e\u200f]", "", s)
    s = re.sub(r"[^\w\s\u0590-\u05FF]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def parse_faq(text: str) -> List[FAQItem]:
    """מפרק את הקובץ לפי שאלה / ניסוחים דומים / תשובה (מתעלם מהוראה)"""
    items: List[FAQItem] = []
    blocks = re.split(r"(?=שאלה\s*:)", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        q_match = re.search(r"שאלה\s*:\s*(.+)", block)
        v_match = re.search(r"(?s)ניסוחים דומים\s*:\s*(.+?)(?:\nתשובה\s*:|\Z)", block)
        a_match = re.search(r"(?s)תשובה\s*:\s*(.+?)(?:\nהוראה\s*:|\Z)", block)

        question = q_match.group(1).strip() if q_match else ""
        variants: List[str] = []
        if v_match:
            raw = v_match.group(1)
            variants = [s.strip(" -\t") for s in raw.split("\n") if s.strip()]

        answer = a_match.group(1).strip() if a_match else ""

        if question or answer:
            items.append(FAQItem(question=question, variants=variants, answer=answer))

    return items

faq_items = parse_faq(raw_faq)

# =========================================================
#   מנוע חיפוש פאזי (ללא OpenAI)
# =========================================================

def search_faq(query: str) -> Optional[FAQItem]:
    nq = normalize_he(query)
    best_score = -1
    best_item: Optional[FAQItem] = None

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
#   שמירת היסטוריה ב-session_state
# =========================================================

if "history" not in st.session_state:
    # רשימה של זוגות: (שאלה, תשובה)
    st.session_state.history = []

# =========================================================
#   דף ראשון – שאלות נפוצות + "איך אפשר לעזור?"
# =========================================================

POPULAR_QUESTIONS = [
    "איך מוסיפים משתמש חדש באתר מייצגים.",
    "מקבל הודעה שאחד או יותר מנתוני ההזדהות שגויים.",
    "איך יוצרים קיצור דרך לאתר מייצגים על שולחן העבודה.",
    "רוצה לקבל את הקוד החד פעמי לדואר אלקטרוני.",
]

# אם אין עדיין היסטוריה – מציגים "שאלות נפוצות" בצד ימין וכותרת במרכז
if len(st.session_state.history) == 0:
    col_right, col_center = st.columns([2, 4])

    with col_right:
        st.markdown("### שאלות נפוצות:")
        for i, q in enumerate(POPULAR_QUESTIONS, start=1):
            st.markdown(f"{i}. {q}")

    with col_center:
        st.markdown("<h2 style='text-align:center;'>איך אפשר לעזור?</h2>", unsafe_allow_html=True)
        st.write("")  # רווח קטן

# =========================================================
#   הצגת היסטוריית שיחה (שאלות ותשובות)
# =========================================================

if len(st.session_state.history) > 0:
    st.markdown("### התכתבות:")
    for q, a in st.session_state.history:
        st.markdown(f"""
<div class="user-bubble">
<strong>שאלה:</strong> {q}
</div>
""", unsafe_allow_html=True)
        st.markdown(f"""
<div class="assistant-text">
<strong>תשובה:</strong><br>{a}
</div>
""", unsafe_allow_html=True)

# =========================================================
#   תיבת שאלה בתחתית – כמו צ'אט
# =========================================================

st.markdown('<div class="question-box"></div>', unsafe_allow_html=True)

with st.form("ask_form", clear_on_submit=True):
    query = st.text_input(" ", placeholder="שאל שאלה והקש Enter")
    submitted = st.form_submit_button("שלח")

if submitted and query.strip():
    item = search_faq(query.strip())
    if item:
        answer = f"{item.answer}\n\nמקור: faq\nשאלה מזוהה: {item.question}"
    else:
        answer = "לא נמצאה תשובה, נסה לנסח את השאלה מחדש."

    st.session_state.history.append((query.strip(), answer))
    # אין צורך ב-experimental_rerun – Streamlit מריץ מחדש אוטומטית
