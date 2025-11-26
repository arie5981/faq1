# ============================================
#   עוזר אתר מייצגים – גרסה ל-Streamlit
#   (מעודכן: פתרון סופי למיקום כפתור וצמצום רווחים)
# ============================================

import streamlit as st
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional
from rapidfuzz import fuzz, process

import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import json 

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
#   משתנה גלובלי לקישורים
# ============================================
GLOBAL_CONTACT_DETAILS = {}


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
    margin: 0.2rem 0 0 0; 
}

/* תיבת הקלט */
.question-box {
    position: relative;
    margin-top: 1rem;
    padding-top: 0.5rem;
    border-top: 1px solid #333;
}

/* הסתרת כפתור "שלח" של הטופס */
div[data-testid="stForm"] div.stButton button {
    visibility: hidden; 
    width: 0.1px;
    padding: 0;
    margin: 0;
    height: 0.1px;
}

/* 💡 CSS לשינוי עיצוב הכפתורים: קטן יותר ומוצמד לשאלה ברשימה */
div.stButton button { 
    /* עיצוב כפתור התשובה הקטן */
    height: 25px; /* גובה נמוך יותר */
    line-height: 1;
    padding: 2px 8px; /* צמצום Padding אנכי */
    font-size: 0.8rem;
    border-radius: 4px;
    background-color: #3b82f6; /* כחול */
    color: white;
    border: none;
    white-space: nowrap;
    width: auto; 
    margin: 0;
}
div.stButton button:hover {
    background-color: #2563eb;
}

/* 💡 כלל קריטי: מכריח את הטור של הכפתור להתיישר לימין (Flex-End) */
[data-testid="stColumn"] {
    display: flex;
    flex-direction: column;
    align-items: flex-end; 
}

/* ודא שהטקסט בתוך הטורים נשאר מיושר לימין */
[data-testid="stColumn"] > div {
    width: 100%;
    text-align: right;
}

/* 💡 כללי צמצום רווחים אנכיים בין השאלות */
.st-emotion-cache-1r6r8qj { /* קונטיינר העמודות הראשי */
    margin-bottom: 0.5rem !important; /* רווח קטן בין השורות */
    padding-bottom: 0px !important; 
    padding-top: 0px !important;
}

/* 💡 צמצום padding בתוך ה-Markdown של השאלה */
.st-emotion-cache-1c9v68d { 
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    line-height: 1.2; /* צמצום גובה השורה */
}

</style>
""", unsafe_allow_html=True)

# ============================================
#   כותרת עליונה עם לוגו וטקסט
# ============================================
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

try:
    raw_faq = read_txt_utf8(FAQ_PATH)
except FileNotFoundError:
    st.error(f"❌ קובץ FAQ לא נמצא בנתיב: {FAQ_PATH}. ודא שהקובץ נמצא בתיקייה הנכונה.")
    st.stop()


# ============================================
#   עיבוד ה-FAQ וריכוז הקישורים
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
    instruction: Optional[str] = None
    contact_details: Optional[dict] = None 

def parse_faq_new(text: str) -> List[FAQItem]:
    items = []
    
    global GLOBAL_CONTACT_DETAILS
    GLOBAL_CONTACT_DETAILS.clear() # איפוס המילון הגלובלי

    # 1. חילוץ כל הקישורים הגלובליים מכל הטקסט
    all_c_matches = re.findall(r">>([^:]+?)\s*:\s*([^<]+?)<<", text)
    GLOBAL_CONTACT_DETAILS = {k.strip(): v.strip() for k, v in all_c_matches}
    
    # 2. הסרת כל הבלוקים של הקישורים הגלובליים מטקסט ה-FAQ כדי למנוע הפרעה לניתוח השאלות
    text_without_links = re.sub(r">>([^:]+?)\s*:\s*([^<]+?)<<", "", text)
    
    # 3. פיצול לבלוקים של שאלות
    blocks = re.split(r"(?=שאלה\s*:)", text_without_links) 

    for b in blocks:
        b = b.strip()
        if not b:
            continue

        q_match = re.search(r"שאלה\s*:\s*(.+)", b)
        v_match = re.search(r"(?s)ניסוחים דומים\s*:\s*(.+?)(?:\nתשובה\s*:|\Z)", b)
        a_match = re.search(r"(?s)תשובה\s*:\s*(.+?)(?:\nהוראה\s*:|\Z)", b)
        i_match = re.search(r"(?s)הוראה\s*:\s*(.+?)(?:\n>>|\Z)", b)
        
        question = q_match.group(1).strip() if q_match else ""
        
        answer = ""
        if a_match:
            raw_answer_content = a_match.group(1)
            lines = raw_answer_content.splitlines()
            cleaned_lines = [line.strip() for line in lines]
            answer = '\n'.join(cleaned_lines).strip()
            
        variants = []
        if v_match:
            raw = v_match.group(1)
            variants = [s.strip(" -\t") for s in raw.split("\n") if s.strip()]

        instruction = i_match.group(1).strip() if i_match else None
        
        items.append(FAQItem(question, variants, answer, instruction, contact_details={}))

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
#   פונקציה לעיבוד תוכן התשובה (משתמשת בגלובלי)
# ============================================
def process_answer_content(item: FAQItem) -> str:
    global GLOBAL_CONTACT_DETAILS 
    
    answer_text = item.answer.strip()
    
    # 2. החלפת מילות מפתח בקישורי Markdown בתוך ה-ANSWER
    if GLOBAL_CONTACT_DETAILS:
        for key, value in GLOBAL_CONTACT_DETAILS.items():
            markdown_link = f"[{key}]({value})"
            answer_text = answer_text.replace(f"[{key}]", markdown_link)
        
        
    # 3. טיפול בשדה 'הוראה' והוספתו בסוף
    if item.instruction: 
        instruction = item.instruction
        
        # 3א. החלפת מילות מפתח בקישורי Markdown בתוך ההוראה
        for key, value in GLOBAL_CONTACT_DETAILS.items():
            markdown_link = f"[{key}]({value})"
            instruction = instruction.replace(f"[{key}]", markdown_link)
        
        answer_text += f"\n\n**הערות והוראות:** {instruction}"

    # הוספת \n\n בין פסקאות
    final_content = answer_text.replace('\n', '\n\n')
    return final_content


# ============================================
#   חיפוש FAQ – fuzzy + embeddings
# ============================================

def search_faq(query: str) -> str:
    nq = normalize_he(query)

    # --- חיפוש פאזי על שאלות וניסוחים ---
    scored = []
    for i, item in enumerate(faq_items):
        all_texts = [item.question]
