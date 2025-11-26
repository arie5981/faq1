# ============================================
#   עוזר אתר מייצגים – גרסה ל-Streamlit
#   קורא faq.txt מהריפו, מציג צ'אט בסגנון ChatGPT
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
/* נשתמש ב-data-testid כדי למקד את ההסתרה רק לכפתור השליחה בתוך הטופס */
div[data-testid="stForm"] div.stButton button {
    visibility: hidden; 
    width: 0.1px;
    padding: 0;
    margin: 0;
    height: 0.1px;
}


/* CSS נוסף: עיצוב כפתורי השאלות כקישורים */
/* הכפתורים של השאלות מחוץ לטופס ייראו כך: */
div.stButton button { 
    text-align: right !important;
    width: 100%;
    margin-bottom: 0.5rem;
    padding: 0.6rem;
    border-radius: 6px;
    border: 1px solid #ccc;
    background-color: #f0f0f0;
    font-size: 1rem;
}
div.stButton button:hover {
    background-color: #e0e0e0;
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
    # 💡 קורא את קובץ faq.txt שצריך להיות באותה תיקייה
    raw_faq = read_txt_utf8(FAQ_PATH)
except FileNotFoundError:
    # טיפול בשגיאה למקרה שהקובץ חסר
    st.error(f"❌ קובץ FAQ לא נמצא בנתיב: {FAQ_PATH}. ודא שהקובץ נמצא בתיקייה הנכונה.")
    st.stop()


# ============================================
#   עיבוד ה-FAQ (לוגיקה משופרת מ-faq7.py)
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
    blocks = re.split(r"(?=שאלה\s*:)", text) 

    for b in blocks:
        b = b.strip()
        if not b:
            continue

        q_match = re.search(r"שאלה\s*:\s*(.+)", b)
        v_match = re.search(r"(?s)ניסוחים דומים\s*:\s*(.+?)(?:\nתשובה\s*:|\Z)", b)
        a_match = re.search(r"(?s)תשובה\s*:\s*(.+?)(?:\nהוראה\s*:|\Z)", b)
        i_match = re.search(r"(?s)הוראה\s*:\s*(.+?)(?:\n>>|\Z)", b)
        c_match = re.findall(r">>([^:]+?)\s*:\s*([^<]+?)<<", b)

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
        contact_details = {k.strip(): v.strip() for k, v in c_match} if c_match else None
        
        items.append(FAQItem(question, variants, answer, instruction, contact_details))

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

def process_answer_content(item: FAQItem) -> str:
    answer_text = item.answer.strip()
    
    if item.instruction and item.contact_details:
        instruction = item.instruction
        contact_details = item.contact_details
        
        # החלפת מילות מפתח בקישורי Markdown
        for key, value in contact_details.items():
            markdown_link = f"[{key}]({value})"
            instruction = instruction.replace(f"[{key}]", markdown_link)
        
        answer_text += f"\n\n**הערות והוראות:** {instruction}"

    # הוספת \n\n בין פסקאות
    final_content = answer_text.replace('\n', '\n\n')
    return final_content


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

    # קריטריון חיפוש פאזי מחמיר יותר
    if best_score >= 80:
        item = faq_items[best_idx]
        
        final_content = process_answer_content(item)
        
        return f"{final_content}\n\nמקור: faq\n\nשאלה מזוהה: {item.question}"

    # --- fallback: embeddings (עם שיפור ניקוד) ---
    hits = faq_store.similarity_search_with_score(query, k=5)
    
    # בונוס ניקוד לדימיון פאזי
    boosted_hits = []
    for doc, score in hits:
        idx = doc.metadata["idx"]
        item = faq_items[idx]
        fuzzy_score = fuzz.token_sort_ratio(nq, normalize_he(item.question))
        boosted_score = (score * 0.7) + (1.0 - (fuzzy_score / 100)) * 0.3
        boosted_hits.append((doc, boosted_score, idx))

    boosted_hits.sort(key=lambda x: x[1])
    
    best_doc, best_score, best_idx = boosted_hits[0]

    if best_score <= 1.1: # סף הצלחה מעודכן
        result_item = faq_items[best_idx]
        
        final_content = process_answer_content(result_item)

        # 💥 הוספת שאלות קשורות (מוחזרות כסטרינג JSON)
        similar_questions = [
            faq_items[d.metadata["idx"]].question
            for d, s, _ in boosted_hits[1:4] 
            if s <= 1.3 and faq_items[d.metadata["idx"]].question.strip() != result_item.question.strip()
        ][:3]
        
        if similar_questions:
            sq_json = json.dumps(similar_questions, ensure_ascii=False)
            final_content += f"\n\n---SIMILAR_QUESTIONS---{sq_json}"

        return f"{final_content}\n\nמקור: faq\n\nשאלה מזוהה (סמנטי): {result_item.question}"

    return "לא נמצאה תשובה, נסה לנסח את השאלה מחדש."

# ============================================
#   פונקציית Callback לטיפול בשליחת הטופס / לחיצה על שאלה
# ============================================
def handle_submit(query_text=None):
    if query_text is None:
        query = st.session_state.query_input
    else:
        query = query_text

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        answer = search_faq(query)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.query_input = "" 


# ============================================
#   ניהול שיחה כמו ChatGPT
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# שאלות נפוצות למסך הראשון
POPULAR_QUESTIONS = [
    "איך מוסיפים משתמש חדש באתר מייצגים.",
    "מקבל הודעה שאחד או יותר מנתוני ההזדהות שגויים.",
    "איך יוצרים קיצור דרך לאתר מייצגים על שולחן העבודה.",
    "רוצה לקבל את הקוד החד פעמי לדואר אלקטרוני.",
]

st.markdown("")

# ----------------------------------------------------
# 💥 מסך פתיחה עם שאלות נפוצות ככפתורים
# ----------------------------------------------------
if len(st.session_state.messages) == 0:
    st.markdown("### שאלות נפוצות:")
    
    for i, q in enumerate(POPULAR_QUESTIONS, start=1):
        # 💥 תיקון: שימוש במרכאות משולשות (f"""...""") למניעת שגיאת תחימה
        st.button(
            f"""{q} **<לתשובה לחץ כאן>**""", 
            key=f"popular_q_{i}", 
            on_click=handle_submit, 
            args=(q,)
        )

    st.markdown("## איך אפשר לעזור?")
    st.markdown("")

# ----------------------------------------------------
# 💥 תיבת הקלט מופיעה כעת ראשונה
# ----------------------------------------------------
st.markdown('<div class="question-box"></div>', unsafe_allow_html=True)

with st.form("ask_form", clear_on_submit=False): 
    query = st.text_input(" ", 
                          placeholder="שאל שאלה והקש Enter", 
                          key="query_input")
    
    # הלחצן "שלח" מוסתר על ידי ה-CSS למעלה
    submitted = st.form_submit_button("שלח", on_click=handle_submit)

# ----------------------------------------------------
# 💥 מפריד ויזואלי בין טופס הקלט להיסטוריה
# ----------------------------------------------------
if len(st.session_state.messages) > 0:
    st.markdown("---") 

# =======================================================================
# 💥 הצגת היסטוריית שיחה בזוגות בסדר הפוך (Q -> A) + שאלות קשורות ככפתורים
# =======================================================================

user_indices = [i for i, msg in enumerate(st.session_state.messages) if msg["role"] == "user"]

for user_idx in user_indices[::-1]:
    
    # 1. הצגת הודעת השאלה
    user_msg = st.session_state.messages[user_idx]
    st.markdown(f"""
<div class="user-bubble">
<strong>שאלה:</strong> {user_msg['content']}
</div>
""", unsafe_allow_html=True)
    
    # 2. הצגת הודעת התשובה (אם קיימת)
    assistant_idx = user_idx + 1
    if assistant_idx < len(st.session_state.messages):
        assistant_msg = st.session_state.messages[assistant_idx]
        raw_display_content = assistant_msg['content'] 
        
        # 💥 חילוץ שאלות קשורות מתוך התוכן
        similar_questions = []
        sq_match = re.search(r"---SIMILAR_QUESTIONS---(.*)", raw_display_content)
        
        if sq_match:
            try:
                sq_json_str = sq_match.group(1).strip()
                similar_questions = json.loads(sq_json_str)
                # הסרת ה-JSON מתוכן התצוגה הראשי
                display_content = raw_display_content.replace(f"\n\n---SIMILAR_QUESTIONS---{sq_json_str}", "").strip()
            except json.JSONDecodeError:
                display_content = raw_display_content
        else:
            display_content = raw_display_content
            
        # הצגת התווית "תשובה:" ועיצוב כללי באמצעות HTML
        st.markdown(f"""
<div class="assistant-text">
<strong>תשובה:</strong>
</div>
""", unsafe_allow_html=True)
        
        # הצגת התוכן (כולל ה-Markdown) ב-st.markdown נפרד
        st.markdown(display_content, unsafe_allow_html=True)

        # 💥 הצגת השאלות הקשורות ככפתורים
        if similar_questions:
            st.markdown("---") # מפריד
            st.markdown("#### שאלות קשורות:")
            
            base_key = f"similar_q_{user_idx}" 
            
            for i, sq in enumerate(similar_questions):
                # 💥 תיקון: שימוש במרכאות משולשות (f"""...""") למניעת שגיאת תחימה
                st.button(
                    f"""{sq} **<לתשובה לחץ כאן>**""", 
                    key=f"{base_key}_{i}", 
                    on_click=handle_submit, 
                    args=(sq,)
                )
