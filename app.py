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
# שלב 3: הגדרות CSS והתאמות עיצוביות
# ============================
st.markdown(
    """
    <style>
    /* הגדרות גלובליות ו-RTL */
    /* הרקע הכללי והטקסט הכללי של האפליקציה (כפי שמופיע בצילום, רוב הדף בהיר) */
    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: "Alef", "Heebo", "Arial", sans-serif;
        color: #000000; /* טקסט כללי שחור */
    }
    
    /* קביעת הרקע הראשי של הדף כבהיר (פותר את הפס האפור החודר) */
    /* הסלקטור הזה מכוון למיכל הראשי של Streamlit */
    [data-testid="stAppViewBlock"] > section, [data-testid="stVerticalBlock"] > div:has([data-testid="stHorizontalBlock"]) {
        background-color: #f0f2f6; /* רקע בהיר (לבן/אפור בהיר) */
    }
    
    /* מכל הכותרת העליונה (לוגו + טקסט) */
    .header-container {
        display: flex;
        flex-direction: row-reverse; /* יישור RTL */
        align-items: center;
        justify-content: flex-end; /* הצמדה לימין */
        gap: 14px;
        margin-bottom: 20px;
        padding-top: 10px;
    }

    /* עיצוב הלוגו */
    .logo-btl {
        height: 40px; 
        width: auto;
        align-self: flex-start; 
        padding-top: 5px;
    }

    /* עיצוב הטקסט הראשי בכותרת (כחול) */
    .header-text-main {
        font-size: 26px;
        font-weight: 700;
        color: #1f9cf0; /* כחול */
        line-height: 1.1;
    }

    /* עיצוב טקסט משני בכותרת (תכלת) */
    .header-text-sub {
        font-size: 16px;
        font-weight: 500;
        color: #4fd1ff; /* תכלת */
        line-height: 1.1;
    }

    /* כותרת מרכזית בדף הראשון */
    .main-prompt-title {
        font-size: 28px;
        font-weight: 600;
        color: #000000; /* שחור על רקע בהיר */
        text-align: center;
        margin-top: 100px; 
        margin-bottom: 30px;
        width: 100%;
    }

    /* הסתרת הכותרת הדיפולטית של Streamlit */
    h1 { display: none; }

    /* ********************************************* */
    /* עיצוב המיכל הבהיר עבור שאלות נפוצות (דף ראשון) */
    /* ********************************************* */
    .faq-container {
        background-color: #f0f2f6; /* רקע בהיר */
        color: black !important; /* טקסט שחור */
        padding: 20px;
        border-radius: 12px;
        margin-top: 30px;
    }
    
    .faq-container h2 {
        color: black !important; /* כותרת הרשימה בשחור */
        font-weight: 600;
    }

    /* סגנון עבור רשימת השאלות הנפוצות */
    .faq-container ul {
        margin-bottom: 0;
        padding: 0 10px;
        list-style-position: inside;
        list-style-type: none; /* הסרת הנקודות המוגדרות כברירת מחדל */
    }
    .faq-container li {
        color: black !important; /* טקסט שחור כנדרש */
        margin-bottom: 8px;
        text-align: right;
    }
    .faq-container li:before {
        content: attr(data-list-number); /* שימוש באטריבוט למיספור */
        color: #1f9cf0; /* צבע כחול למספור */
        font-weight: 600;
        margin-left: 10px;
        display: inline-block;
        direction: ltr; /* להפוך את כיוון המספר */
    }


    /* ************** סגנון בועות צ'אט (לדף השני) ************** */
    
    /* רקע הצ'אט צריך להישאר כהה */
    /* זה מכוון למיכל הראשי של ההיסטוריה */
    [data-testid="stChatMessage"]:nth-child(even) + [data-testid="stChatMessage"] + [data-testid="stVerticalBlock"] {
        background-color: #0e1117; 
        color: white;
    }
    
    /* מיכל ראשי של הצ'אט (האזור השחור בצילום) */
    /* מכוון לכל האזור של הצ'אט - נותן לו רקע שחור */
    [data-testid="stChatMessage"] {
        background-color: #0e1117; 
    }
    
    /* הסתרת האייקון של המשתמש */
    .stChatMessage [data-testid="stChatMessageContent"] > div:first-child > div:first-child {
        display: none;
    }

    /* מיכל השאלה (משתמש) - תיבה אפורה מעוגלת */
    .stChatMessage:nth-child(odd) [data-testid="stMarkdown"] { 
        background-color: #e5e7eb;      
        color: #111111;
        border-radius: 16px;
        padding: 10px 14px;
        max-width: 80%;
        margin-left: 0; 
        margin-right: auto;
        text-align: right;
        direction: rtl;
    }

    /* מיכל התשובה (מערכת) - טקסט לבן רגיל (כי הרקע שחור) */
    .stChatMessage:nth-child(even) [data-testid="stMarkdown"] { 
        background-color: transparent; 
        color: white; 
        border-radius: 0; 
        padding: 10px 0;
        max-width: 95%;
        margin-left: auto; 
        margin-right: 0;
        text-align: right;
        direction: rtl;
    }
    
    .stChatMessage:nth-child(even) { 
        text-align: right !important;
        direction: rtl !important;
        margin-bottom: 15px; 
    }

    /* ************** תיבת השאלה התחתונה - פתרון בעיה 1 ************** */

    /* סלקטור חזק יותר למיקום תיבת השאלה בתחתית המסך (מקובע) */
    /* מכוון ל-Container שמכיל את הטופס הראשי. זה יפתור את הבעיה ברוב המקרים */
    [data-testid="stVerticalBlock"] > div:last-child {
        position: fixed;
        bottom: 0;
        width: 100%;
        max-width: 700px; /* רוחב מרבי של התוכן הראשי */
        left: 50%;
        transform: translateX(-50%);
        padding: 15px 0; /* רווח תחתון וקצת רווח צדדי */
        /* רקע שקוף או כמעט שקוף כדי למנוע את הפס האפור החודר */
        background-color: rgba(240, 242, 246, 0.95); 
        box-shadow: 0 -5px 10px rgba(0,0,0,0.1);
        z-index: 100;
    }
    
    /* סגנון לטופס עצמו בתוך המיכל התחתון */
    [data-testid="stForm"] {
        padding: 0 15px 0 15px; /* רק ריפוד צדדי */
    }
    
    /* עיצוב תיבת הטקסט בתוך ה-Form */
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
        border-radius: 999px; /* מעוגל בפינות */
        border: 1px solid #1f9cf0; /* מסגרת כחולה */
        padding-right: 18px;
        padding-left: 18px;
        background-color: white !important;
        color: black !important;
        height: 50px;
    }

    .stTextInput input::placeholder {
        color: #888 !important; /* טקסט אפור בתוך התיבה */
    }
    
    /* הסתרת כפתור השליחה הדיפולטי של Streamlit (השליחה מתבצעת ב-Enter) */
    .stButton > button {
        display: none !important;
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
# שלב 8: ממשק משתמש וניהול מצב (Session State)
# ============================

# 1. ניהול מצב שיחה (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. כותרת עליונה ולוגו
# הלוגו והטקסט מעוצבים כעת יחד באמצעות Markdown
st.markdown(
    f"""
    <div class="header-container">
      <img class="logo-btl" src="https://raw.githubusercontent.com/arie5981/faq1/main/logobtl.png" alt="לוגו הביטוח הלאומי">
      <div>
        <div class="header-text-main">הביטוח הלאומי</div>
        <div class="header-text-sub">תמיכה לאתר מייצגים בגבייה</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 3. פונקציה לטיפול בשאלה
def handle_question(question_text):
    # הפעלת פונקציית החיפוש
    answer_text = search_faq(question_text)
    
    # הוספת השאלה והתשובה להיסטוריית השיחה
    st.session_state.messages.append({"role": "user", "content": question_text})
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    
    # ניקוי תיבת הקלט והפעלת רענון
    st.experimental_rerun()


# 4. הצגת ממשק המשתמש (מטפלת בשני המצבים: דף ראשון / דף צ'אט)

if not st.session_state.messages:
    # ------------------------------------
    # ממשק דף ראשון (ללא היסטוריה)
    # ------------------------------------
    
    # כותרת מרכזית ("איך אפשר לעזור?")
    st.markdown("<div class='main-prompt-title'>איך אפשר לעזור?</div>", unsafe_allow_html=True)
    
    # מיכל עם רקע בהיר לשאלות הנפוצות והטקסט השחור
    with st.container():
        # כותרת שאלות נפוצות
        st.markdown('<div class="faq-container">', unsafe_allow_html=True)
        st.subheader("שאלות נפוצות:")
        
        # רשימת השאלות (משתמשת ב-Markdown בתוך המיכל הבהיר)
        st.markdown(
            """
            <ul class="faq-list">
                <li data-list-number="1."> איך מוסיפים משתמש חדש באתר מייצגים.</li>
                <li data-list-number="2."> מקבל הודעה שאחד או יותר מנתוני ההזדהות שגויים.</li>
                <li data-list-number="3."> איך יוצרים קיצור דרך לאתר מייצגים על שולחן העבודה.</li>
                <li data-list-number="4."> רוצה לקבל את הקוד החד פעמי לדואר אלקטרוני.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------
    # תיבת שאלה מקובעת (מוצגת תמיד)
    # ------------------------------------
    # נשתמש בטופס גנרי אחד לטיפול בקלט
    
    # Streamlit תמיד מציב את הטופס בסוף העמוד. אנו משתמשים ב-CSS כדי למקם אותו 
    # בצורה מקובעת בתחתית המסך.
    with st.form(key='chat_input_form', clear_on_submit=True):
        question = st.text_input(
            "", 
            placeholder="שאל שאלה והקש Enter", 
            key="question_input", 
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("שאל", help="לחץ Enter כדי לשלוח") 
        
        if submitted and question:
            handle_question(question)


else:
    # ------------------------------------
    # ממשק צ'אט (עם היסטוריה)
    # ------------------------------------
    
    # 5. הצגת היסטוריית השיחה
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 6. תיבת שאלה בתחתית המסך
    # נשתמש שוב באותו טופס כדי לשמור על המיקום הקבוע
    with st.form(key='chat_input_form_after', clear_on_submit=True):
        question = st.text_input(
            "", 
            placeholder="שאל שאלה נוספת והקש Enter", 
            key="question_input_after", 
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("שלח", help="לחץ Enter כדי לשלוח") 
        
        if submitted and question:
            handle_question(question)
