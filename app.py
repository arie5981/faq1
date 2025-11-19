import os
import re
import openai
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rapidfuzz import fuzz
import requests
import unicodedata

# ודא שמפתח API מוגדר בסביבה
openai.api_key = os.getenv('OPENAI_API_KEY')

# ============================
# שלב 1: הגדרת דף האינטרנט
# ============================
st.set_page_config(
    page_title="תמיכה לאתר מייצגים בגבייה",
    page_icon="💬",
    layout="wide",
)

# ============================
# שלב 2: הגדרות CSS מדויקות לכל הרכיבים
# ============================
st.markdown(
    """
    <style>
    /* הגדרות בסיסיות ל-RTL ולגופנים */
    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: "Alef", "Heebo", "Arial", sans-serif;
        color: #000000;
    }
    
    /* **************** רקע כללי לבן/בהיר (פותר את בעיית הפס האפור) **************** */
    /* מכוון לכל מיכלי Streamlit הראשיים כדי לכפות רקע אחיד */
    .stApp, [data-testid="stAppViewBlock"], [data-testid="stVerticalBlock"], 
    [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stHorizontalBlock"] {
        background-color: #f0f2f6 !important; 
    }
    
    /* הסרת הרווחים החיצוניים של הדף כדי למקסם את שטח התוכן */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* הסתרת כותרת ברירת המחדל של Streamlit */
    h1 { display: none; }
    
    /* ********************************************* */
    /* ריבוע אדום עליון: לוגו וכותרות */
    /* ********************************************* */
    .header-container {
        display: flex;
        flex-direction: row-reverse; /* יישור RTL */
        align-items: center;
        justify-content: flex-start; /* הצמדה לימין */
        gap: 14px;
        margin-bottom: 20px;
        padding-top: 10px;
    }

    .logo-btl {
        height: 40px; 
        width: auto;
    }

    .header-text-main {
        font-size: 26px;
        font-weight: 700;
        color: #1f9cf0; /* כחול */
        line-height: 1.1;
    }

    .header-text-sub {
        font-size: 16px;
        font-weight: 500;
        color: #4fd1ff; /* תכלת */
        line-height: 1.1;
    }

    /* ********************************************* */
    /* ריבוע אדום שני (דף 1): שאלות נפוצות */
    /* ********************************************* */
    .faq-container {
        background-color: #ffffff; /* רקע לבן */
        color: black; 
        padding: 20px;
        border-radius: 12px;
        margin-top: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .faq-container h2 {
        color: black; 
        font-weight: 600;
    }

    /* סגנון הרשימה */
    .faq-container ul {
        margin-bottom: 0;
        padding: 0 10px;
        list-style-position: inside;
        list-style-type: none; 
    }
    .faq-container li {
        color: black; 
        margin-bottom: 8px;
        text-align: right;
    }
    /* מיספור כחול מודגש */
    .faq-container li:before {
        content: attr(data-list-number); 
        color: #1f9cf0; 
        font-weight: 600;
        margin-left: 10px;
        display: inline-block;
        direction: ltr; 
    }

    /* ********************************************* */
    /* ריבוע אדום שלישי (דף 1): כותרת מרכזית */
    /* ********************************************* */
    .main-prompt-title {
        font-size: 28px;
        font-weight: 600;
        color: #000000; 
        text-align: center;
        margin-top: 50px; /* מרווח טוב מ"שאלות נפוצות" */
        margin-bottom: 30px;
        width: 100%;
    }

    /* ********************************************* */
    /* דף שני: עיצוב הצ'אט (היסטוריה) */
    /* ********************************************* */
    
    /* רקע כהה לאזור השיחה עצמו */
    [data-testid="stChatMessage"] {
        background-color: #0e1117; /* רקע כהה */
    }
    
    /* הסרת האייקון של המשתמש */
    .stChatMessage [data-testid="stChatMessageContent"] > div:first-child > div:first-child {
        display: none;
    }

    /* שאלת משתמש (תיבה אפורה מעוגלת) */
    .stChatMessage:nth-child(odd) [data-testid="stMarkdown"] { 
        background-color: #e5e7eb;      
        color: #111111;
        border-radius: 16px; /* מעוגל בפינות */
        padding: 10px 14px;
        max-width: 80%;
        margin-left: 0; 
        margin-right: auto;
        text-align: right;
        direction: rtl;
    }

    /* תשובת מערכת (טקסט לבן רגיל) */
    .stChatMessage:nth-child(even) [data-testid="stMarkdown"] { 
        background-color: transparent; 
        color: white; /* טקסט לבן */
        border-radius: 0; 
        padding: 10px 0;
        max-width: 95%;
        margin-left: auto; 
        margin-right: 0;
        text-align: right;
        direction: rtl;
    }

    /* ********************************************* */
    /* ריבוע אדום שלישי (דף 2): תיבת הקלט הקבועה */
    /* ********************************************* */

    /* מפנה מקום בתחתית הדף לתיבת הקלט הקבועה */
    [data-testid="stVerticalBlock"] {
        padding-bottom: 70px; 
    }

    /* מקבע את המיכל האחרון בתחתית המסך */
    [data-testid="stVerticalBlock"] > div:last-child {
        position: fixed;
        bottom: 0;
        width: 100%;
        max-width: 700px; /* רוחב שמתאים לתוכן הראשי */
        left: 50%;
        transform: translateX(-50%);
        padding: 15px 0; 
        background-color: #f0f2f6; /* רקע בהיר */
        box-shadow: 0 -5px 10px rgba(0,0,0,0.1);
        z-index: 100;
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
        color: #888 !important; /* "שאל שאלה והקש enter" באפור */
    }
    
    /* הסתרת כפתור השליחה (כפי שביקשת) */
    .stButton > button, .stFormSubmitButton {
        display: none !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# שלב 3: טעינת FAQ וניתוח טקסט
# ============================
FAQ_URL = "https://raw.githubusercontent.com/arie5981/faq1/main/faq.txt"

# טיפול בשגיאות טעינה
try:
    faq_text = requests.get(FAQ_URL).text
except requests.exceptions.RequestException as e:
    st.error(f"שגיאה בטעינת קובץ השאלות הנפוצות: {e}")
    faq_text = ""

def normalize_he(s: str) -> str:
    """מנקה ומנרמל טקסט לעברית"""
    if not s: return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\u200e\u200f]", "", s)
    s = re.sub(r"[^\w\s\u0590-\u05FF]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def create_faq_index(faq_text):
    """מנתח את קובץ הטקסט ומחזיר רשימת שאלות/תשובות/וריאציות"""
    faq_items = []
    blocks = re.split(r"(?=שאלה\s*:)", faq_text)
    for block in blocks:
        block = block.strip()
        if not block: continue
        q_match = re.search(r"שאלה\s*:\s*(.+)", block)
        a_match = re.search(r"(?s)תשובה\s*:\s*(.+?)(?:\nהוראה\s*:|\Z)", block)
        v_match = re.search(r"(?s)ניסוחים דומים\s*:\s*(.+?)(?:\nתשובה\s*:|\Z)", block)
        question = q_match.group(1).strip() if q_match else ""
        answer = a_match.group(1).strip() if a_match else ""
        variants = [s.strip(" -\t") for s in v_match.group(1).split("\n") if s.strip()] if v_match else []
        faq_items.append({"question": question, "answer": answer, "variants": variants})
    return faq_items

faq_items = create_faq_index(faq_text)

def search_faq(query: str):
    """מבצע חיפוש פאזי מול ה-FAQ ומחזיר את התשובה הטובה ביותר"""
    query = normalize_he(query)
    scored = []
    
    for item in faq_items:
        all_texts = [item['question']] + item['variants']
        for text in all_texts:
            score = fuzz.token_sort_ratio(query, normalize_he(text))
            scored.append((score, item))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:5]
    
    if top and top[0][0] >= 55:
        return top[0][1]['answer']
    else:
        return "לא נמצאה תשובה. נסה לנסח אחרת."

# ============================
# שלב 4: ממשק משתמש וניהול מצב
# ============================

# 1. ניהול מצב שיחה (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. ריבוע עליון: לוגו וכותרות
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

# 3. פונקציה לטיפול בשאלה ושליחה
def handle_question(question_text):
    if not question_text:
        return
        
    answer_text = search_faq(question_text)
    
    # הוספת השאלה והתשובה להיסטוריית השיחה
    st.session_state.messages.append({"role": "user", "content": question_text})
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    
    # ניקוי תיבת הקלט והפעלת רענון
    st.experimental_rerun()


# 4. תצוגת תוכן הדף (משתנה לפי מצב)
if not st.session_state.messages:
    # ------------------------------------
    # דף ראשון
    # ------------------------------------
    
    # ריבוע אדום שני: שאלות נפוצות
    with st.container():
        st.markdown('<div class="faq-container">', unsafe_allow_html=True)
        st.subheader("שאלות נפוצות:")
        
        # רשימת השאלות
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

    # ריבוע אדום שלישי: כותרת "איך אפשר לעזור?"
    st.markdown("<div class='main-prompt-title'>איך אפשר לעזור?</div>", unsafe_allow_html=True)


else:
    # ------------------------------------
    # דף שני
    # ------------------------------------
    
    # ריבוע אדום שני: היסטוריית השיחה (בתוך אזור רקע כהה)
    # Streamlit מטפל אוטומטית ביצירת ה-ChatMessage עם הרקע הכהה שהגדרנו ב-CSS
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# ------------------------------------
# 5. ריבוע שלישי (קבוע): תיבת שאלה תמיד בתחתית
# ------------------------------------
placeholder_text = "שאל שאלה והקש enter" 

with st.form(key='chat_input_form', clear_on_submit=True):
    # תיבת הקלט
    question = st.text_input(
        "", 
        placeholder=placeholder_text, 
        key="question_input", 
        label_visibility="collapsed"
    )
    # כפתור נסתר: חייבים אותו כדי ש-Enter יעבוד ב-Streamlit Form, אבל ה-CSS מסתיר אותו
    submitted = st.form_submit_button("שלח", help="לחץ Enter כדי לשלוח") 
    
    if submitted and question:
        handle_question(question)
