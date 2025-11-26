# ============================================
#   עוזר אתר מייצגים – גרסה ל-Streamlit
#   (קוד סופי: תיקון CSS מלא ליישור כפתורים וצמצום רווחים)
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
#   הגדרות עמוד ו־CSS ל־RTL + עיצוב סופי
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

/* 💡 CSS לשינוי עיצוב הכפתורים: קטן יותר וצמוד */
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

/* ============================================================= */
/* 🎯 תיקון סופי למיקום הכפתור: דריסת Flexbox של Streamlit */
/* ============================================================= */
[data-testid="stColumn"] {
    /* מאפשר חלוקה ל-2 טורים ברוחב שונה */
    display: flex !important;
    flex-direction: row !important; 
    align-items: center !important; /* יישור אנכי למרכז */
    gap: 0.5rem !important; /* צמצום המרווח בין הטורים */
}

/* 💡 עבור הטור של הכפתור (הטור השני, nth-child(2)), נצמיד את התוכן שלו לימין (Flex-End) */
[data-testid="stColumn"]:nth-child(2) > div {
    display: flex;
    justify-content: flex-end; /* CRITICAL: הכפתור נצמד לימין הטור שלו = מיד אחרי השאלה */
    align-items: center;
    width: 100%; 
    padding: 0 !important;
}

/* ודא שהטקסט בטור של השאלה (הראשון) מיושר לימין */
[data-testid="stColumn"]:nth-child(1) > div {
    text-align: right;
    padding: 0 !important;
}

/* ============================================================= */
/* 🎯 תיקון סופי למרווחים: דריסה אגרסיבית של גובה השורה */
/* ============================================================= */

/* קונטיינר העמודות הראשי - צמצום Margin בין השורות */
.st-emotion-cache-1r6r8qj { 
    margin-bottom: 0.25rem !important; 
    padding-bottom: 0px !important; 
    padding-top: 0px !important;
}

/* צמצום padding ו-line-height בתוך ה-Markdown של השאלה */
.st-emotion-cache-1c9v68d { 
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    line-height: 1.2 !important; 
    margin: 0 !important;
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
      <span class="header-text-sub
