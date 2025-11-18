# ============================================
#   עוזר אתר מייצגים – גרסה מלאה ל-Streamlit
# ============================================

import streamlit as st
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional
from rapidfuzz import fuzz
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ============================================
#   הגדרות עמוד ו-CSS ל-RTL
# ============================================
st.set_page_config(page_title="עוזר מייצגים", layout="wide")

st.markdown("""
<style>
html, body, .markdown-text-container, .stTextInput, .stTextArea, .stMarkdown {
    direction: rtl;
    text-align: right !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================
#   שלב 1 – טעינת FAQ מתוך קובץ בריפו
# ============================================
FAQ_PATH = "faq.txt"

def read_txt_utf8(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

raw_faq = read_txt_utf8(FAQ_PATH)



# ============================================
#   שלב 2 – פונקציות מקוריות מה-Colab
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


# ==== שלב Embeddings ====
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = []
for i, item in enumerate(faq_items):
    merged = " | ".join([item.question] + item.variants)
    docs.append(Document(page_content=merged, metadata={"idx": i}))

faq_store = FAISS.from_documents(docs, embeddings)


# ============================================
#   שלב החיפוש – fuzzy + embeddings
# ============================================
def search_faq(query: str) -> str:
    nq = normalize_he(query)

    # fuzzy
    scored = []
    for i, item in enumerate(faq_items):
        all_texts = [item.question] + item.variants
        for t in all_texts:
            score = fuzz.token_sort_ratio(nq, normalize_he(t))
            scored.append((score, i, t))

    scored.sort(reverse=True, key=lambda x: x[0])
    best = scored[0]

    if best[0] >= 55:
        return faq_items[best[1]].answer + f"\n\n🔹 שאלה מזוהה: {faq_items[best[1]].question}"

    # embeddings fallback
    hits = faq_store.similarity_search_with_score(query, k=3)

    best_doc, best_score = hits[0]
    if best_score < 1.1:
        idx = best_doc.metadata["idx"]
        return faq_items[idx].answer + f"\n\n🔹 שאלה מזוהה: {faq_items[idx].question}"

    return "לא נמצאה תשובה, נסה לנסח מחדש."


# ============================================
#   שלב 3 – ניהול שיחה כמו ChatGPT
# ============================================
if "chat" not in st.session_state:
    st.session_state.chat = []


st.title("📘 עוזר אתר מייצגים – שאלות ותשובות")


# תיבת שאלה
with st.form("ask_form", clear_on_submit=True):
    query = st.text_input("❓ שאל שאלה:")
    send = st.form_submit_button("שלח")

if send and query:
    ans = search_faq(query)
    st.session_state.chat.append({"q": query, "a": ans})


# הצגת כל השיחה
for turn in st.session_state.chat:
    st.markdown(f"""
### ❓ שאלה:
{turn['q']}

### 💬 תשובה:
{turn['a']}
---
""" )
