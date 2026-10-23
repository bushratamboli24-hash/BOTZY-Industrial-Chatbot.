import base64
import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------- CONFIGURATION ----------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BACKGROUND_PATH = (
    Path(__file__).parent / "assets" / "background.jpg"
)

st.set_page_config(
    page_title="BOTZY - Industrial AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# ---------------- BACKGROUND IMAGE ----------------

if BACKGROUND_PATH.exists():
    with open(BACKGROUND_PATH, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    background_css = f"""
    .stApp {{
        background-image:
            linear-gradient(
                rgba(255, 255, 255, 0.78),
                rgba(239, 243, 255, 0.82)
            ),
            url("data:image/jpeg;base64,{encoded_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    """
else:
    background_css = """
    .stApp {
        background: linear-gradient(135deg, #f5f7ff, #e9eaff);
    }
    """

st.markdown(
    f"""
    <style>
    {background_css}

    .block-container {{
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }}

    .hero {{
        padding: 26px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.90);
        border: 1px solid #dbe4ff;
        box-shadow: 0 6px 24px rgba(54, 66, 120, 0.10);
        margin-bottom: 22px;
    }}

    .hero h1 {{
        color: #3730a3;
        font-size: 38px;
        margin-bottom: 8px;
    }}

    .hero p {{
        color: #475569;
        font-size: 16px;
    }}

    .panel {{
        padding: 20px;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #dbe4ff;
        border-radius: 18px;
        margin-bottom: 16px;
    }}

    div[data-testid="stChatMessage"] {{
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #e0e7ff;
        border-radius: 14px;
        padding: 12px;
    }}

    div[data-testid="stFileUploader"] {{
        background: rgba(255, 255, 255, 0.88);
        border: 1px dashed #818cf8;
        border-radius: 14px;
        padding: 12px;
    }}

    .stButton > button {{
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }}

    h2, h3 {{
        color: #3730a3;
    }}

    [data-testid="stHeader"] {{
        background: rgba(255, 255, 255, 0.25);
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- BOTZY HEADER ----------------

st.markdown(
    """
    <div class="hero">
        <h1>🤖 BOTZY</h1>
        <p>
            Your friendly AI partner for industrial documents
            and technical knowledge.
        </p>
        <p>
            📄 Document Analysis &nbsp; • &nbsp;
            🧠 AI-Powered Answers &nbsp; • &nbsp;
            ⚙️ Industrial Knowledge
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- LOAD AI MODELS ----------------

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    return embedding_model, tokenizer, model


# ---------------- READ PDF ----------------

def extract_chunks(pdf_file):
    reader = PdfReader(pdf_file)
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        words = page_text.split()

        for start in range(0, len(words), 120):
            passage = " ".join(words[start:start + 120])

            if passage.strip():
                chunks.append({
                    "text": passage,
                    "page": page_number
                })

    return chunks, len(reader.pages)


# ---------------- SEARCH DOCUMENT ----------------

def retrieve_passages(
    question, chunks, document_vectors, embedding_model
):
    if not chunks:
        return []

    question_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    scores = np.dot(document_vectors, question_vector)

    best_indices = np.argsort(scores)[-3:][::-1]

    return [
        chunks[i]
        for i in best_indices
        if scores[i] > 0.20
    ]


# ---------------- GENERATE ANSWER ----------------

def generate_answer(question, passages, tokenizer, model):
    if not passages:
        return (
            "I could not find relevant information in the "
            "uploaded document. Please try another question."
        )

    context = "\n\n".join(
        f"Page {p['page']}: {p['text']}"
        for p in passages
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are BOTZY, an industrial technical assistant. "
                "Answer in simple language using only the supplied "
                "document context. If information is missing, say "
                "it is not available in the document. Do not invent "
                "technical facts. For dangerous industrial operations, "
                "remind the user to follow the official safety manual."
            )
        },
        {
            "role": "user",
            "content": (
                f"Document context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Give a clear, concise answer and mention relevant "
                "page numbers."
            )
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## 🤖 BOTZY")
    st.caption("Your Industrial AI Assistant")
    st.divider()

    st.markdown("### ⚙️ Features")
    st.write("📄 PDF document Q&A")
    st.write("🧠 Hugging Face AI")
    st.write("🔎 Semantic document search")

    if BACKGROUND_PATH.exists():
        st.success("Background image found")
    else:
        st.warning(
            "Image not found. Check assets/background.jpg"
        )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------- DOCUMENT UPLOAD ----------------

left_column, right_column = st.columns([1, 2])

with left_column:
    st.markdown("### 📁 Upload Industrial PDF")

    uploaded_pdf = st.file_uploader(
        "Choose your PDF document",
        type=["pdf"]
    )

    st.caption(
        "Upload a text-based industrial or technical PDF."
    )

    st.markdown("### 💡 Suggested Questions")

    suggestions = [
        "What are the main components in this document?",
        "What safety precautions are mentioned?",
        "What is the maintenance schedule?",
        "Explain the operating procedure."
    ]

    for suggestion in suggestions:
        if st.button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion

with right_column:
    st.markdown("### 💬 Chat with BOTZY")

    if "messages" not in st.session_state:
        st.session_state.messages = []


# ---------------- CHAT INTERFACE ----------------

if uploaded_pdf is None:
    with right_column:
        st.markdown(
            """
            <div class="panel">
                <h3>👋 Welcome to BOTZY!</h3>
                <p>
                    Upload an industrial PDF to get started.
                    BOTZY can search your document and answer
                    questions using a Hugging Face language model.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    try:
        pdf_bytes = uploaded_pdf.getvalue()
        pdf_id = hashlib.md5(pdf_bytes).hexdigest()

        from io import BytesIO

        chunks, page_count = extract_chunks(
            BytesIO(pdf_bytes)
        )

        if not chunks:
            st.error(
                "No readable text found. Please upload a "
                "text-based PDF, not a scanned image PDF."
            )
            st.stop()

        st.success(
            f"📄 {uploaded_pdf.name} | {page_count} pages | "
            f"{len(chunks)} passages"
        )

        with st.spinner("Loading BOTZY AI models..."):
            embedding_model, tokenizer, model = load_models()

        # Calculate document embeddings only when the PDF changes.
        if st.session_state.get("pdf_id") != pdf_id:
            with st.spinner("Preparing document search..."):
                document_vectors = embedding_model.encode(
                    [chunk["text"] for chunk in chunks],
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                st.session_state.pdf_id = pdf_id
                st.session_state.chunks = chunks
                st.session_state.document_vectors = document_vectors
                st.session_state.messages = []

        # Reuse saved data on later Streamlit reruns.
        chunks = st.session_state.chunks
        document_vectors = st.session_state.document_vectors

        with right_column:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            question = st.chat_input(
                "Ask BOTZY a question about your PDF..."
            )

            # Allow suggested-question buttons to start a question.
            if st.session_state.get("pending_question"):
                question = st.session_state.pending_question
                del st.session_state.pending_question

            if question:
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })

                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("BOTZY is preparing your answer..."):
                        passages = retrieve_passages(
                            question,
                            chunks,
                            document_vectors,
                            embedding_model
                        )

                        answer = generate_answer(
                            question,
                            passages,
                            tokenizer,
                            model
                        )

                        st.markdown(answer)

                        if passages:
                            pages = sorted({
                                p["page"] for p in passages
                            })

                            st.caption(
                                "📚 Relevant PDF pages: "
                                + ", ".join(map(str, pages))
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

    except Exception as error:
        st.error(f"Something went wrong: {error}")
        st.info(
            "Check your internet connection for the first model "
            "download. If the error continues, share the error message."
        )

# ---------------- FOOTER ----------------

st.markdown(
    """
    <div style="
        text-align: center;
        padding: 20px;
        margin-top: 35px;
        color: #3730a3;
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid #dbe4ff;
        border-radius: 15px;
        font-size: 16px;
        font-weight: 600;
    ">
        🤖 BOTZY
        <br>
        Made with 💜 by BUSHRA
        <br>
        <span style="
            font-size: 12px;
            color: #64748b;
            font-weight: normal;
        ">
            Your Industrial AI Assistant
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
