import os
import time
import streamlit as st
import openai
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import re
from dotenv import load_dotenv
load_dotenv()

# Load API keys from env or Streamlit secrets
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
TOP_K = 5
MIN_MINILM_SCORE = 0.5

openai.api_key = OPENAI_API_KEY
client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit config
st.set_page_config(page_title="PVE Dual Vector Search",
                   layout="wide", page_icon="🔍")
st.title("🔍 PVE Dual Vector Search (MiniLM First, OpenAI Fallback)")

if "query" not in st.session_state:
    st.session_state.query = ""

if "query" not in st.session_state:
    st.session_state.query = ""

st.markdown("**Quick prompts:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("Who pays if a project is delayed?"):
        st.session_state.query = "Who pays if a project is delayed?"
    if st.button("Does the architect take responsibility for construction cost overruns?"):
        st.session_state.query = "Does the architect take responsibility for construction cost overruns?"
    if st.button("What civil engineering documents are required for a new development?"):
        st.session_state.query = "What civil engineering documents are required for a new development?"
with col2:
    if st.button("What happens if the client cancels the project?"):
        st.session_state.query = "What happens if the client cancels the project?"
    if st.button("What are some new engineering code changes in 2021?"):
        st.session_state.query = "What are some new engineering code changes in 2021?"
    if st.button("What building codes must be followed for commercial projects?"):
        st.session_state.query = "What building codes must be followed for commercial projects?"

query = st.text_input("Enter a search query:", value=st.session_state.query)


def sent_tokenize(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def get_minilm_vector(text):
    sentences = sent_tokenize(text)
    vecs = minilm_model.encode(sentences)
    if len(vecs.shape) == 1:
        vecs = np.expand_dims(vecs, axis=0)
    return np.concatenate([np.mean(vecs, axis=0), np.max(vecs, axis=0)]).tolist()


def get_openai_vector(text):
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=[text]
    )
    return response.data[0].embedding


def search_qdrant(vector, name):
    t0 = time.time()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=(name, vector),
        limit=TOP_K,
        with_payload=True,
        with_vectors=False
    )
    duration = round(time.time() - t0, 2)
    return results, duration


def summarize_results(results):
    docs = [r.payload.get("rawText", "") for r in results]
    joined = "\\n".join(docs[:3])
    if not joined.strip():
        return "No content found to summarize."

    prompt = f"Summarize the following technical context:\\n\\n{joined}"

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def render_results(results, source):
    for r in results:
        text = r.payload.get("rawText", "")
        score = round(r.score, 4)
        meta = r.payload.get("metadata", "")
        page = r.payload.get("pageNumber", "–")
        ref = r.payload.get("refId", "–")
        st.markdown(
            f"**Score:** `{score}` {'🟢' if score >= 0.6 else '🟡' if score >= 0.4 else '🔴'}  | **Page:** {page}  | **Model:** {source}")
        st.markdown(f"`refId:` {ref}")
        if meta:
            st.markdown(f"`Keywords:` {meta}")
        st.markdown(text[:800] + "..." if len(text) > 800 else text)
        st.markdown("---")


if query:
    with st.spinner("Embedding query..."):
        overall_start = time.time()
        minilm_vec = get_minilm_vector(query)
        openai_vec = get_openai_vector(query)
        embed_duration = round(time.time() - overall_start, 2)

    with st.spinner("Searching vectors..."):
        minilm_results, minilm_time = search_qdrant(
            minilm_vec, "all_minilm_embeddings")
        openai_results, openai_time = search_qdrant(
            openai_vec, "openai-3-large")

        max_minilm_score = max((r.score for r in minilm_results), default=0)
        use_openai = not minilm_results or max_minilm_score < MIN_MINILM_SCORE

        chosen_results = openai_results if use_openai else minilm_results
        source_used = "OpenAI" if use_openai else "MiniLM"

    summary = summarize_results(chosen_results)

    st.success(
        f"✅ Summary using **{source_used}** vectors (Query Embed: {embed_duration}s | MiniLM: {minilm_time}s | OpenAI: {openai_time}s)")
    st.markdown(summary)

    st.subheader("🔹 Top Results from MiniLM Vector")
    render_results(minilm_results, "MiniLM")

    st.subheader("🔸 Top Results from OpenAI Vector")
    render_results(openai_results, "OpenAI")
