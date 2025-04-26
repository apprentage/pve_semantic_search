# ✅ FINAL app_dual.py (MiniLM 768d first, fallback to OpenAI 3072d, GPT-4o summarize)

import streamlit as st
import os
import re
import time
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
from openai import OpenAI

st.set_page_config(page_title="PVE Dual Vector Search", layout="wide")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- SECRETS ---


def get_secret(key):
    return os.getenv(key) or (st.secrets.get(key, "") if hasattr(st, 'secrets') else "")


QDRANT_API_KEY = get_secret("QDRANT_API_KEY")
QDRANT_HOST = re.sub(r"^https?://", "", get_secret("QDRANT_HOST"))
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# COLLECTION = "pve_documents"
COLLECTION = "pve_resources_cleanrgx"
GROUP_ID = "26is7eICcEiiiLVb6TK7tw"

# --- INIT CLIENTS ---
client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=f"https://{QDRANT_HOST}", api_key=QDRANT_API_KEY)

# --- LOAD LOCAL MiniLM (768 dims after pooling) ---


@st.cache_resource(show_spinner="Loading MiniLM model...")
def load_minilm():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


minilm_model = load_minilm()

# --- EMBEDDING HELPERS ---


def safe_unicode(text):
    try:
        return text.encode('utf-16', 'surrogatepass').decode('utf-16')
    except Exception:
        return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')


def simple_sent_tokenize(text):
    """Split text into sentences using simple regex (no nltk)."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s for s in sentences if s.strip()]


def embed_minilm(text):
    sentences = simple_sent_tokenize(text)
    sentence_vectors = minilm_model.encode(
        sentences, normalize_embeddings=True)
    mean_vec = np.mean(sentence_vectors, axis=0)
    max_vec = np.max(sentence_vectors, axis=0)
    return np.concatenate([mean_vec, max_vec]).tolist()


def embed_openai(text):
    try:
        response = client.embeddings.create(
            input=text, model="text-embedding-3-large")
        return response.data[0].embedding
    except Exception as e:
        st.error(f"OpenAI embedding failed: {e}")
        return []

# --- SEARCH QDRANT ---


def search_qdrant(vector, name, limit=5):
    try:
        return qdrant.search(
            collection_name=COLLECTION,
            query_vector={"name": name, "vector": vector},
            limit=limit,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=512, exact=False),
            query_filter=None
            # query_filter=Filter(must=[FieldCondition(
            #    key="group_id", match=MatchValue(value=GROUP_ID))]),
        )
    except Exception as e:
        st.error(f"Search error ({name}): {e}")
        return []


# --- STREAMLIT UI ---
st.title("🔎 PVE Dual Vector Search (MiniLM First, OpenAI Fallback)")

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

if st.button("Search") and query:
    with st.spinner("Embedding and searching..."):
        try:
            start = time.time()

            # --- Step 1: Try MiniLM ---
            minilm_vector = embed_minilm(query)
            minilm_results = search_qdrant(
                minilm_vector, "all_minilm_embeddings", limit=4)

            if minilm_results:
                combined = [
                    {"score": r.score, "text": r.payload.get("text", "")[
                        :1000]}
                    for r in minilm_results
                ]
                used_model = "MiniLM"
            else:
                # --- Step 2: Fallback to OpenAI ---
                openai_vector = embed_openai(query)
                openai_results = search_qdrant(
                    openai_vector, "openai-3-large", limit=4)

                combined = [
                    {"score": r.score, "text": r.payload.get("text", "")[
                        :1000]}
                    for r in openai_results
                ]
                used_model = "OpenAI" if openai_results else "None"

            total_time = time.time() - start

            if combined:
                combined = sorted(
                    combined, key=lambda x: x["score"], reverse=True)
                context = "\n\n".join([
                    f"Score: {c['score']:.4f}\nText: {c['text']}" for c in combined
                ])

                # --- Summarize ---
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant summarizing document excerpts."},
                        {"role": "user", "content": f"User asked: '{query}'\n\n{context}"}
                    ],
                    temperature=0.1
                )

                st.success(
                    f"Summary using {used_model} vectors (completed in {total_time:.2f}s):")
                st.markdown(response.choices[0].message.content)
                st.markdown("---")

                for item in combined:
                    badge = "🔴" if item["score"] < 0.3 else "🟡" if item["score"] < 0.5 else "🟢"
                    st.markdown(f"**Score**: {item['score']:.4f} {badge}")
                    st.markdown(f"**Text**: {safe_unicode(item['text'])}...")
                    st.markdown("---")

            else:
                st.warning("❌ No results found.")

        except Exception as e:
            st.error(f"Unhandled exception: {e}")
