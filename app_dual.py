# ✅ app_dual.py (Updated for Qdrant dual vector search)

import streamlit as st
import os
import re
import time
import warnings
from sentence_transformers.util import normalize_embeddings
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
COLLECTION = "pve_documents_384_3072"
GROUP_ID = "26is7eICcEiiiLVb6TK7tw"

# --- INIT CLIENTS ---
client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=f"https://{QDRANT_HOST}", api_key=QDRANT_API_KEY)

# --- LOAD LOCAL MODEL (384 dims) ---


@st.cache_resource(show_spinner="Loading MiniLM...")
def load_minilm():
    return SentenceTransformer("thenlper/gte-small")


minilm_model = load_minilm()

# --- EMBEDDERS ---


def embed_text_openai(text):
    try:
        return client.embeddings.create(input=text, model="text-embedding-3-large").data[0].embedding
    except Exception as e:
        st.error(f"OpenAI embedding failed: {e}")
        return []


def embed_text_minilm(text):
    try:
        return minilm_model.encode(text, normalize_embeddings=True).tolist()
    except Exception as e:
        st.error(f"MiniLM embedding failed: {e}")
        return []

# --- SEARCH QDRANT ---


def search_vector(vector, name):
    try:
        return qdrant.search(
            collection_name=COLLECTION,
            query_vector={"name": name, "vector": vector},
            limit=4,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=64, exact=False),
            query_filter=Filter(must=[
                FieldCondition(
                    key="group_id", match=MatchValue(value=GROUP_ID))
            ])
        )
    except Exception as e:
        st.error(f"Search error ({name}): {e}")
        return []


# --- UI ---
st.title("🔎 PVE Semantic Document Search (Dual Vector)")
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
    if st.button("Hazardous materials handling policy"):
        st.session_state.query = "Hazardous materials handling policy"
    if st.button("What building codes must be followed for commercial projects?"):
        st.session_state.query = "What building codes must be followed for commercial projects?"

query = st.text_input("Enter a search query:", value=st.session_state.query)

if st.button("Search") and query:
    with st.spinner("Embedding and searching..."):
        try:
            start = time.time()
            vector_openai = embed_text_openai(query)
            vector_minilm = embed_text_minilm(query)

            openai_results = search_vector(vector_openai, "openai-3-large")
            minilm_results = search_vector(
                vector_minilm, "all_minilm_embeddings")

            combined = [
                {"score": p.score, "text": p.payload.get("rawText", "")[:1000]}
                for p in (openai_results + minilm_results)
            ]

            combined = sorted(combined, key=lambda x: x["score"], reverse=True)
            elapsed = time.time() - start

            if combined:
                context = "\n\n".join(
                    [f"Score: {c['score']:.4f}\nText: {c['text']}" for c in combined])
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant summarizing document excerpts."},
                        {"role": "user", "content": f"User asked: '{query}'\n\n{context}"}
                    ],
                    temperature=0.2
                )
                st.success(f"Summary (in {elapsed:.2f}s):")
                st.markdown(response.choices[0].message.content)
                st.markdown("---")

                for c in combined:
                    st.markdown(f"**Score**: {c['score']:.4f}")
                    st.markdown(f"**Text**: {c['text']}")
                    st.markdown("---")
            else:
                st.warning(
                    "❌ No results found. Check the embedding quality or query text.")

        except Exception as e:
            st.error(f"Unhandled exception: {e}")
