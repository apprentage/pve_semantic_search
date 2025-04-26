import streamlit as st
import os
import re
import time
import warnings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# --- SET PAGE CONFIG IMMEDIATELY ---
st.set_page_config(page_title="PVE Dual Vector Search", layout="wide")

# --- UTILITY LOGGERS ---


def log(msg):
    st.write(msg)


def log_error(msg):
    st.error(msg)

# --- SECRETS MANAGEMENT ---


def get_secret(key):
    return os.getenv(key) or (st.secrets.get(key, "") if hasattr(st, 'secrets') else "")


QDRANT_API_KEY = get_secret("QDRANT_API_KEY")
QDRANT_HOST = re.sub(r"^https?://", "", get_secret("QDRANT_HOST"))
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
COLLECTION = "pve_documents_384_3072"
GROUP_ID = "26is7eICcEiiiLVb6TK7tw"

# --- CLIENTS ---
client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=f"https://{QDRANT_HOST}", api_key=QDRANT_API_KEY)

# --- MINILM MODEL ---


@st.cache_resource(show_spinner="Loading MiniLM model...")
def load_minilm_model():
    return SentenceTransformer("thenlper/gte-small")


minilm_model = load_minilm_model()

# --- EMBEDDING FUNCTIONS ---


def embed_text_openai(text: str) -> list[float]:
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"OpenAI embedding failed: {e}")
        return []


def embed_text_minilm(text: str) -> list[float]:
    try:
        return minilm_model.encode(text, normalize_embeddings=True).tolist()
    except Exception as e:
        st.error(f"MiniLM embedding failed: {e}")
        return []

# --- SEARCH FUNCTION ---


def search_vector(query_vector, vector_name, top_k=5):
    return qdrant.search(
        collection_name=COLLECTION,
        query_vector={"name": vector_name, "vector": query_vector},
        limit=top_k,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=64, exact=False),
        query_filter=Filter(must=[
            FieldCondition(key="group_id", match=MatchValue(value=GROUP_ID))
        ])
    )


# --- UI ---
st.title("🔎 PVE Semantic Document Search (Dual Vector)")

if "query" not in st.session_state:
    st.session_state.query = ""

# --- Predefined Quick Prompts ---
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
    with st.spinner("Embedding query and searching..."):
        try:
            start_time = time.time()

            query_vector_openai = embed_text_openai(query)
            log(f"OpenAI vector dims: {len(query_vector_openai)}")
            query_vector_minilm = embed_text_minilm(query)
            log(f"MiniLM vector dims: {len(query_vector_minilm)}")

            openai_results = search_vector(
                query_vector_openai, "openai-3-large", top_k=3)
            minilm_results = search_vector(
                query_vector_minilm, "all_minilm_embeddings", top_k=3)

            total_time = time.time() - start_time

            combined = []
            for point in openai_results + minilm_results:
                combined.append({
                    "score": point.score,
                    "text": point.payload.get("rawText", "")[:1000],
                })

            combined = sorted(combined, key=lambda x: x["score"], reverse=True)

            if combined:
                context = "\n\n".join(
                    [f"Score: {c['score']:.4f}\nText: {c['text']}" for c in combined]
                )
                system_msg = (
                    "You are a helpful assistant summarizing document excerpts. "
                    "Provide a 1-2 sentence summary answering the user's question based only on the provided content."
                )
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"The user asked: '{query}'. Here is the context:\n\n{context}"}
                ]

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.2
                )
                summary = response.choices[0].message.content

                st.success(f"Summary (completed in {total_time:.2f} seconds):")
                st.markdown(summary)
                st.markdown("---")

                for item in combined:
                    badge = "🔴" if item["score"] < 0.3 else "🟡" if item["score"] < 0.5 else "🟢"
                    st.markdown(f"**Score**: {item['score']:.4f} {badge}")
                    st.markdown(f"**Text**: {item['text']}...")
                    st.markdown("---")
            else:
                st.warning("No results found matching the query.")

        except Exception as e:
            st.error(f"Search error: {e}")
