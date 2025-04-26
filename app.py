import streamlit as st
import os
import re
import warnings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
from openai import OpenAI

# Disable warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- STREAMLIT CHECK ---
IS_STREAMLIT = True

# --- LOGGING HELPERS ---


def log(msg):
    if IS_STREAMLIT:
        st.write(msg)
    else:
        print(msg)


def log_error(msg):
    if IS_STREAMLIT:
        st.error(msg)
    else:
        print("ERROR:", msg)

# --- CONFIGURATION ---


def get_secret(key):
    return os.getenv(key) or (st.secrets.get(key, "") if hasattr(st, 'secrets') else "")


QDRANT_API_KEY = get_secret("QDRANT_API_KEY")
QDRANT_HOST = re.sub(r"^https?://", "", get_secret("QDRANT_HOST"))
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
COLLECTION = "pve_documents"
GROUP_ID = "26is7eICcEiiiLVb6TK7tw"

# --- INITIALIZE OPENAI CLIENT ---
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    log_error(f"Failed to initialize OpenAI client: {e}")
    client = None

# --- CONNECT TO QDRANT ---
try:
    qdrant = QdrantClient(url=f"https://{QDRANT_HOST}", api_key=QDRANT_API_KEY)
except Exception as e:
    log_error(f"Failed to connect to Qdrant: {e}")
    qdrant = None

# ---  FUNCTIONS ---


def safe_unicode(text):
    try:
        return text.encode('utf-16', 'surrogatepass').decode('utf-16')
    except Exception:
        return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')


def embed_openai(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding


# --- STREAMLIT UI ---
st.set_page_config(page_title="PVE Semantic Search", layout="wide")
st.title("🏗️ PVE Semantic Document Search")

if "query" not in st.session_state:
    st.session_state.query = ""

# Predefined prompt buttons
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

query = st.text_input("Or enter your own query:", value=st.session_state.query)

if st.button("Search") and query:
    if not os.path.exists("requirements.txt"):
        st.warning(
            "requirements.txt not found. Please ensure all dependencies are listed for deployment.")
    with st.spinner("Embedding query and searching..."):
        try:
            if not client or not qdrant:
                raise RuntimeError("API clients not initialized.")

            openai_vector = embed_openai(query)

            group_filter = Filter(must=[
                FieldCondition(
                    key="group_id", match=MatchValue(value=GROUP_ID)
                )
            ])

            results = qdrant.search(
                collection_name=COLLECTION,
                query_vector={
                    "name": "openai-3-large",
                    "vector": openai_vector
                },
                limit=10,
                search_params=SearchParams(hnsw_ef=512, exact=False),
                with_payload=True,
                query_filter=group_filter
            )

            summaries = []
            for point in results:
                score = point.score
                text = point.payload.get("rawText", "")[:1000]
                summaries.append({"score": score, "text": text})

            if summaries:
                context = "\n\n".join(
                    [f"Score: {s['score']:.4f}\nText: {s['text']}" for s in summaries]
                )
                system_msg = "You are a helpful assistant summarizing document excerpts. Provide a 1-2 sentence summary answering the user's question based only on the provided content."
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

                st.success("Summary:")
                st.markdown(summary)
                st.markdown("---")

                for s in summaries:
                    badge = "🔴" if s["score"] < 0.3 else "🟡" if s["score"] < 0.5 else "🟢"
                    st.markdown(f"**Score**: {s['score']:.4f} {badge}")
                    st.markdown(f"**Text**: {safe_unicode(s['text'])}...")
                    st.markdown("---")
            else:
                st.warning(
                    "No results found.\n\n❗️ There are no specific details available regarding that topic within the documents.")
        except Exception as e:
            st.error(f"Search failed: {e}")
