import streamlit as st
import os
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
from openai import OpenAI

# --- CONFIGURATION ---


def get_secret(key):
    return os.getenv(key) or st.secrets.get(key, "")


QDRANT_API_KEY = get_secret("QDRANT_API_KEY")
QDRANT_HOST = re.sub(r"^https?://", "", get_secret("QDRANT_HOST"))
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
COLLECTION = "pve_documents"
GROUP_ID = "26is7eICcEiiiLVb6TK7tw"

# --- INITIALIZE OPENAI CLIENT ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- EMBEDDING FUNCTION ---


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding


# --- CONNECT TO QDRANT ---
qdrant = QdrantClient(url=f"https://{QDRANT_HOST}", api_key=QDRANT_API_KEY)

# --- STREAMLIT UI ---
st.set_page_config(page_title="PVE Semantic Search", layout="wide")
st.title("🏗️ PVE Semantic Document Search")

# Initialize session state for query
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
with col2:
    if st.button("What happens if the client cancels the project?"):
        st.session_state.query = "What happens if the client cancels the project?"
    if st.button("Hazardous materials handling policy"):
        st.session_state.query = "Hazardous materials handling policy"

# Search input
query = st.text_input("Or enter your own query:", value=st.session_state.query)

if st.button("Search") and query:
    st.info("Embedding query and searching...")
    try:
        query_vector = embed_text(query)

        # Prepare filter
        group_filter = Filter(must=[
            FieldCondition(key="group_id", match=MatchValue(value=GROUP_ID))
        ])

        # Perform vector search
        results = qdrant.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=10,
            search_params=SearchParams(hnsw_ef=64, exact=False),
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
                [f"Score: {s['score']:.4f}\nText: {s['text']}" for s in summaries])
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
                st.markdown(f"**Text**: {s['text']}...")
                st.markdown("---")
        else:
            st.warning(
                "No results found.\n\n❗️ There are no specific details available regarding that topic within the documents.")
    except Exception as e:
        st.error(f"Search failed: {e}")
