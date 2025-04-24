import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
import openai
import os
import re

# --- CONFIG ---
raw_host = os.getenv(
    "QDRANT_HOST") or "2395df23-344c-4086-b107-fda95c30fce6.us-east-1-0.aws.cloud.qdrant.io"
QDRANT_HOST = re.sub(r"^https?://", "", raw_host)  # Remove scheme if present
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "pve_documents"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROUP_ID = "26is7eICcEiiiLVb6TK7tw"

# --- SETUP OPENAI KEY ---
openai.api_key = OPENAI_API_KEY

# --- EMBED TEXT ---


def embed_text(text: str) -> list[float]:
    response = openai.Embedding.create(
        input=text, model="text-embedding-3-large")
    return response["data"][0]["embedding"]


# --- CONNECT TO QDRANT ---
qdrant = QdrantClient(url=f"https://{QDRANT_HOST}", api_key=QDRANT_API_KEY)

# --- UI ---
st.title("🏗️ PVE Semantic Document Search")
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

query = st.text_input("Or enter your own query:",
                      value=st.session_state.get("query", ""))

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
            summary = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2
            )["choices"][0]["message"]["content"]

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
