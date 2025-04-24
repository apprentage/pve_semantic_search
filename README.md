# PVE Semantic Search App

A lightweight Streamlit app for semantic search over PVE document embeddings using Qdrant and OpenAI.

## 🔧 Setup

1. Create a `.env` or export these variables in your shell:
   ```bash
   export OPENAI_API_KEY=your-openai-key
   export QDRANT_API_KEY=your-qdrant-key
   export QDRANT_HOST=https://your-qdrant-instance
   ```

2. Install dependencies:
   ```bash
   pip install streamlit qdrant-client openai
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## 💡 Features

- Textbox + prefilled query prompts
- Color-coded result scores
- Vector search with cosine similarity (using OpenAI embeddings)
