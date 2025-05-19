# 🧠 PVE Dual Vector Semantic Search

A lightweight Streamlit app for semantic search over PVE document embeddings using Qdrant and dual embeddings (OpenAI + MiniLM).

---

## 🔧 Setup with `uv`

### 1. Install `uv` (if not already):

```bash
curl -Ls https://astral.sh/uv/install.sh | bash
```

### 2. Set required environment variables (via `.env` or export):

```bash
export OPENAI_API_KEY=your-openai-key
export QDRANT_API_KEY=your-qdrant-key
export QDRANT_HOST=https://your-qdrant-instance
```

Or create a `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY = "your-openai-key"
QDRANT_API_KEY = "your-qdrant-key"
QDRANT_HOST = "https://your-qdrant-instance"
```

### 3. Install Python dependencies:

```bash
uv pip install -r requirements.txt
```

> 💡 Tip: If `streamlit` fails to run due to an old Python link, remove it with:
```bash
sudo rm -f /usr/local/bin/streamlit
```

---

## 🚀 Run the App

```bash
uv run streamlit run app_dual.py
```

Then open:

[http://localhost:8501](http://localhost:8501)

or browse the demo

[https://pve-dual-search.streamlit.app/](https://pve-dual-search.streamlit.app/)

---

## 💡 Features

- ✅ Dual-vector search using:
  - `openai-3-large` (3072-dim)
  - `all_minilm_embeddings` (768-dim mean + max pooled)
- ✅ Color-coded results with score
- ✅ GPT-4 summary of top results
- ✅ Fallback logic if one model fails

---

## 📁 Project Structure

- `app_dual.py` – Main Streamlit UI
- `requirements.txt` – All dependencies
- `README.md` – This file
- `JSONL/` – Embedded documents (input)
- `.streamlit/secrets.toml` – Secure keys (optional)

---

## 🧪 Example Query

> What building codes must be followed for commercial projects?

Try variations and inspect the vector scores + GPT summary!
