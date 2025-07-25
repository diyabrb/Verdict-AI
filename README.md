# 🏛️ Court Document Digitizer & Search Engine

A Flask-based web app for uploading, processing, and searching **court PDFs** using OCR, NLP, and **semantic search** with **Elasticsearch**.

- 🧠 Uses `sentence-transformers` for embeddings  
- 🕵️ Extracts metadata like Title, Date, and Full Text  
- 🔍 Supports both keyword and **semantic vector search**  
- 🧾 PDF-to-text via `pdfplumber` + OCR fallback with `pytesseract`  
- ⚡ Powered by Elasticsearch (set up locally via Docker)

---

## ⚙️ Prerequisites

- Python 3.10+
- [Docker](https://www.docker.com/)
- Tesseract OCR (`brew install tesseract` for macOS)
- Elasticsearch & Kibana (set up using Elastic's official script)

---

## 🚀 Quickstart

### 1. Clone this repo

```bash
git clone https://github.com/yourusername/document-digitizer.git
cd document-digitizer
```

### 2. Set up virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Dependencies include:**
> - Flask  
> - pdfplumber  
> - pytesseract  
> - Pillow  
> - elasticsearch  
> - sentence-transformers  
> - numpy  

### 3. Set up Tesseract (if not installed)

```bash
brew install tesseract
```

---

## 🔌 Spin Up Elasticsearch & Kibana (Official Method)

```bash
curl -fsSL https://elastic.co/start-local | sh
```

✅ After successful setup:
- Kibana: http://localhost:5601  
- Elasticsearch API: http://localhost:9200  
- Username: `elastic`  
- Password: `R3DeJHu` *(or the generated one shown after setup)*

The script creates a folder `elastic-start-local/` with:
- `.env`, `docker-compose.yml`
- `start`, `stop`, `uninstall` scripts

To restart services:

```bash
./elastic-start-local/start
```

---

## 📦 Project Structure

```
document-digitizer/
├── app.py                   # Flask app
├── processor.py             # PDF OCR + NLP + Elasticsearch
├── templates/
│   └── index.html           # Basic UI
├── pdfs/                    # Uploaded PDFs
└── elastic-start-local/     # Elastic + Kibana (auto-created)
```

---

## 🧠 How It Works

1. Upload a court PDF via browser or API.
2. Extract text using `pdfplumber`. If that fails, fallback to OCR via Tesseract.
3. Extract:
   - Title (based on text heuristics)
   - Date (regex + natural formats)
   - Full content
4. Generate semantic embedding with `sentence-transformers`
5. Store everything in Elasticsearch
6. Search via keyword OR semantic (vector similarity)

---

## 🛠️ API Endpoints

### `GET /`
Renders the upload form.

### `POST /upload`
Upload a PDF to index.

**Request:**
- `multipart/form-data` with a key `pdf`

**Response:**
```json
{
  "message": "Document indexed successfully",
  "title": "Some Legal Title"
}
```

### `POST /search`
Search documents.

**Request:**
```json
{
  "query": "rigorous imprisonment",
  "semantic": true,
  "date_from": "2010-01-01",
  "date_to": "2023-12-31"
}
```

**Response:**
```json
[
  {
    "title": "Court Judgement on XYZ",
    "date": "2018-07-14",
    "score": 0.923,
    ...
  }
]
```

---

## 🧪 Standalone Indexing

To process all PDFs in the `/pdfs` folder via command line:

```bash
python processor.py
```

To test Elasticsearch connection:

```python
from processor import test_elasticsearch_connection
test_elasticsearch_connection(processor)
```

---

## 🧯 Troubleshooting

- **PDF text empty?**  
  → OCR kicks in if `ocr_enabled=True`

- **Semantic search returns all docs?**  
  → Ensure `embedding` is populated and normalized

- **Elasticsearch 400/500 errors?**  
  → Might be mapping mismatch. Try deleting and reindexing:

```bash
curl -X DELETE http://localhost:9200/court_documents
```

- **Embedding norms close to 0?**  
  → Debug text length and vector normalization

---

## 🙌 Credits

- [Elasticsearch](https://www.elastic.co/)
- [SentenceTransformers](https://www.sbert.net/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

---

## 📃 License

MIT License — feel free to use, remix, and adapt. Just don't give it to your shady lawyer uncle 👨‍⚖️.
