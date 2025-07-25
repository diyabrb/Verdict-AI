from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os

from processor import CourtDocumentProcessor  # Your class file (rename accordingly)

app = Flask(__name__)
UPLOAD_FOLDER = "/Users/nvj/Dev/Python/document-digitization/pdfs"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Init processor
processor = CourtDocumentProcessor(
    elastic_host="localhost",
    elastic_port=9200,
    ocr_enabled=True,
    model_name="all-MiniLM-L6-v2"
)

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400
    
    file = request.files["pdf"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Process and index
    try:
        doc_info = processor.process_pdf(file_path)
        success = processor.index_document(doc_info)
        if success:
            return jsonify({"message": "Document indexed successfully", "title": doc_info["title"]})
        else:
            return jsonify({"error": "Failed to index document"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    date_from = data.get("date_from")
    date_to = data.get("date_to")
    semantic = data.get("semantic", False)

    results = processor.search_documents(
        query=query,
        date_from=date_from,
        date_to=date_to,
        semantic_search=semantic
    )
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
