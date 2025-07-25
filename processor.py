import os
import re
import datetime
from typing import Dict, Any, List, Optional

import pdfplumber
import traceback
import pytesseract

import numpy as np

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
from PIL import Image
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CourtDocumentProcessor:
    """
    A class for processing court documents from PDFs, extracting key information,
    and storing them in Elasticsearch for search capabilities.
    """
    def __init__(self, 
                elastic_host: str = "localhost", 
                elastic_port: int = 9200,
                ocr_enabled: bool = True,
                model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the court document processor.
        
        Args:
            elastic_host: Elasticsearch host address
            elastic_port: Elasticsearch port number
            ocr_enabled: Whether to use OCR for image-based PDFs
            model_name: Sentence transformer model for semantic embeddings
        """
        self.elastic_client = Elasticsearch(hosts=[f"http://{elastic_host}:{elastic_port}"], basic_auth=("elastic", "R3DeJHuN"), verify_certs=False)
        self.ocr_enabled = ocr_enabled
        
        # Initialize semantic model for embeddings
        if model_name:
            self.semantic_model = SentenceTransformer(model_name)
        else:
            self.semantic_model = None
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF document and extract relevant information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted document information
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        document_info = {
            'file_path': pdf_path,
            'file_name': os.path.basename(pdf_path),
            'processed_date': datetime.datetime.now().isoformat(),
            'title': None,
            'date': None,
            'content': None,
            'embedding': None
        }
        
        # Extract text from PDF
        text_content = self._extract_text_from_pdf(pdf_path)
        document_info['content'] = text_content
        
        # Extract title
        document_info['title'] = self._extract_title(text_content)
        
        # Extract date
        document_info['date'] = self._extract_date(text_content)
        
        # Generate embedding if semantic model is available
        if self.semantic_model and text_content:
            document_info['embedding'] = self.semantic_model.encode(text_content[:5000] if len(text_content) > 5000 else text_content).tolist()
        else:
            document_info['embedding'] = None
        
        return document_info
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from PDF using pdfplumber and OCR if needed.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        full_text = []
        
        try:
            # Try direct text extraction first
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text.append(text)
                    elif self.ocr_enabled:
                        # If no text found and OCR is enabled, use OCR
                        image = page.to_image()
                        image_path = f"temp_page_{page.page_number}.png"
                        image.save(image_path, resolution=300)
                        
                        # Perform OCR
                        ocr_text = pytesseract.image_to_string(Image.open(image_path))
                        full_text.append(ocr_text)
                        
                        # Clean up temp image
                        if os.path.exists(image_path):
                            os.remove(image_path)
        
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
        
        return "\n".join(full_text)
    
    def _extract_title(self, text: str) -> Optional[str]:
        """
        Extract document title from text content.
        
        Args:
            text: Document text content
            
        Returns:
            Extracted title or None if not found
        """
        if not text:
            return None
        
        # Split by lines and take the first few non-empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Patterns for common court document titles
        title_patterns = [
            r"^(?:IN THE )?(?:SUPREME|DISTRICT|FEDERAL|CIRCUIT|APPELLATE|SUPERIOR|HIGH|SESSIONS) COURT",
            r"^(?:KERALA|INDIA) (?:DISTRICT|CIRCUIT|SUPREME|SESSIONS|HIGH) COURT",
            r"^(?:ORDER|JUDGMENT|OPINION|MEMORANDUM|DECISION|RULING|VERDICT)",
            r"^(?:CASE|CIVIL ACTION|CRIMINAL) NO\.",
            r"^(?:MOTION|PETITION|APPLICATION|COMPLAINT|BRIEF|AFFIDAVIT)"
        ]
        
        # Check first 10 lines for title patterns
        for i in range(min(10, len(lines))):
            line = lines[i]
            
            # If line is in all caps or title case, it's likely a title
            if line.isupper() or line.istitle():
                return line
            
            # Check against title patterns
            for pattern in title_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    return line
        
        # If no title found, use the first line as a fallback
        if lines:
            return lines[0]
        
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """
        Extract document date from text content.
        
        Args:
            text: Document text content
            
        Returns:
            Extracted date in ISO format or None if not found
        """
        if not text:
            return None
        
        # Common date patterns in court documents
        date_patterns = [
            # MM/DD/YYYY
            r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',
            # Month Day, Year
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})',
            # Abbreviated Month Day, Year
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\.|\s]+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})',
            # Day Month Year
            r'(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)[,\s]+(\d{4})',
            # YYYY-MM-DD
            r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})'
        ]
        
        # Search for date patterns
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 3:
                        if groups[0].isdigit() and int(groups[0]) > 1000:  # YYYY-MM-DD format
                            year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        elif groups[2].isdigit() and int(groups[2]) > 1000:  # Other formats
                            if groups[0].isdigit():  # MM/DD/YYYY
                                month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                            else:  # Month Day, Year
                                # Map month name to month number
                                month_names = {
                                    'jan': 1, 'january': 1,
                                    'feb': 2, 'february': 2,
                                    'mar': 3, 'march': 3,
                                    'apr': 4, 'april': 4,
                                    'may': 5,
                                    'jun': 6, 'june': 6,
                                    'jul': 7, 'july': 7,
                                    'aug': 8, 'august': 8,
                                    'sep': 9, 'september': 9,
                                    'oct': 10, 'october': 10,
                                    'nov': 11, 'november': 11,
                                    'dec': 12, 'december': 12
                                }
                                month = month_names.get(groups[0].lower(), 1)
                                day = int(groups[1])
                                year = int(groups[2])
                        
                        # Validate date
                        if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                            return datetime.date(year, month, day).isoformat()
                except Exception:
                    continue
        
        return None
    
    def index_document(self, document_info: Dict[str, Any], index_name: str = "court_documents") -> bool:
        """
        Index document information in Elasticsearch.
        
        Args:
            document_info: Document information dictionary
            index_name: Name of the Elasticsearch index
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            # Ensure index exists
            if not self.elastic_client.indices.exists(index=index_name).body:
                # Create index with mappings for text, date and vector fields
                mapping = {
                    "mappings": {
                        "properties": {
                            "file_path": {"type": "keyword"},
                            "file_name": {"type": "keyword"},
                            "title": {"type": "text", "analyzer": "standard"},
                            "date": {"type": "date"},
                            "processed_date": {"type": "date"},
                            "content": {"type": "text", "analyzer": "standard"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": 384,  # Ensure this matches your SentenceTransformer output
                                "index": True,
                                "similarity": "dot_product"  # Ensures correct vector search
                            }# for semantic search
                        }
                    }
                }
                self.elastic_client.indices.create(index=index_name, body=mapping)
            
            # Index the document
            self.elastic_client.index(index=index_name, document=document_info)
            return True
            
        except Exception as e:
            print(f"Error indexing document: {e}")
            return False
    
    def search_documents(self, 
                        query: str = None, 
                        date_from: str = None, 
                        date_to: str = None,
                        semantic_search: bool = False,
                        index_name: str = "court_documents",
                        size: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents using keywords, date range, or semantic search.
        
        Args:
            query: Search query string
            date_from: Start date for filtering (ISO format)
            date_to: End date for filtering (ISO format)
            semantic_search: Whether to use semantic search
            index_name: Name of the Elasticsearch index
            size: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            search_body = {"query": {"bool": {"must": []}}}

            if query:
                if semantic_search and self.semantic_model:
                    query_embedding = self.semantic_model.encode(query).tolist()
                    
                    query_embedding1 = np.array(query_embedding)
                    normed_embedding = (query_embedding1 / np.linalg.norm(query_embedding1)).tolist()

                    print("Normalized Query Embedding:", np.linalg.norm(normed_embedding))  # Should not be close to 0

                    search_body = {
                        "query": {
                            "knn": {
                                "field": "embedding",
                                "query_vector": normed_embedding,
                                "k": size,
                                "num_candidates": 5  # you can tweak this
                            }
                        }
                    }
                    
                    

                else:
                    # Full-text search
                    search_body["query"]["bool"]["must"].append({
                        "multi_match": {
                            "query": query,
                            "fields": ["title^3", "content"]
                        }
                    })

            # Add date range filter if provided
            if date_from or date_to:
                date_range = {"range": {"date": {}}}
                
                if date_from:
                    date_range["range"]["date"]["gte"] = date_from
                    
                if date_to:
                    date_range["range"]["date"]["lte"] = date_to
                    
                search_body["query"]["bool"]["must"].append(date_range)

            # If no conditions provided, search for all documents
            if not query and not date_from and not date_to:
                search_body = {"query": {"match_all": {}}}

            # print("Final Search Body:", search_body)  # Debugging step


            results = self.elastic_client.search(
                index=index_name,
                query=search_body["query"],  # Only pass the query part
                size=size
            )
            # Process results
            documents = []
            for hit in results["hits"]["hits"]:
                doc = hit["_source"]
                doc["score"] = hit["_score"]
                if "embedding" in doc:
                    print("Document Embedding:", np.linalg.norm(doc["embedding"]))  # Should not be close to 0

                documents.append(doc)

            return documents

        except Exception as e:
            print(f"Error searching documents: {e}")
            traceback.print_exc()
            return []

#test for elasticsearch connection
def test_elasticsearch_connection(processor: CourtDocumentProcessor) -> bool:
    """
    Test connection to Elasticsearch.
    Args:

        processor: CourtDocumentProcessor instance
    Returns:
        Boolean indicating success or failure           
    """
    try:
        client = processor.elastic_client
        if client.ping():
            print("Elasticsearch connection successful")
            return True
        else:
            print("Elasticsearch connection failed")
            return False
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
        return False        

# Example usage
def process_directory(directory_path: str, processor: CourtDocumentProcessor) -> None:
    """
    Process all PDF files in a directory and index them.
    
    Args:
        directory_path: Path to directory containing PDF files
        processor: CourtDocumentProcessor instance
    """
    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        print(f"Processing {pdf_path}")
        
        try:
            # Process document
            document_info = processor.process_pdf(pdf_path)
            
            # Index document
            success = processor.index_document(document_info)

            print(document_info)
            
            if success:
                print(f"Successfully indexed document: {document_info['title']}")
            else:
                print(f"Failed to index document: {pdf_path}")
                
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")


if __name__ == "__main__":
    # Example configuration
    processor = CourtDocumentProcessor(
        elastic_host="localhost",
        elastic_port=9200,
        ocr_enabled=True,
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # Small but effective semantic model
    )
    

    test_elasticsearch_connection(processor)
    # Process a directory of PDFs
    # process_directory("/Users/nvj/Dev/Python/document-digitization/pdfs", processor)

    # Example searches
    # Keyword search
    # results = processor.search_documents(query="The learned Magistrate after evaluating the\nevidence held that there is no evidence to connect the accused")
    # print(results[0]['title'])

    # Date range search
    # results = processor.search_documents(date_from="2001-01-01", date_to="2020-12-31")
    # print(len(results))
    # print(results[1]['title'])

    
    # Semantic search
    # results = processor.search_documents(query="rigorous imprisonment", semantic_search=True)
    # print(len(results))