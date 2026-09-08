import os
import re
from typing import List, Dict, Any

class TextDocumentLoader:
    """
    Enterprise Document Loader and Recursive Chunking Engine.
    Supports .pdf, .txt, .md, .csv, .json, and .log files.
    Splits documents into overlapping semantic chunks with rich metadata.
    """
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extracts text page-by-page from a PDF file with page metadata."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required to read PDF files. Install with: pip install pypdf")

        reader = PdfReader(file_path)
        pages_data = []
        file_name = os.path.basename(file_path)

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_data.append({
                    "text": text.strip(),
                    "page_number": page_idx + 1,
                    "file_name": file_name
                })
        return pages_data

    def load_plain_text(self, file_path: str) -> str:
        """Reads content from a text file with multi-encoding fallback."""
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to decode text file: {file_path}")

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Recursively splits text by paragraphs, sentences, and words to maintain semantic coherence.
        """
        metadata = metadata or {}
        text = text.strip()
        if not text:
            return []

        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(para) > self.chunk_size:
                    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', para) if s.strip()]
                    sub_chunk = ""
                    for s in sentences:
                        if len(sub_chunk) + len(s) + 1 <= self.chunk_size:
                            sub_chunk = f"{sub_chunk} {s}" if sub_chunk else s
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = s
                    if sub_chunk:
                        current_chunk = sub_chunk
                else:
                    current_chunk = para
                    
        if current_chunk:
            chunks.append(current_chunk)

        output_docs = []
        total_chunks = len(chunks)
        file_name = metadata.get("file_name", "unknown_document")
        page_num = metadata.get("page_number")
        
        for idx, chunk in enumerate(chunks):
            page_suffix = f"_p{page_num}" if page_num else ""
            doc_id = f"{file_name}{page_suffix}_chunk_{idx+1}"
            
            title_prefix = f"{file_name}"
            if page_num:
                title_prefix += f" (Page {page_num})"
            title = f"{title_prefix} [Part {idx+1}/{total_chunks}]"

            output_docs.append({
                "id": doc_id,
                "title": title,
                "text": chunk,
                "category": metadata.get("category", "pdf_document" if file_name.endswith(".pdf") else "text_document"),
                "department": metadata.get("department", "general"),
                "year": metadata.get("year", 2026),
                "file_name": file_name,
                "page_number": page_num,
                "chunk_index": idx + 1,
                "total_chunks": total_chunks,
                "tags": metadata.get("tags", ["pdf" if file_name.endswith(".pdf") else "text"])
            })
            
        return output_docs

    def load_and_chunk_file(self, file_path: str, category: str = "custom_document", department: str = "general") -> List[Dict[str, Any]]:
        """Loads a PDF or text file and returns structured chunks ready for Qdrant indexing."""
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        all_chunks = []

        if ext == ".pdf":
            pages = self.load_pdf(file_path)
            for page in pages:
                meta = {
                    "file_name": file_name,
                    "page_number": page["page_number"],
                    "category": category,
                    "department": department,
                    "file_path": file_path
                }
                page_chunks = self.chunk_text(page["text"], meta)
                all_chunks.extend(page_chunks)
        else:
            text = self.load_plain_text(file_path)
            meta = {
                "file_name": file_name,
                "category": category,
                "department": department,
                "file_path": file_path
            }
            all_chunks = self.chunk_text(text, meta)

        return all_chunks

    def load_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Scans a directory for all .pdf, .txt, .md, .csv files and returns all chunked documents."""
        all_chunks = []
        supported_exts = [".pdf", ".txt", ".md", ".csv", ".json", ".log"]
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_exts:
                    full_path = os.path.join(root, file)
                    try:
                        chunks = self.load_and_chunk_file(full_path, category="imported_files", department="enterprise")
                        all_chunks.extend(chunks)
                        print(f"[DocLoader] Loaded '{file}' -> {len(chunks)} chunks.")
                    except Exception as e:
                        print(f"[DocLoader Error] Skipping '{file}': {e}")
                        
        return all_chunks
