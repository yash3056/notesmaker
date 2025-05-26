"""
Document loader for the AI Multi-Agent Note Taking System
Handles loading documents from various file formats and splitting them into chunks.
"""

import os
import glob
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Document readers for different file types
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class DocumentLoader:
    """Handles loading documents from various file formats"""
    
    @staticmethod
    def load_documents_from_folder(folder_path: str = "documents") -> List[str]:
        """Load all documents from the specified folder"""
        if not os.path.exists(folder_path):
            logger.warning(f"Documents folder '{folder_path}' not found. Creating it...")
            os.makedirs(folder_path)
            logger.info(f"Created folder '{folder_path}'. Please add your documents there.")
            return []
        
        documents = []
        supported_extensions = ['.txt', '.md', '.pdf', '.docx']
        
        for ext in supported_extensions:
            pattern = os.path.join(folder_path, f"**/*{ext}")
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                logger.info(f"Loading document: {file_path}")
                try:
                    content = DocumentLoader.load_single_document(file_path)
                    if content.strip():
                        # Split large documents into chunks
                        chunks = DocumentLoader.split_document(content, file_path)
                        documents.extend(chunks)
                        logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
                    else:
                        logger.warning(f"Empty content in {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Total document chunks loaded: {len(documents)}")
        return documents
    
    @staticmethod
    def load_single_document(file_path: str) -> str:
        """Load a single document based on its file extension"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.txt', '.md']:
            return DocumentLoader.load_text_file(file_path)
        elif file_ext == '.pdf':
            return DocumentLoader.load_pdf_file(file_path)
        elif file_ext == '.docx':
            return DocumentLoader.load_docx_file(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return ""
    
    @staticmethod
    def load_text_file(file_path: str) -> str:
        """Load plain text or markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    @staticmethod
    def load_pdf_file(file_path: str) -> str:
        """Load PDF file content"""
        if not PDF_AVAILABLE:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            return ""
        
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
        
        return text
    
    @staticmethod
    def load_docx_file(file_path: str) -> str:
        """Load DOCX file content"""
        if not DOCX_AVAILABLE:
            logger.error("python-docx not installed. Install with: pip install python-docx")
            return ""
        
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    @staticmethod
    def split_document(content: str, file_path: str, text_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split document content into overlapping chunks"""
        if len(content) <= text_chunk_size:
            return [f"Source: {file_path}\n\n{content}"]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + text_chunk_size, len(content))
            chunk = content[start:end]
            
            # Try to break at sentence boundary if not at the end
            if end < len(content):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + text_chunk_size // 2:
                    end = start + break_point + 1
                    chunk = content[start:end]
            
            chunks.append(f"Source: {file_path}\n\n{chunk.strip()}")
            
            # Move start forward, but ensure we make progress
            next_start = end - overlap
            if next_start <= start:  # Prevent infinite loops
                next_start = start + max(1, text_chunk_size // 2)
            start = next_start
            
            if start >= len(content):
                break
        
        return chunks

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Get list of supported file extensions"""
        extensions = ['.txt', '.md']
        if PDF_AVAILABLE:
            extensions.append('.pdf')
        if DOCX_AVAILABLE:
            extensions.append('.docx')
        return extensions
    
    @staticmethod
    def validate_folder(folder_path: str) -> bool:
        """Validate if the folder exists and contains supported documents"""
        if not os.path.exists(folder_path):
            return False
        
        supported_extensions = DocumentLoader.get_supported_extensions()
        for ext in supported_extensions:
            pattern = os.path.join(folder_path, f"**/*{ext}")
            if glob.glob(pattern, recursive=True):
                return True
        return False
