"""
Tests for the DocumentLoader class
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.core.document_loader import DocumentLoader


class TestDocumentLoader:
    """Test cases for DocumentLoader"""
    
    def test_get_supported_extensions(self):
        """Test getting supported file extensions"""
        extensions = DocumentLoader.get_supported_extensions()
        assert '.txt' in extensions
        assert '.md' in extensions
    
    def test_load_text_file(self):
        """Test loading a simple text file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.")
            temp_path = f.name
        
        try:
            content = DocumentLoader.load_text_file(temp_path)
            assert "This is a test document." in content
            assert "multiple lines" in content
        finally:
            os.unlink(temp_path)
    
    def test_split_document(self):
        """Test document splitting functionality"""
        content = "This is a long document. " * 100  # Create long content
        chunks = DocumentLoader.split_document(content, "test.txt", text_chunk_size=200)
        
        assert len(chunks) > 1
        assert all("Source: test.txt" in chunk for chunk in chunks)
    
    def test_validate_folder(self):
        """Test folder validation"""
        # Test non-existent folder
        assert not DocumentLoader.validate_folder("/non/existent/path")
        
        # Test empty folder
        with tempfile.TemporaryDirectory() as temp_dir:
            assert not DocumentLoader.validate_folder(temp_dir)
            
            # Add a text file
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("Test content")
            assert DocumentLoader.validate_folder(temp_dir)
    
    def test_load_documents_from_folder(self):
        """Test loading documents from a folder"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            (Path(temp_dir) / "test1.txt").write_text("Content of file 1")
            (Path(temp_dir) / "test2.md").write_text("# Content of file 2")
            
            documents = DocumentLoader.load_documents_from_folder(temp_dir)
            
            assert len(documents) >= 2
            assert any("Content of file 1" in doc for doc in documents)
            assert any("Content of file 2" in doc for doc in documents)
