"""
Tests for utility helper functions
"""

import pytest
import tempfile
import json
from pathlib import Path

from src.utils.helpers import (
    ensure_directory_exists,
    safe_filename,
    truncate_text,
    save_json_safely,
    load_json_safely,
    validate_topic,
    extract_keywords,
    Timer
)


class TestHelpers:
    """Test cases for utility helpers"""
    
    def test_ensure_directory_exists(self):
        """Test directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "new" / "nested" / "directory"
            assert ensure_directory_exists(test_path)
            assert test_path.exists()
    
    def test_safe_filename(self):
        """Test filename sanitization"""
        assert safe_filename("normal_file.txt") == "normal_file.txt"
        assert safe_filename("file with spaces.txt") == "file_with_spaces.txt"
        assert safe_filename("file/with\\bad:chars.txt") == "file_with_bad_chars.txt"
        
        # Test length limitation
        long_name = "a" * 300
        safe_name = safe_filename(long_name)
        assert len(safe_name) <= 255
    
    def test_truncate_text(self):
        """Test text truncation"""
        text = "This is a long text that should be truncated"
        truncated = truncate_text(text, max_length=20)
        assert len(truncated) <= 20
        assert truncated.endswith("...")
        
        # Test short text
        short_text = "Short"
        assert truncate_text(short_text, max_length=20) == short_text
    
    def test_save_and_load_json(self):
        """Test JSON save and load operations"""
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test save
            assert save_json_safely(test_data, temp_path)
            
            # Test load
            loaded_data = load_json_safely(temp_path)
            assert loaded_data == test_data
            
            # Test load non-existent file
            assert load_json_safely("/non/existent/file.json") is None
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validate_topic(self):
        """Test topic validation"""
        assert validate_topic("Machine Learning")
        assert validate_topic("AI and Neural Networks")
        assert not validate_topic("")
        assert not validate_topic("   ")
        assert not validate_topic("ab")  # Too short
        assert not validate_topic("!!!!")  # No alphanumeric
        assert not validate_topic(None)
    
    def test_extract_keywords(self):
        """Test keyword extraction"""
        text = "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models."
        keywords = extract_keywords(text, max_keywords=5)
        
        assert len(keywords) <= 5
        assert "machine" in keywords
        assert "learning" in keywords
        assert "artificial" in keywords
        # Stop words should not be included
        assert "the" not in keywords
        assert "and" not in keywords
    
    def test_timer(self):
        """Test Timer context manager"""
        import time
        
        with Timer("Test operation") as timer:
            time.sleep(0.1)  # Sleep for 100ms
        
        assert timer.elapsed >= 0.1
        assert timer.elapsed < 0.2  # Should be close to 0.1
