"""
Utility helper functions for the AI Multi-Agent Note Taking System
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import hashlib


logger = logging.getLogger(__name__)


def ensure_directory_exists(directory_path: Union[str, Path]) -> bool:
    """Ensure a directory exists, create if it doesn't"""
    try:
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        return False


def safe_filename(filename: str, max_length: int = 255) -> str:
    """Create a safe filename by removing/replacing problematic characters"""
    # Replace problematic characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.() "
    safe_name = "".join(c if c in safe_chars else "_" for c in filename)
    
    # Remove extra spaces and underscores
    safe_name = "_".join(safe_name.split())
    
    # Truncate if too long
    if len(safe_name) > max_length:
        # Keep the extension if present
        if "." in safe_name:
            name, ext = safe_name.rsplit(".", 1)
            max_name_length = max_length - len(ext) - 1
            safe_name = name[:max_name_length] + "." + ext
        else:
            safe_name = safe_name[:max_length]
    
    return safe_name


def calculate_file_hash(file_path: Union[str, Path]) -> Optional[str]:
    """Calculate SHA256 hash of a file"""
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def save_json_safely(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> bool:
    """Save JSON data with error handling and backup"""
    try:
        file_path = Path(file_path)
        
        # Create backup if file exists
        if file_path.exists():
            backup_path = file_path.with_suffix(f".bak_{int(time.time())}")
            file_path.rename(backup_path)
        
        # Ensure directory exists
        ensure_directory_exists(file_path.parent)
        
        # Save new data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def load_json_safely(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load JSON data with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.debug(f"JSON file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def merge_dicts_recursive(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_recursive(result[key], value)
        else:
            result[key] = value
    
    return result


def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage information"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent()
        }
    except ImportError:
        logger.debug("psutil not available for memory monitoring")
        return {}
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}")
        return {}


def validate_topic(topic: str) -> bool:
    """Validate if a topic string is suitable for note generation"""
    if not topic or not isinstance(topic, str):
        return False
    
    # Check length
    if len(topic.strip()) < 3:
        return False
    
    # Check for only whitespace/punctuation
    if not any(c.isalnum() for c in topic):
        return False
    
    return True


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text (simple implementation)"""
    import re
    
    # Convert to lowercase and extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'can', 'this', 'that', 'these', 'those', 'what', 'when',
        'where', 'who', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very'
    }
    
    # Filter out stop words and count occurrences
    word_count = {}
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_count[word] = word_count.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:max_keywords]]


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation", logger_func=None):
        self.name = name
        self.logger_func = logger_func or logger.info
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.logger_func(f"{self.name} completed in {format_duration(duration)}")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
