"""
Core module for the AI Multi-Agent Note Taking System
Contains the main data structures, document loading, LLM wrappers, and system orchestration.
"""

from .data_structures import Message, Plan, SearchQuery, SearchResult, SynthesisConfig
from .document_loader import DocumentLoader
from .llm_wrapper import BaseLLM, QwenLLM

__all__ = [
    "Message",
    "Plan", 
    "SearchQuery",
    "SearchResult",
    "SynthesisConfig",
    "DocumentLoader",
    "BaseLLM",
    "QwenLLM"
]
