"""
Core data structures for the AI Multi-Agent Note Taking System
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class Message:
    """Message structure for inter-agent communication"""
    sender: str
    recipient: str
    message_type: str
    content: Dict[str, Any]
    timestamp: float


@dataclass
class SearchQuery:
    """Structure for search queries"""
    query: str
    max_results: int = 5
    context_needed: bool = True


@dataclass
class SearchResult:
    """Structure for search results"""
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class Plan:
    """Structure for Agent 1's plans"""
    objective: str
    steps: List[str]
    search_queries: List[SearchQuery]
    expected_sections: List[str]


@dataclass
class SynthesisConfig:
    """Configuration for synthesis parameters"""
    main_synthesis_tokens: int = 8196
    gap_filling_tokens: int = 8196
    chunked_synthesis_tokens: int = 8196
    final_organization_tokens: int = 8196
    batch_size: int = 5  # Number of sources per batch
    chunked_threshold: int = 10  # Use chunked synthesis when more than this many sources


@dataclass
class AgentStatus:
    """Status information for agents"""
    agent_id: str
    agent_name: str
    is_running: bool
    current_task: Optional[str] = None
    last_activity: Optional[float] = None
    messages_processed: int = 0
    errors_count: int = 0


@dataclass
class SystemMetrics:
    """System-wide metrics and statistics"""
    session_start_time: float
    total_documents: int
    total_searches: int
    total_llm_calls: int
    total_messages: int
    agents_status: List[AgentStatus]
    memory_usage: Optional[Dict[str, Any]] = None
    performance_stats: Optional[Dict[str, Any]] = None
