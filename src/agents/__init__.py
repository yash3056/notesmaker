"""
AI Notes Maker - Agents Package

This package contains the specialized agents that make up the multi-agent system:
- PlannerAgent: Breaks down topics into structured research plans
- WebSearcherAgent: Searches and retrieves relevant content from the web using Tavily
- SynthesizerAgent: Synthesizes research into comprehensive notes
"""

from .planner import Agent1_Planner
from .web_searcher import Agent3_WebSearcher
from .synthesizer import Agent2_Synthesizer
from .base_agent import BaseAgent

# Provide more intuitive names as aliases
PlannerAgent = Agent1_Planner
WebSearcherAgent = Agent3_WebSearcher
SynthesizerAgent = Agent2_Synthesizer

__all__ = [
    'BaseAgent',
    'Agent1_Planner',
    'Agent2_Synthesizer', 
    'Agent3_WebSearcher',
    'PlannerAgent',
    'WebSearcherAgent',
    'SynthesizerAgent'
]
