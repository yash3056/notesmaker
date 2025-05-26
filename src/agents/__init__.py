"""
AI Notes Maker - Agents Package

This package contains the specialized agents that make up the multi-agent system:
- PlannerAgent: Breaks down topics into structured research plans
- RetrieverAgent: Searches and retrieves relevant content from documents
- SynthesizerAgent: Synthesizes research into comprehensive notes
"""

from .planner import Agent1_Planner
from .retriever import Agent3_Retriever
from .synthesizer import Agent2_Synthesizer
from .base_agent import BaseAgent

# Provide more intuitive names as aliases
PlannerAgent = Agent1_Planner
RetrieverAgent = Agent3_Retriever
SynthesizerAgent = Agent2_Synthesizer

__all__ = [
    'BaseAgent',
    'Agent1_Planner',
    'Agent2_Synthesizer', 
    'Agent3_Retriever',
    'PlannerAgent',
    'RetrieverAgent',
    'SynthesizerAgent'
]
