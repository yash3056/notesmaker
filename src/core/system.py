"""
Main system orchestration for the AI Multi-Agent Note Taking System
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from ..agents.planner import Agent1_Planner
from ..agents.synthesizer import Agent2_Synthesizer
from ..agents.web_searcher import Agent3_WebSearcher
from .llm_wrapper import QwenLLM
from .data_structures import Message, SystemMetrics, AgentStatus
from .document_loader import DocumentLoader
from ..utils.recorder import Record, get_global_recorder, set_global_recorder

logger = logging.getLogger(__name__)


class MultiAgentSystem:
    """Main system orchestrating all agents"""
    
    def __init__(self, documents_folder: str = "documents", model_name: str = "Qwen/Qwen3-1.7B", config=None, tavily_api_key: str = None):
        self.config = config
        self.session_start_time = asyncio.get_event_loop().time()
        
        # Note: Documents folder is kept for compatibility but web search is now primary source
        logger.info(f"System starting with web search capabilities")
        
        # Initialize shared LLM
        self.llm = QwenLLM(model_name)
        
        # Initialize agents with shared LLM
        self.agent1 = Agent1_Planner(self.llm)
        self.agent2 = Agent2_Synthesizer(self.llm, config.agents if config else None)
        self.agent3 = Agent3_WebSearcher(tavily_api_key)
        
        self.agents = {
            "agent_1": self.agent1,
            "agent_2": self.agent2,
            "agent_3": self.agent3
        }
        
        # Set the message router for each agent
        for agent in self.agents.values():
            agent.message_router = self.route_message
        
        # Initialize recorder and set it globally
        self.recorder = Record(truncate_content=False)
        self.recorder.enable_full_logging()
        set_global_recorder(self.recorder)
        
        for agent in self.agents.values():
            agent.set_recorder(self.recorder)
        
        self.message_broker = asyncio.Queue()
        self.is_running = False
        self.notes_completed = None
        
        # System metrics
        self.metrics = SystemMetrics(
            session_start_time=self.session_start_time,
            total_documents=0,  # No longer using documents
            total_searches=0,
            total_llm_calls=0,
            total_messages=0,
            agents_status=[]
        )
    
    def add_documents_from_folder(self, folder_path: str):
        """Add more documents from another folder - deprecated for web search version"""
        logger.warning("Document loading is deprecated in web search version")
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about web search capabilities"""
        return {
            "web_search_enabled": self.agent3.client is not None,
            "tavily_api_configured": bool(self.agent3.api_key),
            "content_storage_path": str(self.agent3.content_dir),
            "search_capability": "Web Search via Tavily API"
        }
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get comprehensive system metrics"""
        # Update agent statuses
        agent_statuses = []
        for agent_id, agent in self.agents.items():
            status = AgentStatus(
                agent_id=agent_id,
                agent_name=agent.agent_name,
                is_running=getattr(agent, 'is_running', False),
                current_task=getattr(agent, 'current_task', None),
                last_activity=getattr(agent, 'last_activity', None),
                messages_processed=getattr(agent, 'messages_processed', 0),
                errors_count=getattr(agent, 'errors_count', 0)
            )
            agent_statuses.append(status)
        
        self.metrics.agents_status = agent_statuses
        return self.metrics
    
    async def route_message(self, message: Message):
        """Route messages between agents"""
        logger.info(f"Routing message from {message.sender} to {message.recipient}: {message.message_type}")
        self.metrics.total_messages += 1
        
        if message.recipient in self.agents:
            await self.agents[message.recipient].receive_message(message)
        elif message.recipient == "system":
            # Handle system messages (final output)
            await self.handle_system_message(message)
        else:
            logger.warning(f"Unknown recipient: {message.recipient}")
    
    async def handle_system_message(self, message: Message):
        """Handle system-level messages"""
        if message.message_type == "notes_complete":
            final_notes = message.content.get("final_notes", "")
            
            # Print the final notes
            print("\n" + "="*50)
            print("FINAL NOTES:")
            print("="*50)
            print(final_notes)
            print("="*50)
            
            # Set the result for the future to signal completion
            if hasattr(self, 'notes_completed') and self.notes_completed and not self.notes_completed.done():
                self.notes_completed.set_result(final_notes)
    
    async def create_notes(self, topic: str, requirements: Dict[str, Any] = None) -> str:
        """Main entry point for creating notes"""
        if requirements is None:
            requirements = {}
        
        # Record user request
        self.recorder.dump_user_request(topic, requirements)
        
        # Send initial request to Agent 1
        request_message = Message(
            sender="user",
            recipient="agent_1",
            message_type="create_notes_request",
            content={
                "topic": topic,
                "requirements": requirements
            },
            timestamp=asyncio.get_event_loop().time()
        )
        
        # Create a future to track completion
        self.notes_completed = asyncio.Future()
        
        await self.route_message(request_message)
        
        # Start all agents
        tasks = [
            asyncio.create_task(agent.start()) 
            for agent in self.agents.values()
        ]
        
        try:
            # Wait for completion with timeout
            final_notes = await asyncio.wait_for(self.notes_completed, timeout=300.0)  # 5 minute timeout
            logger.info("Notes generation completed successfully!")
            
            # Record final summary
            self.recorder.dump_final_summary(
                final_notes=final_notes,
                plan_executed={"objective": "Note generation completed"},
                gaps_filled=[]
            )
            
            # Save JSON dump
            self.recorder.save_json_dump()
            
            # Print session stats
            stats = self.recorder.get_session_stats()
            logger.info(f"Session recorded: {stats}")
            
            return final_notes
            
        except asyncio.TimeoutError:
            logger.error("Note generation timed out after 5 minutes")
            return "Note generation failed: Timeout error"
        except Exception as e:
            logger.error(f"Note generation failed: {e}")
            return f"Note generation failed: {e}"
        finally:
            # Stop agents
            for agent in self.agents.values():
                if hasattr(agent, 'stop'):
                    agent.stop()
            
            for task in tasks:
                if not task.done():
                    task.cancel()
    
    def shutdown(self):
        """Clean shutdown of the system"""
        logger.info("Shutting down Multi-Agent System...")
        
        # Stop all agents
        for agent in self.agents.values():
            if hasattr(agent, 'stop'):
                agent.stop()
        
        # Save final recorder state
        if self.recorder:
            self.recorder.save_json_dump()
        
        logger.info("System shutdown complete")
