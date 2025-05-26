import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ..core.data_structures import Message

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.message_queue = asyncio.Queue()
        self.is_running = False
        self.message_router = None  # Will be set by the MultiAgentSystem
    
    @abstractmethod
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming messages and return response if needed"""
        pass
    
    async def send_message(self, recipient: str, message_type: str, content: Dict[str, Any]) -> Message:
        """Send message to another agent"""
        message = Message(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            timestamp=asyncio.get_event_loop().time()
        )
        
        # Record the message (import here to avoid circular imports)
        from ..utils.recorder import get_global_recorder
        recorder = get_global_recorder()
        recorder.dump_agent_message(message, f"Sent by {self.name}")
        
        return message
    
    async def receive_message(self, message: Message):
        """Receive message from another agent"""
        await self.message_queue.put(message)
    
    async def start(self):
        """Start the agent's message processing loop"""
        self.is_running = True
        while self.is_running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                response = await self.process_message(message)
                
                if response and self.message_router:
                    await self.message_router(response)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in {self.name}: {e}")
    
    def stop(self):
        """Stop the agent"""
        self.is_running = False
    
    def set_recorder(self, recorder):
        """Set the recorder for this agent"""
        self.recorder = recorder
