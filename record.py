import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import asdict

class Record:
    """Comprehensive conversation and interaction recorder"""
    
    def __init__(self, session_name: str = None, truncate_content: bool = False):
        self.session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.truncate_content = truncate_content  # New option for truncation control
        self.log_file = f"records/{self.session_name}.txt"
        self.json_file = f"records/{self.session_name}.json"
        
        # Create records directory if it doesn't exist
        os.makedirs("records", exist_ok=True)
        
        # Session data
        self.session_data = {
            "session_id": self.session_name,
            "start_time": datetime.now().isoformat(),
            "interactions": [],
            "llm_calls": [],
            "agent_messages": [],
            "plans": [],
            "synthesis_results": [],
            "search_results": [],
            "errors": []
        }
        
        # Initialize text log
        self._init_text_log()
    
    def _init_text_log(self):
        """Initialize the text log file with header"""
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"""
{'='*80}
AI AGENT CONVERSATION RECORD
{'='*80}
Session: {self.session_name}
Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

""")
    
    def dump_user_request(self, topic: str, additional_info: Dict[str, Any] = None):
        """Record initial user request"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "user_request",
            "topic": topic,
            "additional_info": additional_info or {}
        }
        
        self.session_data["interactions"].append(data)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] USER REQUEST\n")
            f.write(f"Topic: {topic}\n")
            if additional_info:
                f.write(f"Additional Info: {additional_info}\n")
            f.write("-" * 50 + "\n\n")
    
    def dump_agent_message(self, message, context: str = ""):
        """Record agent-to-agent messages"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "agent_message",
            "sender": message.sender,
            "recipient": message.recipient,
            "message_type": message.message_type,
            "content": str(message.content),
            "context": context
        }
        
        self.session_data["agent_messages"].append(data)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] AGENT MESSAGE\n")
            f.write(f"{message.sender} → {message.recipient}\n")
            f.write(f"Type: {message.message_type}\n")
            if context:
                f.write(f"Context: {context}\n")
            f.write(f"Content: {str(message.content)}\n")
            f.write("-" * 50 + "\n\n")
    
    def dump_llm_interaction(self, agent_name: str, prompt: str, system_prompt: str, 
                           thinking: str, response: str, token_info: Dict[str, Any] = None):
        """Record LLM interactions with full context"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "llm_interaction",
            "agent": agent_name,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "thinking": thinking,
            "response": response,
            "token_info": token_info or {}
        }
        
        self.session_data["llm_calls"].append(data)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] LLM INTERACTION - {agent_name}\n")
            f.write(f"System Prompt: {system_prompt}\n")
            f.write(f"User Prompt: {prompt}\n")
            f.write(f"Thinking: {thinking}\n")
            f.write(f"Response: {response}\n")
            if token_info:
                f.write(f"Token Info: {token_info}\n")
            f.write("=" * 80 + "\n\n")
    
    def dump_plan_creation(self, plan, thinking: str = ""):
        """Record plan creation"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "plan_creation",
            "plan": asdict(plan) if hasattr(plan, '__dict__') else str(plan),
            "thinking": thinking
        }
        
        self.session_data["plans"].append(data)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] PLAN CREATED\n")
            f.write(f"Objective: {getattr(plan, 'objective', 'N/A')}\n")
            f.write(f"Search Queries: {[getattr(q, 'query', str(q)) for q in getattr(plan, 'search_queries', [])]}\n")
            f.write(f"Expected Sections: {getattr(plan, 'expected_sections', [])}\n")
            f.write(f"Full Thinking Process:\n{thinking}\n")
            f.write(f"Complete Plan Details:\n{asdict(plan) if hasattr(plan, '__dict__') else str(plan)}\n")
            f.write("-" * 80 + "\n\n")
    
    def dump_search_results(self, query: str, results: List[Any], agent: str = "Agent3"):
        """Record search results"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "search_results",
            "agent": agent,
            "query": query,
            "results_count": len(results),
            "results": [str(r) for r in results]  # Full results, no truncation
        }
        
        self.session_data["search_results"].append(data)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] SEARCH RESULTS - {agent}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Results Found: {len(results)}\n")
            f.write("COMPLETE SEARCH RESULTS:\n")
            f.write("=" * 60 + "\n")
            for i, result in enumerate(results):
                f.write(f"\nResult {i+1}:\n")
                f.write("-" * 30 + "\n")
                f.write(f"{str(result)}\n")
            f.write("=" * 60 + "\n\n")
    
    def dump_synthesis(self, notes: str, thinking: str, method: str = "standard", 
                      chunk_info: Dict[str, Any] = None):
        """Record synthesis results"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "synthesis",
            "method": method,
            "notes_length": len(notes),
            "notes": notes,
            "thinking": thinking,
            "chunk_info": chunk_info
        }
        
        self.session_data["synthesis_results"].append(data)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] SYNTHESIS COMPLETE\n")
            f.write(f"Method: {method}\n")
            f.write(f"Notes Length: {len(notes)} characters\n")
            if chunk_info:
                f.write(f"Chunk Info: {chunk_info}\n")
            f.write("COMPLETE THINKING PROCESS:\n")
            f.write("=" * 60 + "\n")
            f.write(f"{thinking}\n")
            f.write("=" * 60 + "\n")
            f.write("COMPLETE GENERATED NOTES:\n")
            f.write("=" * 60 + "\n")
            f.write(f"{notes}\n")
            f.write("=" * 60 + "\n\n")
    
    def dump_error(self, error: Exception, context: str, agent: str = "Unknown"):
        """Record errors"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "agent": agent,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        self.session_data["errors"].append(data)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR - {agent}\n")
            f.write(f"Type: {type(error).__name__}\n")
            f.write(f"Message: {str(error)}\n")
            f.write(f"Context: {context}\n")
            f.write("!" * 50 + "\n\n")
    
    def dump_final_summary(self, final_notes: str, plan_executed: Dict, gaps_filled: List[str]):
        """Record final completion"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "final_summary",
            "notes_length": len(final_notes),
            "final_notes": final_notes,
            "plan_executed": plan_executed,
            "gaps_filled": gaps_filled,
            "session_complete": True
        }
        
        self.session_data["interactions"].append(data)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] SESSION COMPLETE\n")
            f.write(f"Final Notes Length: {len(final_notes)} characters\n")
            f.write(f"Gaps Filled: {gaps_filled}\n")
            f.write(f"Plan Executed: {plan_executed.get('objective', 'N/A')}\n")
            f.write("=" * 80 + "\n")
            f.write("FINAL NOTES:\n")
            f.write("=" * 80 + "\n")
            f.write(final_notes)
            f.write("\n" + "=" * 80 + "\n\n")
    
    def save_json_dump(self):
        """Save complete session data as JSON"""
        self.session_data["end_time"] = datetime.now().isoformat()
        
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.session_data, f, indent=2, ensure_ascii=False)
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Helper to truncate long text (only if truncation is enabled)"""
        if not self.truncate_content:
            return text  # Return full text if truncation is disabled
        
        if len(text) <= max_length:
            return text
        return text[:max_length] + "...[TRUNCATED]"
    
    def enable_full_logging(self):
        """Enable complete, untruncated logging"""
        self.truncate_content = False
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] LOGGING MODE: FULL (No Truncation)\n")
            f.write("-" * 50 + "\n\n")
    
    def enable_truncated_logging(self):
        """Enable truncated logging to save space"""
        self.truncate_content = True
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] LOGGING MODE: TRUNCATED\n")
            f.write("-" * 50 + "\n\n")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            "session_id": self.session_name,
            "total_interactions": len(self.session_data["interactions"]),
            "llm_calls": len(self.session_data["llm_calls"]),
            "agent_messages": len(self.session_data["agent_messages"]),
            "search_results": len(self.session_data["search_results"]),
            "synthesis_count": len(self.session_data["synthesis_results"]),
            "errors": len(self.session_data["errors"]),
            "log_files": {
                "text": self.log_file,
                "json": self.json_file
            }
        }
