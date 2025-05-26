import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .base_agent import BaseAgent
from ..core.data_structures import Message, Plan, SearchQuery
from ..core.llm_wrapper import QwenLLM

logger = logging.getLogger(__name__)

class Agent1_Planner(BaseAgent):
    """Strategic Planner & Orchestrator"""
    
    def __init__(self, llm: QwenLLM):
        super().__init__("agent_1", "Strategic Planner")
        self.llm = llm
        self.current_plan = None
        self.execution_state = {}
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process messages and coordinate the workflow"""
        logger.info(f"Agent 1 processing message: {message.message_type}")
        
        if message.message_type == "create_notes_request":
            return await self.create_plan(message.content)
        
        elif message.message_type == "search_results":
            return await self.handle_search_results(message.content)
        
        elif message.message_type == "synthesis_complete":
            return await self.review_and_finalize(message.content)
        
        elif message.message_type == "gaps_filled":
            return await self.handle_gaps_filled(message.content)
        
        return None
    
    async def create_plan(self, request: Dict[str, Any]) -> Message:
        """Create a comprehensive plan for note-taking using Qwen"""
        topic = request.get("topic", "")
        requirements = request.get("requirements", {})
        
        system_prompt = """You are a strategic planner for an AI note-taking system. Your job is to create comprehensive plans for gathering and organizing information on any given topic.

You must respond with a structured JSON plan containing:
1. An objective statement
2. A list of logical steps to gather information
3. Specific search queries to find relevant information
4. Expected sections for the final notes

Be thorough and consider what information would be most valuable for someone learning about this topic."""
        
        user_prompt = f"""Create a detailed plan for creating comprehensive notes on the topic: "{topic}"

Requirements: {json.dumps(requirements, indent=2)}

Please provide a structured plan that will help gather all necessary information to create excellent study notes on this topic. Focus on what specific information needs to be searched for and how it should be organized."""
        
        thinking, response = self.llm.generate_response(user_prompt, system_prompt, max_new_tokens=8196)
        
        # Parse the LLM response to extract plan components
        plan = self.parse_plan_response(response, topic)
        
        # Record plan creation
        from ..utils.recorder import get_global_recorder
        recorder = get_global_recorder()
        recorder.dump_plan_creation(plan, thinking)
        
        self.current_plan = plan
        self.execution_state = {"current_query_index": 0, "gathered_info": []}
        
        logger.info(f"Agent 1 thinking: {thinking[:200]}...")
        logger.info(f"Created plan with {len(plan.search_queries)} search queries")
        
        # Send first search request to Agent 3
        return await self.send_message(
            "agent_3",
            "search_request",
            {
                "query": asdict(plan.search_queries[0]),
                "plan_context": asdict(plan)
            }
        )
    
    def parse_plan_response(self, response: str, topic: str) -> Plan:
        """Parse LLM response into a structured plan"""
        try:
            # Try to extract JSON from the response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                plan_data = json.loads(json_str)
                
                # Extract components
                objective = plan_data.get('objective', f"Create comprehensive notes on {topic}")
                steps = plan_data.get('steps', [])
                queries = plan_data.get('search_queries', [])
                sections = plan_data.get('expected_sections', [])
                
                # Convert queries to SearchQuery objects
                search_queries = []
                for q in queries:
                    if isinstance(q, str):
                        search_queries.append(SearchQuery(query=q))
                    elif isinstance(q, dict):
                        search_queries.append(SearchQuery(**q))
                
                return Plan(
                    objective=objective,
                    steps=steps,
                    search_queries=search_queries,
                    expected_sections=sections
                )
        except Exception as e:
            logger.error(f"Error parsing plan response: {e}")
        
        # Fallback: create a basic plan
        return Plan(
            objective=f"Create comprehensive notes on {topic}",
            steps=[
                "Search for fundamental concepts",
                "Gather detailed information",
                "Organize into structured notes"
            ],
            search_queries=[
                SearchQuery(query=f"fundamentals of {topic}"),
                SearchQuery(query=f"key concepts in {topic}"),
                SearchQuery(query=f"examples and applications of {topic}")
            ],
            expected_sections=["Introduction", "Key Concepts", "Examples", "Summary"]
        )
    
    async def handle_search_results(self, content: Dict[str, Any]) -> Optional[Message]:
        """Handle search results and decide next steps"""
        results = content.get("results", [])
        self.execution_state["gathered_info"].extend(results)
        
        # Move to next query
        self.execution_state["current_query_index"] += 1
        
        if self.execution_state["current_query_index"] < len(self.current_plan.search_queries):
            # Send next search request
            next_query = self.current_plan.search_queries[self.execution_state["current_query_index"]]
            return await self.send_message(
                "agent_3",
                "search_request",
                {
                    "query": asdict(next_query),
                    "plan_context": asdict(self.current_plan)
                }
            )
        else:
            # All searches complete, send to synthesizer
            return await self.send_message(
                "agent_2",
                "synthesize_request",
                {
                    "plan": asdict(self.current_plan),
                    "gathered_info": self.execution_state["gathered_info"]
                }
            )
    
    async def review_and_finalize(self, content: Dict[str, Any]) -> Message:
        """Review final output and identify any gaps using Qwen"""
        notes = content.get("notes", "")
        
        # Store the original notes in execution state for later use
        self.execution_state["original_notes"] = notes
        
        system_prompt = """You are a quality reviewer for educational notes. Your job is to review generated notes and identify any missing sections or gaps that should be addressed.

Analyze the provided notes and expected sections, then identify what's missing or could be improved. Focus on educational completeness and structure."""
        
        user_prompt = f"""Review these generated notes and identify any gaps or missing sections:

EXPECTED SECTIONS:
{json.dumps(self.current_plan.expected_sections, indent=2)}

GENERATED NOTES:
{notes}

Please identify:
1. Which expected sections are missing or incomplete
2. What additional information would improve the notes
3. Specific search queries that could fill the gaps

Respond with a JSON object containing 'gaps' (list of missing sections) and 'additional_queries' (list of search queries to fill gaps)."""

        thinking, response = self.llm.generate_response(user_prompt, system_prompt, max_new_tokens=8196)
        
        # Parse gaps response
        gaps_data = self.parse_gaps_response(response)
        
        if gaps_data.get("additional_queries"):
            # Send gap-filling request to Agent 3
            return await self.send_message(
                "agent_3",
                "gap_search_request",
                {
                    "queries": gaps_data["additional_queries"],
                    "original_notes": notes,
                    "gaps": gaps_data["gaps"]
                }
            )
        else:
            # No gaps found, finalize
            return await self.send_message(
                "system",
                "notes_complete",
                {"final_notes": notes}
            )
    
    def parse_gaps_response(self, response: str) -> Dict[str, List[str]]:
        """Parse gaps analysis response"""
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error parsing gaps response: {e}")
        
        return {"gaps": [], "additional_queries": []}
    
    async def handle_gaps_filled(self, content: Dict[str, Any]) -> Message:
        """Handle completed gap filling"""
        final_notes = content.get("enhanced_notes", self.execution_state.get("original_notes", ""))
        
        return await self.send_message(
            "system",
            "notes_complete",
            {"final_notes": final_notes}
        )
