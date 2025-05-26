import json
import asyncio
import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from record import Record

# Document readers for different file types
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create global recorder instance with full logging (no truncation)
recorder = Record(truncate_content=False)

class DocumentLoader:
    """Handles loading documents from various file formats"""
    
    @staticmethod
    def load_documents_from_folder(folder_path: str = "documents") -> List[str]:
        """Load all documents from the specified folder"""
        if not os.path.exists(folder_path):
            logger.warning(f"Documents folder '{folder_path}' not found. Creating it...")
            os.makedirs(folder_path)
            logger.info(f"Created folder '{folder_path}'. Please add your documents there.")
            return []
        
        documents = []
        supported_extensions = ['.txt', '.md', '.pdf', '.docx']
        
        # Get all files in the documents folder
        for ext in supported_extensions:
            pattern = os.path.join(folder_path, f"*{ext}")
            files = glob.glob(pattern)
            
            for file_path in files:
                logger.info(f"Loading document: {file_path}")
                try:
                    content = DocumentLoader.load_single_document(file_path)
                    if content:
                        # Split large documents into chunks
                        chunks = DocumentLoader.split_document(content, file_path)
                        documents.extend(chunks)
                        logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Total loaded documents/chunks: {len(documents)}")
        return documents
    
    @staticmethod
    def load_single_document(file_path: str) -> str:
        """Load a single document based on its file extension"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.txt' or ext == '.md':
            return DocumentLoader.load_text_file(file_path)
        elif ext == '.pdf':
            return DocumentLoader.load_pdf_file(file_path)
        elif ext == '.docx':
            return DocumentLoader.load_docx_file(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return ""
    
    @staticmethod
    def load_text_file(file_path: str) -> str:
        """Load text or markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    @staticmethod
    def load_pdf_file(file_path: str) -> str:
        """Load PDF file"""
        if not PDF_AVAILABLE:
            logger.warning("PyPDF2 not installed. Cannot read PDF files. Install with: pip install PyPDF2")
            return ""
        
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n[Page {page_num + 1}]\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Error reading page {page_num + 1} from {file_path}: {e}")
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return ""
    
    @staticmethod
    def load_docx_file(file_path: str) -> str:
        """Load DOCX file"""
        if not DOCX_AVAILABLE:
            logger.warning("python-docx not installed. Cannot read DOCX files. Install with: pip install python-docx")
            return ""
        
        try:
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    @staticmethod
    def split_document(content: str, file_path: str, text_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split large documents into smaller chunks for better search"""
        if len(content) <= text_chunk_size:
            return [content]
        
        chunks = []
        filename = os.path.basename(file_path)
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        current_chunk = f"[Source: {filename}]\n"
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > text_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = f"[Source: {filename}]\n{paragraph}\n"
            else:
                current_chunk += paragraph + "\n\n"
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If we still have very large chunks, split them further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > text_chunk_size * 2:
                # Split by sentences
                sentences = chunk.split('. ')
                current_subchunk = f"[Source: {filename}]\n"
                
                for sentence in sentences:
                    if len(current_subchunk) + len(sentence) > text_chunk_size:
                        if current_subchunk.strip():
                            final_chunks.append(current_subchunk.strip())
                        current_subchunk = f"[Source: {filename}]\n{sentence}. "
                    else:
                        current_subchunk += sentence + ". "
                
                if current_subchunk.strip():
                    final_chunks.append(current_subchunk.strip())
            else:
                final_chunks.append(chunk)
        
        return final_chunks

class QwenLLM:
    """Wrapper for Qwen model to handle agent reasoning"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B"):
        logger.info(f"Loading Qwen model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        logger.info("Qwen model loaded successfully")
    
    def generate_response(self, prompt: str, system_prompt: str = "", max_new_tokens: int = 8196, thinking: bool = True) -> tuple[str, str]:
        """Generate response with optional thinking process"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Check if generation was truncated (ended with EOS token naturally vs hit max_new_tokens)
        generation_truncated = len(output_ids) >= max_new_tokens
        if generation_truncated:
            logger.warning(f"Generation may have been truncated at {max_new_tokens} tokens")
        
        # Parse thinking content if enabled
        thinking_content = ""
        content = ""
        
        if thinking:
            try:
                # Find the end of thinking token (151668 is </think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            except ValueError:
                # No thinking tokens found
                content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        else:
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        logger.info(f"Generated content length: {len(content)} characters")
        if generation_truncated:
            logger.warning("Content may be incomplete due to token limit")
        
        # Record the LLM interaction
        recorder.dump_llm_interaction(
            agent_name="QwenLLM",
            prompt=prompt,
            system_prompt=system_prompt,
            thinking=thinking_content,
            response=content,
            token_info={
                "max_new_tokens": max_new_tokens,
                "generated_tokens": len(output_ids),
                "truncated": generation_truncated
            }
        )
        
        return thinking_content, content

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
        """Process incoming messages"""
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
        
        # Record the message
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
                    # Route the response through the message router
                    logger.info(f"{self.name} sending response: {response.message_type}")
                    await self.message_router(response)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in {self.name}: {e}")
    
    def stop(self):
        """Stop the agent"""
        self.is_running = False

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
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end]
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")
            
            plan_data = json.loads(json_str)
            
            # Extract search queries
            search_queries = []
            for q in plan_data.get("search_queries", []):
                if isinstance(q, str):
                    search_queries.append(SearchQuery(q))
                elif isinstance(q, dict):
                    search_queries.append(SearchQuery(
                        query=q.get("query", ""),
                        max_results=q.get("max_results", 5)
                    ))
            
            return Plan(
                objective=plan_data.get("objective", f"Create notes on {topic}"),
                steps=plan_data.get("steps", []),
                search_queries=search_queries,
                expected_sections=plan_data.get("expected_sections", [])
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback to default plan
            return Plan(
                objective=f"Create comprehensive notes on {topic}",
                steps=[
                    "Gather foundational information",
                    "Identify key concepts",
                    "Find examples and applications",
                    "Organize information"
                ],
                search_queries=[
                    SearchQuery(f"introduction to {topic}", max_results=5),
                    SearchQuery(f"{topic} key concepts", max_results=5),
                    SearchQuery(f"{topic} examples", max_results=5)
                ],
                expected_sections=[
                    "Introduction", "Key Concepts", "Examples", "Summary"
                ]
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
            # All searches complete, send to Agent 2 for synthesis
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
        
        logger.info(f"Agent 1 review thinking: {thinking[:150]}...")
        
        # Parse gaps from response
        gaps = self.parse_gaps_response(response)
        
        if gaps and gaps.get("additional_queries"):
            # Request additional information
            gap_queries = [SearchQuery(query, max_results=3) for query in gaps.get("additional_queries", [])]
            return await self.send_message(
                "agent_3",
                "gap_search_request",
                {
                    "queries": [asdict(q) for q in gap_queries],
                    "missing_sections": gaps.get("gaps", [])
                }
            )
        else:
            # Notes are complete
            return await self.send_message(
                "system",
                "notes_complete",
                {
                    "final_notes": notes,
                    "plan_executed": asdict(self.current_plan)
                }
            )
    
    def parse_gaps_response(self, response: str) -> Dict[str, List[str]]:
        """Parse LLM response for gap analysis"""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end]
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                return {}
            
            gaps_data = json.loads(json_str)
            return gaps_data
            
        except Exception as e:
            logger.warning(f"Failed to parse gaps response: {e}")
            return {}
    
    async def handle_gaps_filled(self, content: Dict[str, Any]) -> Message:
        """Handle gaps filled response and finalize notes"""
        status = content.get("status", "")
        additional_content = content.get("additional_content", "")
        filled_sections = content.get("filled_sections", [])
        
        logger.info(f"Gaps filled - Status: {status}, Sections: {filled_sections}")
        
        # Get the original notes from the execution state
        original_notes = self.execution_state.get("original_notes", "")
        
        if additional_content and additional_content.strip():
            # Combine original notes with additional content
            final_notes = f"{original_notes}\n\n## Additional Sections\n\n{additional_content}"
        else:
            # No additional content, use original notes
            final_notes = original_notes
        
        # Send final notes to system
        return await self.send_message(
            "system",
            "notes_complete",
            {
                "final_notes": final_notes,
                "plan_executed": asdict(self.current_plan),
                "gaps_filled": filled_sections
            }
        )

class Agent2_Synthesizer(BaseAgent):
    """Content Synthesizer & Writer"""
    
    def __init__(self, llm: QwenLLM, config: SynthesisConfig = None):
        super().__init__("agent_2", "Content Synthesizer")
        self.llm = llm
        self.config = config or SynthesisConfig()
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process synthesis requests"""
        logger.info(f"Agent 2 processing message: {message.message_type}")
        
        if message.message_type == "synthesize_request":
            # Check if we should use chunked synthesis
            gathered_info = message.content.get("gathered_info", [])
            if len(gathered_info) > 10:
                logger.info("Using chunked synthesis for large dataset")
                return await self.synthesize_notes_chunked(message.content)
            else:
                return await self.synthesize_notes(message.content)
        
        elif message.message_type == "gap_fill_request":
            return await self.fill_gaps(message.content)
        
        return None
    
    async def synthesize_notes(self, content: Dict[str, Any]) -> Message:
        """Synthesize search results into coherent notes using Qwen"""
        plan = content.get("plan", {})
        gathered_info = content.get("gathered_info", [])
        
        # Prepare information for the LLM
        search_results_text = "\n\n".join([
            f"Source {i+1} (Score: {info.get('score', 'N/A')}):\n{info.get('text', '')}"
            for i, info in enumerate(gathered_info)
        ])
        
        system_prompt = """You are an expert content synthesizer and educational writer. Your job is to take search results and create comprehensive, well-structured study notes.

Create notes that are:
1. Well-organized with clear sections and subsections
2. Educational and easy to understand
3. Comprehensive but concise
4. Properly formatted using markdown
5. Include key concepts, definitions, examples, and explanations

Focus on creating high-quality educational content that would help someone learn the topic effectively."""
        
        user_prompt = f"""Create comprehensive study notes based on the following information:

OBJECTIVE: {plan.get('objective', 'Create study notes')}

EXPECTED SECTIONS: {json.dumps(plan.get('expected_sections', []), indent=2)}

SEARCH RESULTS:
{search_results_text}

Please synthesize this information into well-structured, comprehensive study notes. Use markdown formatting and organize the content logically according to the expected sections."""
        
        logger.info(f"Agent 2 starting synthesis with {len(gathered_info)} sources")
        thinking, notes = self.llm.generate_response(user_prompt, system_prompt, max_new_tokens=self.config.main_synthesis_tokens)
        
        # Record synthesis
        recorder.dump_synthesis(
            notes=notes,
            thinking=thinking,
            method="standard",
            chunk_info={"total_sources": len(gathered_info)}
        )
        
        logger.info(f"Agent 2 synthesis completed - Notes length: {len(notes)} characters")
        logger.info(f"Agent 2 synthesis thinking: {thinking[:150]}...")
        
        return await self.send_message(
            "agent_1",
            "synthesis_complete",
            {
                "notes": notes,
                "sections_covered": plan.get('expected_sections', []),
                "total_sources": len(gathered_info)
            }
        )
    
    async def synthesize_notes_chunked(self, content: Dict[str, Any]) -> Message:
        """Synthesize search results in chunks if the content is too large"""
        plan = content.get("plan", {})
        gathered_info = content.get("gathered_info", [])
        
        # If we have a lot of information, break it into chunks
        if len(gathered_info) > 10:  # More than 10 sources
            logger.info(f"Using chunked synthesis for {len(gathered_info)} sources")
            
            batch_size = 5  # Process 5 sources at a time
            all_sections = []
            
            for i in range(0, len(gathered_info), batch_size):
                chunk = gathered_info[i:i+batch_size]
                
                search_results_text = "\n\n".join([
                    f"Source {i+j+1} (Score: {info.get('score', 'N/A')}):\n{info.get('text', '')}"
                    for j, info in enumerate(chunk)
                ])
                
                system_prompt = """You are an expert content synthesizer. Create a focused section of study notes based on the provided sources. Focus on key concepts, definitions, and explanations. Use markdown formatting."""
                
                user_prompt = f"""Create a section of study notes based on these sources (part {i//batch_size + 1}):

                    OBJECTIVE: {plan.get('objective', 'Create study notes')}

                    SEARCH RESULTS:
                    {search_results_text}

                    Create well-structured content that covers the key points from these sources."""
                
                thinking, section_notes = self.llm.generate_response(user_prompt, system_prompt, max_new_tokens=8196)
                all_sections.append(section_notes)
                
                logger.info(f"Completed chunk {i//batch_size + 1}, section length: {len(section_notes)} characters")
            
            # Now combine all sections into final notes
            combined_content = "\n\n".join(all_sections)
            
            final_system_prompt = """You are organizing and finalizing study notes. Take the provided sections and organize them into a coherent, well-structured document with proper headings and flow."""
            
            final_user_prompt = f"""Organize these sections into comprehensive, well-structured study notes:

OBJECTIVE: {plan.get('objective', 'Create study notes')}
EXPECTED SECTIONS: {json.dumps(plan.get('expected_sections', []), indent=2)}

CONTENT SECTIONS:
{combined_content}

Please organize this into a final, coherent document with proper markdown formatting and logical section organization."""
            
            thinking, final_notes = self.llm.generate_response(final_user_prompt, final_system_prompt, max_new_tokens=8196)
            
            logger.info(f"Chunked synthesis completed - Final notes length: {len(final_notes)} characters")
            
            return await self.send_message(
                "agent_1",
                "synthesis_complete",
                {
                    "notes": final_notes,
                    "sections_covered": plan.get('expected_sections', []),
                    "total_sources": len(gathered_info),
                    "synthesis_method": "chunked"
                }
            )
        else:
            # Use regular synthesis for smaller amounts of data
            return await self.synthesize_notes(content)
    
    async def fill_gaps(self, content: Dict[str, Any]) -> Message:
        """Fill gaps in notes using additional search results"""
        gap_results = content.get("gap_results", [])
        missing_sections = content.get("missing_sections", [])
        
        if not gap_results:
            logger.info("No gap results to process")
            return await self.send_message(
                "agent_1",
                "gaps_filled",
                {
                    "status": "no_additional_content",
                    "additional_content": "",
                    "filled_sections": []
                }
            )
        
        # Prepare gap information for the LLM
        gap_results_text = "\n\n".join([
            f"Gap Source {i+1} (Score: {info.get('score', 'N/A')}):\n{info.get('text', '')}"
            for i, info in enumerate(gap_results)
        ])
        
        system_prompt = """You are an expert content writer specializing in filling gaps in educational notes. Your job is to create additional content that addresses missing sections or incomplete information.

Create content that is:
1. Focused on the missing sections identified
2. Well-structured with clear headings
3. Educational and comprehensive
4. Properly formatted using markdown
5. Complementary to existing notes

Focus on creating high-quality educational content that fills the identified gaps."""
        
        user_prompt = f"""Create additional content to fill the following gaps in study notes:

MISSING SECTIONS: {json.dumps(missing_sections, indent=2)}

ADDITIONAL SEARCH RESULTS:
{gap_results_text}

Please create well-structured content that addresses the missing sections. Use markdown formatting and organize the content with appropriate headings for each missing section."""
        
        logger.info(f"Agent 2 filling gaps with {len(gap_results)} additional sources")
        logger.info(f"Missing sections: {missing_sections}")
        
        thinking, additional_content = self.llm.generate_response(
            user_prompt, 
            system_prompt, 
            max_new_tokens=self.config.gap_filling_tokens
        )
        
        logger.info(f"Agent 2 gap filling completed - Additional content length: {len(additional_content)} characters")
        logger.info(f"Agent 2 gap filling thinking: {thinking[:150]}...")
        
        return await self.send_message(
            "agent_1",
            "gaps_filled",
            {
                "status": "gaps_filled",
                "additional_content": additional_content,
                "filled_sections": missing_sections,
                "total_gap_sources": len(gap_results)
            }
        )
        

class Agent3_Retriever(BaseAgent):
    """Information Retrieval Specialist with FAISS"""
    
    def __init__(self, document_texts: List[str] = None):
        super().__init__("agent_3", "Information Retriever")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = document_texts or []
        self.index = None
        self.document_embeddings = None
        
        if self.documents:
            self.build_index()
    
    def build_index(self):
        """Build FAISS index from documents"""
        logger.info(f"Building FAISS index for {len(self.documents)} documents...")
        
        # Create embeddings
        self.document_embeddings = self.encoder.encode(self.documents)
        
        # Build FAISS index
        dimension = self.document_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.document_embeddings)
        self.index.add(self.document_embeddings)
        
        logger.info("FAISS index built successfully")
    
    def add_documents(self, new_documents: List[str]):
        """Add new documents to the index"""
        self.documents.extend(new_documents)
        self.build_index()
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process search requests"""
        logger.info(f"Agent 3 processing message: {message.message_type}")
        
        if message.message_type == "search_request":
            return await self.handle_search_request(message.content)
        
        elif message.message_type == "gap_search_request":
            return await self.handle_gap_search(message.content)
        
        return None
    
    async def handle_search_request(self, content: Dict[str, Any]) -> Message:
        """Handle single search request"""
        query_data = content.get("query", {})
        query_text = query_data.get("query", "")
        max_results = query_data.get("max_results", 5)
        
        results = self.search_documents(query_text, max_results)
        
        # Record search results
        recorder.dump_search_results(query_text, results, self.name)
        
        return await self.send_message(
            "agent_1",
            "search_results",
            {
                "query": query_text,
                "results": [asdict(r) for r in results],
                "total_found": len(results)
            }
        )
    
    async def handle_gap_search(self, content: Dict[str, Any]) -> Message:
        """Handle gap-filling search requests"""
        queries = content.get("queries", [])
        all_results = []
        
        for query_data in queries:
            query_text = query_data.get("query", "")
            max_results = query_data.get("max_results", 3)
            results = self.search_documents(query_text, max_results)
            all_results.extend(results)
        
        return await self.send_message(
            "agent_2",
            "gap_fill_request",
            {
                "gap_results": [asdict(r) for r in all_results],
                "missing_sections": content.get("missing_sections", [])
            }
        )
    
    def search_documents(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search documents using FAISS"""
        if not self.index or not self.documents:
            logger.warning("No documents indexed")
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(max_results, len(self.documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append(SearchResult(
                    text=self.documents[idx],
                    score=float(score),
                    metadata={
                        "document_index": int(idx),
                        "query": query
                    }
                ))
        
        return results

class MultiAgentSystem:
    """Main system orchestrating all agents"""
    
    def __init__(self, documents_folder: str = "documents", model_name: str = "Qwen/Qwen3-1.7B"):
        # Load documents from folder
        logger.info(f"Loading documents from folder: {documents_folder}")
        documents = DocumentLoader.load_documents_from_folder(documents_folder)
        
        if not documents:
            logger.warning("No documents loaded! The system will work but won't have any content to search.")
            logger.info("Please add .txt, .md, .pdf, or .docx files to the 'documents' folder.")
        
        # Initialize shared LLM
        self.llm = QwenLLM(model_name)
        
        # Initialize agents with shared LLM
        self.agent1 = Agent1_Planner(self.llm)
        self.agent2 = Agent2_Synthesizer(self.llm)
        self.agent3 = Agent3_Retriever(documents)
        
        self.agents = {
            "agent_1": self.agent1,
            "agent_2": self.agent2,
            "agent_3": self.agent3
        }
        
        # Set the message router for each agent
        for agent in self.agents.values():
            agent.message_router = self.route_message
        
        # Enable full logging (no truncation)
        recorder.enable_full_logging()
        
        self.message_broker = asyncio.Queue()
        self.is_running = False
    
    def add_documents_from_folder(self, folder_path: str):
        """Add more documents from another folder"""
        new_documents = DocumentLoader.load_documents_from_folder(folder_path)
        if new_documents:
            self.agent3.add_documents(new_documents)
            logger.info(f"Added {len(new_documents)} new document chunks")
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents"""
        return {
            "total_documents": len(self.agent3.documents),
            "index_built": self.agent3.index is not None,
            "embedding_dimension": self.agent3.document_embeddings.shape[1] if self.agent3.document_embeddings is not None else 0
        }
    
    async def route_message(self, message: Message):
        """Route messages between agents"""
        logger.info(f"Routing message from {message.sender} to {message.recipient}: {message.message_type}")
        
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
            if hasattr(self, 'notes_completed') and not self.notes_completed.done():
                self.notes_completed.set_result(final_notes)
    
    async def create_notes(self, topic: str, requirements: Dict[str, Any] = None) -> str:
        """Main entry point for creating notes"""
        if requirements is None:
            requirements = {}
        
        # Record user request
        recorder.dump_user_request(topic, requirements)
        
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
            # Wait for completion - no timeout
            final_notes = await self.notes_completed
            logger.info("Notes generation completed successfully!")
            
            # Record final summary
            recorder.dump_final_summary(
                final_notes=final_notes,
                plan_executed={"objective": "Note generation completed"},
                gaps_filled=[]
            )
            
            # Save JSON dump
            recorder.save_json_dump()
            
            # Print session stats
            stats = recorder.get_session_stats()
            logger.info(f"Session recorded: {stats}")
            
            return final_notes
        except Exception as e:
            logger.error(f"Note generation failed: {e}")
            return f"Note generation failed: {e}"
        finally:
            # Stop agents
            for agent in self.agents.values():
                agent.stop()
            
            for task in tasks:
                task.cancel()

# Example usage and testing
async def main():
    # Create the multi-agent system (will automatically load from 'documents' folder)
    system = MultiAgentSystem(documents_folder="documents")
    
    # Print document statistics
    stats = system.get_document_stats()
    print(f"Loaded {stats['total_documents']} document chunks")
    print(f"Index built: {stats['index_built']}")
    
    if stats['total_documents'] == 0:
        print("\n" + "="*50)
        print("NO DOCUMENTS FOUND!")
        print("="*50)
        print("Please add documents to the 'documents' folder:")
        print("- .txt files (plain text)")
        print("- .md files (markdown)")
        print("- .pdf files (requires: pip install PyPDF2)")
        print("- .docx files (requires: pip install python-docx)")
        print("="*50)
        return
    
    # Create notes on a topic
    print("\nGenerating notes... (this may take a few minutes)")
    print("Processing documents, generating content, and synthesizing notes...")
    
    notes = await system.create_notes(
        topic="artificial intelligence and machine learning",
        requirements={
            "depth": "comprehensive",
            "include_examples": True,
            "format": "structured",
            "audience": "students"
        }
    )
    
    if not notes.startswith("Note generation timed out"):
        print("\nNote generation completed successfully!")

if __name__ == "__main__":
    
    # Run the system
    asyncio.run(main())