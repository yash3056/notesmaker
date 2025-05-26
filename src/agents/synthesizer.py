import logging
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .base_agent import BaseAgent
from ..core.data_structures import Message, SynthesisConfig
from ..core.llm_wrapper import QwenLLM

logger = logging.getLogger(__name__)

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
            return await self.synthesize_notes(message.content)
        
        elif message.message_type == "gap_fill_request":
            return await self.fill_gaps(message.content)
        
        return None
    
    async def synthesize_notes(self, content: Dict[str, Any]) -> Message:
        """Synthesize comprehensive notes from gathered information"""
        plan = content.get("plan", {})
        gathered_info = content.get("gathered_info", [])
        
        logger.info(f"Synthesizing notes from {len(gathered_info)} sources")
        
        # Choose synthesis method based on amount of information
        if len(gathered_info) > self.config.chunked_threshold:
            return await self.synthesize_notes_chunked(content)
        else:
            return await self.synthesize_notes_standard(content)
    
    async def synthesize_notes_standard(self, content: Dict[str, Any]) -> Message:
        """Standard synthesis for moderate amounts of information"""
        plan = content.get("plan", {})
        gathered_info = content.get("gathered_info", [])
        
        # Prepare sources text
        sources_text = "\n\n".join([
            f"Source {i+1}:\n{info.get('text', str(info))}"
            for i, info in enumerate(gathered_info)
        ])
        
        system_prompt = """You are an expert educational content synthesizer. Your job is to create comprehensive, well-structured notes from multiple sources of information.

Create notes that are:
1. Comprehensive and educational
2. Well-organized with clear sections
3. Include examples and explanations
4. Suitable for studying and learning
5. Properly formatted with headers and structure

Focus on synthesis rather than summarization - combine information from multiple sources into a coherent educational resource."""
        
        user_prompt = f"""Create comprehensive educational notes based on the following plan and sources:

PLAN OBJECTIVE:
{plan.get('objective', 'Create comprehensive notes')}

EXPECTED SECTIONS:
{plan.get('expected_sections', [])}

SOURCES:
{sources_text}

Create well-structured, comprehensive notes that synthesize all the information above. Include clear headings, explanations, examples, and ensure educational value."""
        
        thinking, response = self.llm.generate_response(
            user_prompt, 
            system_prompt, 
            max_new_tokens=self.config.main_synthesis_tokens
        )
        
        # Record synthesis
        from ..utils.recorder import get_global_recorder
        recorder = get_global_recorder()
        recorder.dump_synthesis(response, thinking, method="standard")
        
        logger.info(f"Standard synthesis complete: {len(response)} characters")
        
        return await self.send_message(
            "agent_1",
            "synthesis_complete",
            {"notes": response, "method": "standard"}
        )
    
    async def synthesize_notes_chunked(self, content: Dict[str, Any]) -> Message:
        """Chunked synthesis for large amounts of information"""
        plan = content.get("plan", {})
        gathered_info = content.get("gathered_info", [])
        
        logger.info(f"Using chunked synthesis for {len(gathered_info)} sources")
        
        # Process sources in batches
        batch_summaries = []
        for i in range(0, len(gathered_info), self.config.batch_size):
            batch = gathered_info[i:i + self.config.batch_size]
            batch_text = "\n\n".join([
                f"Source {j+1}:\n{info.get('text', str(info))}"
                for j, info in enumerate(batch)
            ])
            
            system_prompt = """You are synthesizing information from multiple sources. Create a comprehensive summary that captures all key information, concepts, and examples from the provided sources."""
            
            user_prompt = f"""Synthesize the following sources into a comprehensive summary:

{batch_text}

Create a detailed synthesis that preserves important information, concepts, and examples."""
            
            thinking, batch_summary = self.llm.generate_response(
                user_prompt,
                system_prompt,
                max_new_tokens=self.config.chunked_synthesis_tokens
            )
            
            batch_summaries.append(batch_summary)
            logger.info(f"Processed batch {i//self.config.batch_size + 1}")
        
        # Final synthesis of all batch summaries
        combined_summaries = "\n\n".join([
            f"Section {i+1}:\n{summary}"
            for i, summary in enumerate(batch_summaries)
        ])
        
        system_prompt = """You are creating final comprehensive educational notes. Synthesize all the provided sections into well-structured, comprehensive notes suitable for learning and studying."""
        
        user_prompt = f"""Create comprehensive educational notes from these synthesized sections:

PLAN OBJECTIVE:
{plan.get('objective', 'Create comprehensive notes')}

EXPECTED SECTIONS:
{plan.get('expected_sections', [])}

SYNTHESIZED SECTIONS:
{combined_summaries}

Create the final comprehensive notes with proper structure, headings, and educational value."""
        
        thinking, final_notes = self.llm.generate_response(
            user_prompt,
            system_prompt,
            max_new_tokens=self.config.final_organization_tokens
        )
        
        # Record synthesis
        from ..utils.recorder import get_global_recorder
        recorder = get_global_recorder()
        recorder.dump_synthesis(
            final_notes, 
            thinking, 
            method="chunked",
            chunk_info={
                "total_sources": len(gathered_info),
                "batch_size": self.config.batch_size,
                "num_batches": len(batch_summaries)
            }
        )
        
        logger.info(f"Chunked synthesis complete: {len(final_notes)} characters")
        
        return await self.send_message(
            "agent_1",
            "synthesis_complete",
            {"notes": final_notes, "method": "chunked"}
        )
    
    async def fill_gaps(self, content: Dict[str, Any]) -> Message:
        """Fill gaps in existing notes with additional information"""
        original_notes = content.get("original_notes", "")
        additional_info = content.get("additional_info", [])
        gaps = content.get("gaps", [])
        
        logger.info(f"Filling gaps with {len(additional_info)} additional sources")
        
        # Prepare additional sources
        additional_text = "\n\n".join([
            f"Additional Source {i+1}:\n{info.get('text', str(info))}"
            for i, info in enumerate(additional_info)
        ])
        
        system_prompt = """You are enhancing existing educational notes with additional information. Your job is to integrate new information into existing notes while maintaining structure and coherence.

Enhance the notes by:
1. Adding missing sections identified in the gaps
2. Expanding existing sections with new information
3. Maintaining the overall structure and flow
4. Ensuring all information is well-integrated"""
        
        user_prompt = f"""Enhance these existing notes with additional information:

ORIGINAL NOTES:
{original_notes}

IDENTIFIED GAPS:
{gaps}

ADDITIONAL INFORMATION:
{additional_text}

Please enhance the notes by filling the gaps and integrating the additional information. Maintain the structure and improve comprehensiveness."""
        
        thinking, enhanced_notes = self.llm.generate_response(
            user_prompt,
            system_prompt,
            max_new_tokens=self.config.gap_filling_tokens
        )
        
        # Record gap filling
        from ..utils.recorder import get_global_recorder
        recorder = get_global_recorder()
        recorder.dump_synthesis(enhanced_notes, thinking, method="gap_filling")
        
        logger.info(f"Gap filling complete: {len(enhanced_notes)} characters")
        
        return await self.send_message(
            "agent_1",
            "gaps_filled",
            {"enhanced_notes": enhanced_notes}
        )
