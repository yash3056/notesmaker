"""
LLM wrapper classes for the AI Multi-Agent Note Taking System
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Base class for all LLM implementations"""
    
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str = "", max_new_tokens: int = 8196, thinking: bool = True) -> Tuple[str, str]:
        """Generate response with optional thinking process"""
        pass


class QwenLLM(BaseLLM):
    """Wrapper for Qwen model to handle agent reasoning"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B"):
        logger.info(f"Loading Qwen model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        logger.info("Qwen model loaded successfully")
    
    def generate_response(self, prompt: str, system_prompt: str = "", max_new_tokens: int = 8196, thinking: bool = True) -> Tuple[str, str]:
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
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Check if generation was truncated (ended with EOS token naturally vs hit max_new_tokens)
        generation_truncated = len(output_ids) >= max_new_tokens
        if generation_truncated:
            logger.warning(f"LLM generation truncated at {max_new_tokens} tokens")
        
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
        
        # Record LLM interaction
        try:
            from ..utils.recorder import get_global_recorder
            recorder = get_global_recorder()
            recorder.dump_llm_interaction(
                prompt=prompt,
                system_prompt=system_prompt,
                thinking=thinking_content,
                response=content,
                model_name=self.model_name,
                max_new_tokens=max_new_tokens,
                truncated=generation_truncated
            )
        except Exception as e:
            logger.debug(f"Failed to record LLM interaction: {e}")
        
        logger.info(f"Generated content length: {len(content)} characters")
        if generation_truncated:
            logger.warning("Content may be incomplete due to token limit")
        
        return thinking_content, content
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "vocab_size": len(self.tokenizer)
        }
