import logging
import numpy as np
import faiss
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from sentence_transformers import SentenceTransformer

from .base_agent import BaseAgent
from ..core.data_structures import Message, SearchResult

logger = logging.getLogger(__name__)

class Agent3_Retriever(BaseAgent):
    """Information Retrieval Specialist with FAISS"""
    
    def __init__(self, document_texts: List[str] = None):
        super().__init__("agent_3", "Information Retriever")
        self.document_texts = document_texts or []
        self.embeddings = None
        self.index = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if self.document_texts:
            self.build_index()
    
    def build_index(self):
        """Build FAISS index from document embeddings"""
        if not self.document_texts:
            logger.warning("No documents provided for indexing")
            return
        
        logger.info(f"Building FAISS index for {len(self.document_texts)} documents...")
        
        # Generate embeddings
        self.embeddings = self.embedding_model.encode(self.document_texts)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {self.index.ntotal} documents")
    
    def add_documents(self, new_documents: List[str]):
        """Add new documents to the index"""
        if not new_documents:
            return
        
        self.document_texts.extend(new_documents)
        
        # Rebuild index with all documents
        self.build_index()
        
        logger.info(f"Added {len(new_documents)} new documents. Total: {len(self.document_texts)}")
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process search requests"""
        logger.info(f"Agent 3 processing message: {message.message_type}")
        
        if message.message_type == "search_request":
            return await self.handle_search_request(message.content)
        
        elif message.message_type == "gap_search_request":
            return await self.handle_gap_search(message.content)
        
        return None
    
    async def handle_search_request(self, content: Dict[str, Any]) -> Message:
        """Handle regular search requests"""
        query_data = content.get("query", {})
        query_text = query_data.get("query", "")
        max_results = query_data.get("max_results", 5)
        
        logger.info(f"Searching for: '{query_text}' (max {max_results} results)")
        
        # Perform search
        results = self.search_documents(query_text, max_results)
        
        # Record search results
        from ..utils.recorder import get_global_recorder
        recorder = get_global_recorder()
        recorder.dump_search_results(query_text, results, "Agent3")
        
        return await self.send_message(
            "agent_1",
            "search_results",
            {"results": [asdict(result) for result in results]}
        )
    
    async def handle_gap_search(self, content: Dict[str, Any]) -> Message:
        """Handle gap-filling search requests"""
        queries = content.get("queries", [])
        original_notes = content.get("original_notes", "")
        gaps = content.get("gaps", [])
        
        logger.info(f"Gap search for {len(queries)} queries")
        
        # Perform searches for all gap queries
        all_results = []
        for query in queries:
            results = self.search_documents(query, max_results=3)
            all_results.extend(results)
        
        # Record gap search results
        from ..utils.recorder import get_global_recorder
        recorder = get_global_recorder()
        recorder.dump_search_results(f"Gap search: {queries}", all_results, "Agent3")
        
        # Send to synthesizer for gap filling
        return await self.send_message(
            "agent_2",
            "gap_fill_request",
            {
                "original_notes": original_notes,
                "additional_info": [asdict(result) for result in all_results],
                "gaps": gaps
            }
        )
    
    def search_documents(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search documents using FAISS similarity search"""
        if not self.index or not self.document_texts:
            logger.warning("No index available for search")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), max_results)
        
        # Create search results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.document_texts):
                results.append(SearchResult(
                    text=self.document_texts[idx],
                    score=float(score),
                    metadata={
                        "document_index": int(idx),
                        "rank": i + 1,
                        "query": query
                    }
                ))
        
        logger.info(f"Found {len(results)} results for query: '{query}'")
        return results
