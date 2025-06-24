import logging
import os
import requests
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from pathlib import Path
from bs4 import BeautifulSoup
import time

from .base_agent import BaseAgent
from ..core.data_structures import Message, SearchResult

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

logger = logging.getLogger(__name__)

class Agent3_WebSearcher(BaseAgent):
    """Web Search Information Retrieval Specialist using Tavily"""
    
    def __init__(self, api_key: str = None): # type: ignore
        super().__init__("agent_3", "Web Search Retriever")
        
        # Initialize Tavily client
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("No Tavily API key provided. Set TAVILY_API_KEY environment variable or pass api_key parameter.")
            self.client = None
        else:
            if TAVILY_AVAILABLE:
                self.client = TavilyClient(self.api_key) # type: ignore
                logger.info("Tavily client initialized successfully")
            else:
                logger.error("tavily-python not installed. Install with: pip install tavily-python")
                self.client = None
        
        # Create directories for storing scraped content
        self.content_dir = Path("scraped_content")
        self.content_dir.mkdir(exist_ok=True)
        
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
        
        logger.info(f"Web searching for: '{query_text}' (max {max_results} results)")
        
        # Perform web search
        results = await self.search_web(query_text, max_results)
        
        # Record search results
        from ..utils.recorder import get_global_recorder
        recorder = get_global_recorder()
        recorder.dump_search_results(query_text, results, "Agent3_WebSearcher")
        
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
            results = await self.search_web(query, max_results=3)
            all_results.extend(results)
        
        # Record gap search results
        from ..utils.recorder import get_global_recorder
        recorder = get_global_recorder()
        recorder.dump_search_results(f"Gap search: {queries}", all_results, "Agent3_WebSearcher")
        
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
    
    async def search_web(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search the web using Tavily and scrape content"""
        if not self.client:
            logger.error("Tavily client not available")
            # Fallback for when Tavily is not available
            return await self._search_fallback(query, max_results)
        
        try:
            # Perform Tavily search
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )
            
            results = []
            web_results = response.get("results", [])
            
            for i, result in enumerate(web_results):
                url = result.get("url", "")
                title = result.get("title", "")
                content = result.get("content", "")
                score = result.get("score", 0.0)
                
                # Scrape full content from URL
                scraped_content = await self._scrape_url(url)
                
                # Use scraped content if available, fallback to Tavily content
                full_content = scraped_content if scraped_content else content
                
                # Save content to file for human reference
                await self._save_content_to_file(query, i, title, url, full_content)
                
                # Create search result
                search_result = SearchResult(
                    text=full_content,
                    score=float(score),
                    metadata={
                        "url": url,
                        "title": title,
                        "rank": i + 1,
                        "query": query,
                        "source": "web_search",
                        "scraped": bool(scraped_content)
                    }
                )
                
                results.append(search_result)
            
            logger.info(f"Found {len(results)} web search results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return []
    
    async def _scrape_url(self, url: str) -> str:
        """Scrape content from a URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.warning(f"Error scraping URL {url}: {e}")
            return ""
    
    async def _save_content_to_file(self, query: str, index: int, title: str, url: str, content: str):
        """Save scraped content to a file for human reference"""
        try:
            # Create safe filename
            safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_query = safe_query.replace(' ', '_')[:50]
            
            filename = f"{safe_query}_{index}_{int(time.time())}.txt"
            filepath = self.content_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Title: {title}\n")
                f.write(f"URL: {url}\n")
                f.write(f"Scraped at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(content)
            
            logger.info(f"Saved content to: {filepath}")
            
        except Exception as e:
            logger.warning(f"Error saving content to file: {e}")
    
    async def _search_fallback(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback search method when Tavily is not available"""
        logger.warning("Using fallback search - limited functionality")
        
        # This is a basic fallback that could be enhanced with other search APIs
        # For now, return empty results with a message
        fallback_result = SearchResult(
            text=f"Web search not available. Query was: {query}. Please ensure Tavily API key is configured.",
            score=0.0,
            metadata={
                "query": query,
                "source": "fallback",
                "error": "Tavily not available"
            }
        )
        
        return [fallback_result]
    
    async def search_beginner_topics(self, subject: str) -> List[SearchResult]:
        """Search for beginner-friendly topics when no syllabus is provided"""
        logger.info(f"Searching for beginner topics in: {subject}")
        
        beginner_queries = [
            f"introduction to {subject} for beginners",
            f"basic concepts of {subject}",
            f"fundamental principles of {subject}",
            f"getting started with {subject}",
            f"{subject} basics tutorial"
        ]
        
        all_results = []
        for query in beginner_queries:
            results = await self.search_web(query, max_results=2)
            all_results.extend(results)
        
        return all_results
