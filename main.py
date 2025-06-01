#!/usr/bin/env python3
"""
AI Notes Maker - Multi-Agent Note Taking System

This is the main entry point for the AI Notes Maker system.
It creates comprehensive notes on any topic using a multi-agent architecture.
"""

import asyncio
import logging
import argparse
import os
from pathlib import Path

from src.core.system import MultiAgentSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_notes_maker.log')
    ]
)

logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ['records', 'logs', 'scraped_content']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def check_scraped_content_folder():
    """Check and create scraped content folder"""
    content_path = Path("scraped_content")
    if not content_path.exists():
        content_path.mkdir()
        logger.info("Created scraped_content folder for web content storage")
    return True


def check_tavily_api_key():
    """Check if Tavily API key is available"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("\n" + "="*60)
        print("TAVILY API KEY NOT SET")
        print("="*60)
        print("This system uses web search to find information on any topic.")
        print("For optimal functionality, please set your Tavily API key:")
        print("1. Get an API key from: https://tavily.com")
        print("2. Set environment variable: export TAVILY_API_KEY='your-key-here'")
        print("3. Or add it to your .bashrc/.zshrc file")
        print("\nThe system will work in fallback mode without the API key.")
        print("Fallback mode provides basic functionality with limited search.")
        print("="*60)
        return False
    logger.info("Tavily API key found")
    return True


async def create_notes_interactive():
    """Interactive mode for creating notes"""
    print("\n" + "="*60)
    print("AI NOTES MAKER - INTERACTIVE MODE")
    print("="*60)
    
    # Get topic from user
    topic = input("\nEnter the topic for your notes: ").strip()
    if not topic:
        print("No topic provided. Exiting...")
        return
    
    # Get optional requirements
    print("\nOptional requirements (press Enter to skip):")
    depth = input("Depth (basic/intermediate/comprehensive): ").strip() or "comprehensive"
    include_examples = input("Include examples? (y/n): ").strip().lower() in ['y', 'yes', '1', 'true']
    audience = input("Target audience (students/professionals/general): ").strip() or "students"
    
    requirements = {
        "depth": depth,
        "include_examples": include_examples,
        "format": "structured",
        "audience": audience
    }
    
    print(f"\nCreating notes on: {topic}")
    print(f"Requirements: {requirements}")
    print("\nStarting note generation... (this may take a few minutes)")
    print("Searching web sources, generating content, and synthesizing notes...")
    
    # Create the system and generate notes
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    system = MultiAgentSystem(tavily_api_key=tavily_api_key)
    
    # Print system status
    print(f"\nWeb search system initialized")
    if tavily_api_key:
        print("✅ Tavily API available for enhanced web search")
    else:
        print("⚠️  Limited search mode (no Tavily API key)")
    
    try:
        notes = await system.create_notes(topic, requirements)
        
        if notes and not notes.startswith("Note generation failed"):
            print("\n✅ Note generation completed successfully!")
            
            # Save notes to file
            timestamp = asyncio.get_event_loop().time()
            filename = f"notes_{topic.replace(' ', '_').replace('/', '_')}_{int(timestamp)}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Notes: {topic}\n\n")
                f.write(f"Generated on: {asyncio.get_event_loop().time()}\n\n")
                f.write(notes)
            
            print(f"📝 Notes saved to: {filename}")
        else:
            print("\n❌ Note generation failed!")
            if notes:
                print(f"Error: {notes}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Note generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during note generation: {e}")
        logger.error(f"Error in interactive mode: {e}", exc_info=True)


async def create_notes_batch(topic: str, requirements: dict):
    """Batch mode for creating notes"""
    logger.info(f"Creating notes in batch mode for topic: {topic}")
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    system = MultiAgentSystem(tavily_api_key=tavily_api_key)
    
    # Print system status
    logger.info("Web search system initialized")
    if tavily_api_key:
        logger.info("Tavily API available for enhanced web search")
    else:
        logger.info("Limited search mode (no Tavily API key)")
    
    try:
        notes = await system.create_notes(topic, requirements)
        
        if notes and not notes.startswith("Note generation failed"):
            # Save notes to file
            timestamp = asyncio.get_event_loop().time()
            filename = f"notes_{topic.replace(' ', '_').replace('/', '_')}_{int(timestamp)}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Notes: {topic}\n\n")
                f.write(f"Generated on: {asyncio.get_event_loop().time()}\n\n")
                f.write(notes)
            
            print(f"✅ Notes generated and saved to: {filename}")
            return filename
        else:
            print("❌ Note generation failed!")
            if notes:
                print(f"Error: {notes}")
            return None
    
    except Exception as e:
        logger.error(f"Error in batch mode: {e}", exc_info=True)
        print(f"❌ Error during note generation: {e}")
        return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Notes Maker - Multi-Agent Note Taking System")
    parser.add_argument("--topic", type=str, help="Topic for note generation")
    parser.add_argument("--depth", type=str, default="comprehensive", 
                       choices=["basic", "intermediate", "comprehensive"],
                       help="Depth of notes")
    parser.add_argument("--audience", type=str, default="students",
                       choices=["students", "professionals", "general"],
                       help="Target audience")
    parser.add_argument("--examples", action="store_true", default=True,
                       help="Include examples in notes")
    parser.add_argument("--no-examples", action="store_false", dest="examples",
                       help="Don't include examples in notes")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                       help="LLM model to use")
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    check_scraped_content_folder()
    api_key_exists = check_tavily_api_key()
    
    if not api_key_exists:
        response = input("\nContinue with limited search functionality? (y/n): ").strip().lower()
        if response not in ['y', 'yes', '1', 'true']:
            print("Exiting...")
            return
    
    # Determine mode
    if args.interactive or not args.topic:
        # Interactive mode
        asyncio.run(create_notes_interactive())
    else:
        # Batch mode
        requirements = {
            "depth": args.depth,
            "include_examples": args.examples,
            "format": "structured",
            "audience": args.audience
        }
        
        asyncio.run(create_notes_batch(args.topic, requirements))


if __name__ == "__main__":
    main()
