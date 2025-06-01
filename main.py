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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    directories = ['documents', 'records', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def check_documents_folder():
    """Check if documents folder has content"""
    docs_path = Path("documents")
    if not docs_path.exists():
        docs_path.mkdir()
        logger.info("Created documents folder")
        return False
    
    # Check for supported file types
    supported_extensions = ['.txt', '.md', '.pdf', '.docx']
    files = []
    for ext in supported_extensions:
        files.extend(list(docs_path.glob(f'**/*{ext}')))
    
    if not files:
        logger.warning("No documents found in the documents folder!")
        print("\n" + "="*60)
        print("NO DOCUMENTS FOUND!")
        print("="*60)
        print("Please add documents to the 'documents' folder:")
        print("- .txt files (plain text)")
        print("- .md files (markdown)")
        print("- .pdf files (requires: pip install pypdf)")
        print("- .docx files (requires: pip install python-docx)")
        print("\nThe system can still work but will have limited content to search.")
        print("="*60)
        return False
    
    logger.info(f"Found {len(files)} documents in the documents folder")
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
    print("Processing documents, generating content, and synthesizing notes...")
    
    # Create the system and generate notes
    system = MultiAgentSystem()
    
    # Print document statistics
    stats = system.get_document_stats()
    print(f"\nLoaded {stats['total_documents']} document chunks")
    print(f"Index built: {stats['index_built']}")
    
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
    
    system = MultiAgentSystem()
    
    # Print document statistics
    stats = system.get_document_stats()
    logger.info(f"Loaded {stats['total_documents']} document chunks")
    
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
    has_documents = check_documents_folder()
    
    if not has_documents:
        response = input("\nContinue without documents? (y/n): ").strip().lower()
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
