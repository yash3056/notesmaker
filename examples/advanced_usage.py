"""
Advanced usage example for the AI Multi-Agent Note Taking System

This example demonstrates advanced features like configuration management,
multiple document sources, custom synthesis settings, and metrics monitoring.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.system import MultiAgentSystem
from src.core.data_structures import SynthesisConfig
from src.utils.helpers import ensure_directory_exists, Timer, get_memory_usage
from src.utils.recorder import get_global_recorder, create_recorder
from config.settings import get_settings, setup_logging, create_settings


class AdvancedNoteGenerator:
    """Advanced note generator with monitoring and configuration"""
    
    def __init__(self, config_file: str = None):
        self.settings = create_settings(config_file or ".env")
        self.system = None
        self.metrics = {}
        
        # Setup logging with config
        setup_logging(self.settings.logging)
        self.logger = logging.getLogger(__name__)
    
    async def initialize_system(self):
        """Initialize the multi-agent system with advanced configuration"""
        self.logger.info("Initializing advanced multi-agent system...")
        
        # Create custom synthesis configuration
        synthesis_config = SynthesisConfig(
            main_synthesis_tokens=self.settings.agents.main_synthesis_tokens,
            chunked_synthesis_tokens=self.settings.agents.chunked_synthesis_tokens,
            final_organization_tokens=self.settings.agents.final_organization_tokens,
            gap_filling_tokens=self.settings.agents.gap_filling_tokens,
            batch_size=self.settings.agents.batch_size,
            chunked_threshold=self.settings.agents.chunked_synthesis_threshold
        )
        
        # Initialize system with configuration
        self.system = MultiAgentSystem(
            documents_folder=self.settings.paths.documents_folder,
            model_name=self.settings.model.name,
            config=self.settings
        )
        
        # Set custom recorder if needed
        if self.settings.recording.enabled:
            session_name = time.strftime(self.settings.recording.session_name_format)
            custom_recorder = create_recorder(
                session_name=session_name,
                truncate_content=not self.settings.logging.enable_full_logging
            )
            self.system.recorder = custom_recorder
        
        self.logger.info("System initialized successfully")
    
    async def load_multiple_document_sources(self, sources: list):
        """Load documents from multiple sources"""
        self.logger.info(f"Loading documents from {len(sources)} sources...")
        
        for source_path in sources:
            if Path(source_path).exists():
                self.system.add_documents_from_folder(source_path)
                self.logger.info(f"Loaded documents from: {source_path}")
            else:
                self.logger.warning(f"Source path not found: {source_path}")
    
    def monitor_system_performance(self) -> Dict[str, Any]:
        """Monitor system performance metrics"""
        metrics = {
            "timestamp": time.time(),
            "memory": get_memory_usage(),
            "system_metrics": self.system.get_system_metrics() if self.system else None
        }
        
        self.metrics[time.time()] = metrics
        return metrics
    
    async def generate_notes_with_monitoring(self, topics: list, batch_mode: bool = False):
        """Generate notes with comprehensive monitoring"""
        results = {}
        
        for i, topic_config in enumerate(topics):
            topic = topic_config.get("topic", "")
            requirements = topic_config.get("requirements", {})
            
            self.logger.info(f"Processing topic {i+1}/{len(topics)}: {topic}")
            
            # Monitor performance
            start_metrics = self.monitor_system_performance()
            
            with Timer(f"Note generation for '{topic}'", self.logger.info):
                try:
                    # Generate notes
                    final_notes = await self.system.create_notes(topic, requirements)
                    
                    # Monitor post-generation metrics
                    end_metrics = self.monitor_system_performance()
                    
                    # Store results
                    results[topic] = {
                        "success": not final_notes.startswith("Note generation failed"),
                        "notes": final_notes,
                        "length": len(final_notes),
                        "start_metrics": start_metrics,
                        "end_metrics": end_metrics,
                        "requirements": requirements
                    }
                    
                    if results[topic]["success"]:
                        # Save individual notes
                        output_file = f"notes_{topic.replace(' ', '_').lower()}_{int(time.time())}.md"
                        self.save_notes_with_metadata(final_notes, output_file, topic_config)
                        results[topic]["output_file"] = output_file
                        
                        self.logger.info(f"✅ Successfully generated notes for '{topic}'")
                    else:
                        self.logger.error(f"❌ Failed to generate notes for '{topic}': {final_notes}")
                
                except Exception as e:
                    self.logger.error(f"❌ Error generating notes for '{topic}': {e}")
                    results[topic] = {
                        "success": False,
                        "error": str(e),
                        "notes": "",
                        "length": 0
                    }
            
            # Add delay between topics if in batch mode
            if batch_mode and i < len(topics) - 1:
                await asyncio.sleep(2)
        
        return results
    
    def save_notes_with_metadata(self, notes: str, filename: str, config: Dict[str, Any]):
        """Save notes with comprehensive metadata"""
        metadata = {
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "topic": config.get("topic", ""),
            "requirements": config.get("requirements", {}),
            "system_config": {
                "model_name": self.settings.model.name,
                "max_tokens": self.settings.model.max_tokens,
                "agent_config": {
                    "batch_size": self.settings.agents.batch_size,
                    "chunked_threshold": self.settings.agents.chunked_synthesis_threshold
                }
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {config.get('topic', 'Generated Notes')}\n\n")
            f.write("<!-- Generated by AI Multi-Agent Note Taking System -->\n")
            f.write(f"<!-- Generation Time: {metadata['generation_time']} -->\n")
            f.write(f"<!-- Model: {metadata['system_config']['model_name']} -->\n\n")
            f.write("## Configuration\n\n")
            f.write(f"- **Topic**: {metadata['topic']}\n")
            f.write(f"- **Requirements**: {metadata['requirements']}\n")
            f.write(f"- **Model**: {metadata['system_config']['model_name']}\n\n")
            f.write("## Generated Notes\n\n")
            f.write(notes)
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a performance report"""
        total_topics = len(results)
        successful = sum(1 for r in results.values() if r.get("success", False))
        failed = total_topics - successful
        
        total_length = sum(r.get("length", 0) for r in results.values())
        avg_length = total_length / total_topics if total_topics > 0 else 0
        
        report = f"""
# Performance Report

## Summary
- **Total Topics**: {total_topics}
- **Successful**: {successful}
- **Failed**: {failed}
- **Success Rate**: {(successful/total_topics*100):.1f}%

## Content Statistics
- **Total Characters Generated**: {total_length:,}
- **Average Length per Topic**: {avg_length:,.0f} characters

## Individual Results
"""
        
        for topic, result in results.items():
            status = "✅" if result.get("success", False) else "❌"
            length = result.get("length", 0)
            report += f"- {status} **{topic}**: {length:,} characters\n"
        
        return report
    
    def shutdown(self):
        """Clean shutdown with final reporting"""
        if self.system:
            self.logger.info("Shutting down system...")
            self.system.shutdown()
            
        # Save performance metrics
        if self.metrics:
            from ..src.utils.helpers import save_json_safely
            save_json_safely(self.metrics, "performance_metrics.json")
            self.logger.info("Performance metrics saved to performance_metrics.json")


async def advanced_example():
    """Advanced example with multiple topics and monitoring"""
    
    print("="*60)
    print("AI Multi-Agent Note Taking System - Advanced Example")
    print("="*60)
    
    # Initialize advanced generator
    generator = AdvancedNoteGenerator()
    
    try:
        # Initialize system
        await generator.initialize_system()
        
        # Load multiple document sources (if they exist)
        sources = ["documents", "additional_docs", "research_papers"]
        await generator.load_multiple_document_sources(sources)
        
        # Define multiple topics with different requirements
        topics = [
            {
                "topic": "Deep Learning Fundamentals",
                "requirements": {
                    "depth": "advanced",
                    "include_examples": True,
                    "format": "academic",
                    "audience": "researchers"
                }
            },
            {
                "topic": "Natural Language Processing",
                "requirements": {
                    "depth": "intermediate",
                    "include_examples": True,
                    "format": "tutorial",
                    "audience": "practitioners"
                }
            },
            {
                "topic": "Computer Vision Applications",
                "requirements": {
                    "depth": "beginner",
                    "include_examples": True,
                    "format": "overview",
                    "audience": "students"
                }
            }
        ]
        
        print(f"\nGenerating notes for {len(topics)} topics...")
        
        # Generate notes with monitoring
        results = await generator.generate_notes_with_monitoring(topics, batch_mode=True)
        
        # Generate and display performance report
        report = generator.generate_performance_report(results)
        print(report)
        
        # Save performance report
        with open("performance_report.md", 'w') as f:
            f.write(report)
        
        print("\n✅ Advanced example completed successfully!")
        print("📊 Performance report saved to: performance_report.md")
    
    except Exception as e:
        print(f"\n❌ Error in advanced example: {e}")
        logging.exception("Detailed error information:")
    
    finally:
        generator.shutdown()


def main():
    """Main function to run the advanced example"""
    try:
        asyncio.run(advanced_example())
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
