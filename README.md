# AI Notes Maker - Multi-Agent Note Taking System

A modular, multi-agent system for generating comprehensive notes on any topic using AI. The system uses specialized agents to plan research, retrieve relevant content, and synthesize high-quality notes.

## 🚀 Features

- **Multi-Agent Architecture**: Specialized agents for planning, retrieval, and synthesis
- **Document Intelligence**: Automatically processes PDF, TXT, MD, and DOCX files
- **Configurable Output**: Customizable depth, audience, and format
- **Interactive & Batch Modes**: CLI interface for both interactive and automated usage
- **Comprehensive Logging**: Built-in recording and monitoring capabilities
- **Modular Design**: Clean, maintainable codebase with proper separation of concerns

## Architecture

The system consists of three main agents:

1. **Agent 1 (Planner)**: Strategic planning and orchestration
2. **Agent 2 (Synthesizer)**: Content synthesis and note writing
3. **Agent 3 (Retriever)**: Information retrieval using vector search

## Installation

```bash
# Clone the repository
git clone https://github.com/yash3056/notesmaker.git
cd notesmaker

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. Place your documents in the `documents/` folder
2. Run the system:

```python
from src.main import create_notes

notes = await create_notes(
    topic="your topic here",
    requirements={
        "depth": "comprehensive",
        "include_examples": True,
        "format": "structured",
        "audience": "students"
    }
)
```

## 📁 Project Structure

```
notesmaker/
├── config/                 # Configuration management
│   ├── __init__.py
│   └── settings.py        # Dataclass-based configuration
├── src/                   # Main source code
│   ├── agents/           # Specialized AI agents
│   │   ├── __init__.py
│   │   ├── base_agent.py     # Abstract base agent
│   │   ├── planner.py        # Research planning agent
│   │   ├── retriever.py      # Content retrieval agent
│   │   └── synthesizer.py    # Note synthesis agent
│   ├── core/             # Core system components
│   │   ├── __init__.py
│   │   ├── data_structures.py # Core data models
│   │   ├── document_loader.py # Document processing
│   │   ├── llm_wrapper.py     # LLM abstraction layer
│   │   └── system.py          # Main system orchestration
│   └── utils/            # Utility functions
│       ├── __init__.py
│       ├── helpers.py        # Helper functions
│       └── recorder.py       # Logging and monitoring
├── tests/                # Test suite
│   ├── test_agents/
│   ├── test_core/
│   └── test_utils/
├── examples/             # Usage examples
│   ├── basic_usage.py
│   └── advanced_usage.py
├── backup_original/      # Original monolithic files
├── documents/            # Source documents directory
├── main.py              # Main entry point
├── setup.py             # Package installation
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## 🛠️ Installation

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/yash3056/notesmaker.git
cd notesmaker
pip install -r requirements.txt
```

### 2. Environment Setup

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:
- Set your preferred LLM model
- Configure API keys if using external services
- Adjust system parameters

### 3. Install as Package (Optional)

```bash
pip install -e .
```

## 📖 Usage

### Interactive Mode

```bash
python main.py --interactive
```

The system will prompt you for:
- Topic for note generation
- Depth level (basic/intermediate/comprehensive)
- Target audience (students/professionals/general)
- Whether to include examples

### Batch Mode

```bash
python main.py --topic "Machine Learning" --depth comprehensive --audience students
```

### Command Line Options

```bash
python main.py [options]

Options:
  --topic TOPIC                 Topic for note generation
  --depth LEVEL                 Depth: basic, intermediate, comprehensive
  --audience AUDIENCE           Target: students, professionals, general
  --examples / --no-examples    Include examples (default: yes)
  --interactive, -i             Run in interactive mode
  --model MODEL                 LLM model to use
```

## 📚 Adding Documents

Place your source documents in the `documents/` directory. Supported formats:

- **Text files**: `.txt`, `.md`
- **PDF files**: `.pdf` (requires pypdf)
- **Word documents**: `.docx` (requires python-docx)

The system will automatically:
1. Scan the documents directory
2. Load and process all supported files
3. Build a searchable index
4. Use relevant content during note generation

## 🧩 Architecture

### Core Components

- **MultiAgentSystem**: Main orchestrator that coordinates all agents
- **DocumentLoader**: Handles loading and processing of various document formats
- **LLMWrapper**: Provides abstraction layer for different language models
- **Data Structures**: Type-safe models for messages, plans, and configurations

### Agent System

1. **PlannerAgent**: Analyzes the topic and creates a structured research plan
2. **RetrieverAgent**: Searches documents and retrieves relevant content
3. **SynthesizerAgent**: Combines research into comprehensive, well-formatted notes

## 🔧 Development

### Running Tests

```bash
pytest tests/
```

### Adding New Agents

1. Create new agent file in `src/agents/`
2. Inherit from `BaseAgent`
3. Implement required methods
4. Add to `src/agents/__init__.py`
5. Register in the `MultiAgentSystem`

## 🔄 Migration from Monolithic Version

This project was refactored from two monolithic scripts (`aI_agent_2.py` and `record.py`) into a clean, modular architecture. The original files are preserved in `backup_original/` for reference.

### Key Improvements

- **Modular Design**: Separated concerns into logical modules
- **Type Safety**: Added comprehensive type hints and Pydantic models
- **Better Testing**: Proper test structure with pytest
- **Configuration Management**: Centralized, dataclass-based configuration
- **Package Structure**: Proper Python package with setup.py
- **Enhanced Documentation**: Comprehensive README and code documentation
