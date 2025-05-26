"""
Test to verify all imports work correctly
"""

import pytest
import sys
import importlib
from pathlib import Path


class TestImports:
    """Test cases for verifying all module imports"""
    
    def test_config_imports(self):
        """Test that all config modules can be imported"""
        try:
            import config
            import config.settings
            from config.settings import Settings, ModelConfig, PathConfig, AgentConfig
            from config.settings import SearchConfig, LoggingConfig, RecordingConfig
        except ImportError as e:
            pytest.fail(f"Failed to import config modules: {e}")
    
    def test_core_imports(self):
        """Test that all core modules can be imported"""
        try:
            import src.core
            import src.core.data_structures
            import src.core.document_loader
            import src.core.llm_wrapper
            import src.core.system
            
            from src.core.data_structures import Message, Plan, SearchQuery, SearchResult
            from src.core.data_structures import SynthesisConfig, SystemMetrics, AgentStatus
            from src.core.document_loader import DocumentLoader
            from src.core.llm_wrapper import BaseLLM, QwenLLM
            from src.core.system import MultiAgentSystem
        except ImportError as e:
            pytest.fail(f"Failed to import core modules: {e}")
    
    def test_agents_imports(self):
        """Test that all agent modules can be imported"""
        try:
            import src.agents
            import src.agents.base_agent
            import src.agents.planner
            import src.agents.retriever
            import src.agents.synthesizer
            
            from src.agents.base_agent import BaseAgent
            from src.agents.planner import Agent1_Planner
            from src.agents.retriever import Agent3_Retriever
            from src.agents.synthesizer import Agent2_Synthesizer
        except ImportError as e:
            pytest.fail(f"Failed to import agent modules: {e}")
    
    def test_utils_imports(self):
        """Test that all utility modules can be imported"""
        try:
            import src.utils
            import src.utils.helpers
            import src.utils.recorder
            
            from src.utils.helpers import ensure_directory_exists, safe_filename
            from src.utils.helpers import truncate_text, save_json_safely, load_json_safely
            from src.utils.helpers import validate_topic, extract_keywords, Timer
            from src.utils.recorder import Record, get_global_recorder, set_global_recorder
        except ImportError as e:
            pytest.fail(f"Failed to import utils modules: {e}")
    
    def test_main_module_import(self):
        """Test that the main module can be imported"""
        try:
            import main
        except ImportError as e:
            pytest.fail(f"Failed to import main module: {e}")
    
    def test_examples_imports(self):
        """Test that example modules can be imported"""
        try:
            import examples
            import examples.basic_usage
            import examples.advanced_usage
        except ImportError as e:
            pytest.fail(f"Failed to import example modules: {e}")
    
    def test_all_python_files_importable(self):
        """Test that all Python files in the project can be imported"""
        project_root = Path(__file__).parent.parent
        python_files = []
        
        # Collect all Python files (excluding __pycache__ and test files)
        for py_file in project_root.rglob("*.py"):
            # Skip test files, __pycache__, and setup files
            if (
                "__pycache__" not in str(py_file) and
                "test_" not in py_file.name and
                py_file.name not in ["setup.py", "validate_system.py"] and
                "backup_original" not in str(py_file)
            ):
                python_files.append(py_file)
        
        failed_imports = []
        
        for py_file in python_files:
            # Convert file path to module name
            relative_path = py_file.relative_to(project_root)
            if relative_path.name == "__init__.py":
                module_parts = relative_path.parent.parts
            else:
                module_parts = relative_path.with_suffix("").parts
            
            if module_parts:
                module_name = ".".join(module_parts)
                
                try:
                    # Add project root to path if not already there
                    if str(project_root) not in sys.path:
                        sys.path.insert(0, str(project_root))
                    
                    importlib.import_module(module_name)
                except Exception as e:
                    failed_imports.append(f"{module_name}: {str(e)}")
        
        if failed_imports:
            pytest.fail(f"Failed to import modules:\n" + "\n".join(failed_imports))
    
    def test_circular_imports(self):
        """Test for circular import issues"""
        # This test tries to import all modules in sequence to catch circular imports
        modules_to_test = [
            "config.settings",
            "src.core.data_structures",
            "src.core.document_loader", 
            "src.core.llm_wrapper",
            "src.agents.base_agent",
            "src.agents.planner",
            "src.agents.retriever", 
            "src.agents.synthesizer",
            "src.core.system",
            "src.utils.helpers",
            "src.utils.recorder"
        ]
        
        failed_modules = []
        
        for module_name in modules_to_test:
            try:
                # Clear module from cache to test fresh import
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                importlib.import_module(module_name)
            except Exception as e:
                failed_modules.append(f"{module_name}: {str(e)}")
        
        if failed_modules:
            pytest.fail(f"Circular import or other import issues detected:\n" + "\n".join(failed_modules))
    
    def test_required_dependencies(self):
        """Test that all required dependencies are available"""
        required_packages = [
            "torch",
            "transformers", 
            "sentence_transformers",
            "faiss",
            "numpy",
            "dataclasses"  # Should be built-in for Python 3.7+
        ]
        
        # Special handling for PDF libraries
        pdf_packages = ["pypdf", "PyPDF2"]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == "faiss":
                    # FAISS can be installed as faiss-cpu or faiss-gpu
                    try:
                        import faiss
                    except ImportError:
                        import faiss_cpu as faiss
                else:
                    importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        # Check for PDF library (either pypdf or PyPDF2)
        pdf_available = False
        for pdf_pkg in pdf_packages:
            try:
                importlib.import_module(pdf_pkg)
                pdf_available = True
                break
            except ImportError:
                continue
        
        if not pdf_available:
            missing_packages.append("pypdf or PyPDF2")
        
        if missing_packages:
            pytest.fail(f"Missing required packages: {', '.join(missing_packages)}")
    
    def test_optional_dependencies(self):
        """Test optional dependencies and warn if missing"""
        optional_packages = [
            "jupyter",
            "matplotlib", 
            "pandas",
            "tqdm"
        ]
        
        missing_optional = []
        
        for package in optional_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_optional.append(package)
        
        if missing_optional:
            print(f"Warning: Optional packages not available: {', '.join(missing_optional)}")
            # Don't fail the test for optional dependencies
