"""
Configuration settings for the AI Multi-Agent Note Taking System
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    name: str = "Qwen/Qwen3-1.7B"
    max_tokens: int = 8196
    temperature: float = 0.7
    top_p: float = 0.8
    device: str = "auto"
    use_torch_compile: bool = False


@dataclass
class PathConfig:
    """Configuration for file paths"""
    documents_folder: str = "documents"
    records_folder: str = "records"
    models_cache_dir: str = "./models"
    
    def __post_init__(self):
        """Convert relative paths to absolute paths"""
        self.documents_folder = str(PROJECT_ROOT / self.documents_folder)
        self.records_folder = str(PROJECT_ROOT / self.records_folder)
        self.models_cache_dir = str(PROJECT_ROOT / self.models_cache_dir)


@dataclass
class AgentConfig:
    """Configuration for agent behavior"""
    chunked_synthesis_threshold: int = 10
    batch_size: int = 5
    max_search_results: int = 5
    main_synthesis_tokens: int = 8196
    chunked_synthesis_tokens: int = 4096
    final_organization_tokens: int = 8196
    gap_filling_tokens: int = 4096


@dataclass
class SearchConfig:
    """Configuration for search and retrieval"""
    embedding_model: str = "all-MiniLM-L6-v2"
    faiss_index_type: str = "FlatIP"  # FlatIP, FlatL2, HNSW
    normalize_embeddings: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_full_logging: bool = True
    truncate_content: bool = False


@dataclass
class RecordingConfig:
    """Configuration for session recording"""
    enabled: bool = True
    auto_save_json: bool = True
    session_name_format: str = "session_%Y%m%d_%H%M%S"


@dataclass
class Settings:
    """Main settings container"""
    model: ModelConfig
    paths: PathConfig
    agents: AgentConfig
    search: SearchConfig
    logging: LoggingConfig
    recording: RecordingConfig
    debug_mode: bool = False
    verbose_errors: bool = True
    profile_performance: bool = False


def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """Load environment variables from file"""
    env_vars = {}
    env_file_path = PROJECT_ROOT / env_path
    
    if env_file_path.exists():
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    
    return env_vars


def get_env_value(key: str, default: Any = None, env_vars: Dict[str, str] = None) -> Any:
    """Get environment variable with type conversion"""
    if env_vars is None:
        env_vars = {}
    
    value = env_vars.get(key) or os.environ.get(key, default)
    
    if value is None:
        return default
    
    # Convert string values to appropriate types
    if isinstance(default, bool):
        return str(value).lower() in ('true', '1', 'yes', 'on')
    elif isinstance(default, int):
        return int(value)
    elif isinstance(default, float):
        return float(value)
    else:
        return str(value)


def create_settings(env_file: str = ".env") -> Settings:
    """Create settings from environment variables and defaults"""
    env_vars = load_env_file(env_file)
    
    model_config = ModelConfig(
        name=get_env_value("DEFAULT_MODEL_NAME", "Qwen/Qwen3-1.7B", env_vars),
        max_tokens=get_env_value("MODEL_MAX_TOKENS", 8196, env_vars),
        temperature=get_env_value("MODEL_TEMPERATURE", 0.7, env_vars),
        top_p=get_env_value("MODEL_TOP_P", 0.8, env_vars),
        device=get_env_value("TORCH_DEVICE", "auto", env_vars),
        use_torch_compile=get_env_value("USE_TORCH_COMPILE", False, env_vars)
    )
    
    path_config = PathConfig(
        documents_folder=get_env_value("DOCUMENTS_FOLDER", "documents", env_vars),
        records_folder=get_env_value("RECORDS_FOLDER", "records", env_vars),
        models_cache_dir=get_env_value("MODELS_CACHE_DIR", "./models", env_vars)
    )
    
    agent_config = AgentConfig(
        chunked_synthesis_threshold=get_env_value("CHUNKED_SYNTHESIS_THRESHOLD", 10, env_vars),
        batch_size=get_env_value("BATCH_SIZE", 5, env_vars),
        max_search_results=get_env_value("MAX_SEARCH_RESULTS", 5, env_vars)
    )
    
    search_config = SearchConfig(
        embedding_model=get_env_value("EMBEDDING_MODEL", "all-MiniLM-L6-v2", env_vars),
        faiss_index_type=get_env_value("FAISS_INDEX_TYPE", "FlatIP", env_vars)
    )
    
    logging_config = LoggingConfig(
        level=get_env_value("LOG_LEVEL", "INFO", env_vars),
        format=get_env_value("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s", env_vars),
        enable_full_logging=get_env_value("ENABLE_FULL_LOGGING", True, env_vars),
        truncate_content=get_env_value("TRUNCATE_CONTENT", False, env_vars)
    )
    
    recording_config = RecordingConfig(
        enabled=get_env_value("RECORDING_ENABLED", True, env_vars),
        auto_save_json=get_env_value("AUTO_SAVE_JSON", True, env_vars),
        session_name_format=get_env_value("SESSION_NAME_FORMAT", "session_%Y%m%d_%H%M%S", env_vars)
    )
    
    return Settings(
        model=model_config,
        paths=path_config,
        agents=agent_config,
        search=search_config,
        logging=logging_config,
        recording=recording_config,
        debug_mode=get_env_value("DEBUG_MODE", False, env_vars),
        verbose_errors=get_env_value("VERBOSE_ERRORS", True, env_vars),
        profile_performance=get_env_value("PROFILE_PERFORMANCE", False, env_vars)
    )


def setup_logging(config: LoggingConfig):
    """Setup logging configuration"""
    level = getattr(logging, config.level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=config.format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(PROJECT_ROOT / "logs" / "notesmaker.log", mode='a')
        ]
    )
    
    # Create logs directory if it doesn't exist
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)


# Default settings instance
default_settings = create_settings()


def get_settings() -> Settings:
    """Get the current settings"""
    return default_settings


def update_settings(env_file: str = ".env") -> Settings:
    """Reload settings from environment file"""
    global default_settings
    default_settings = create_settings(env_file)
    return default_settings
