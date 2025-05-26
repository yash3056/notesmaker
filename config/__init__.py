"""
Configuration package for the AI Multi-Agent Note Taking System
"""

from .settings import (
    ModelConfig,
    PathConfig, 
    AgentConfig,
    SearchConfig,
    LoggingConfig,
    RecordingConfig,
    Settings,
    get_settings,
    update_settings,
    setup_logging,
    default_settings
)

__all__ = [
    "ModelConfig",
    "PathConfig", 
    "AgentConfig",
    "SearchConfig",
    "LoggingConfig",
    "RecordingConfig",
    "Settings",
    "get_settings",
    "update_settings", 
    "setup_logging",
    "default_settings"
]
