"""
Configuration settings for Smart CV Filter application.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration."""
    
    # API Keys
    openai_api_key: Optional[str] = None
    
    # Database
    chroma_db_path: str = "./chroma_db"
    
    # Analysis Settings
    max_keywords: int = 30
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Scoring Weights
    keyword_weight: float = 0.3
    skill_weight: float = 0.4
    similarity_weight: float = 0.3
    
    # LLM Settings
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 500
    
    # File Upload Settings
    max_file_size_mb: int = 10
    allowed_extensions: list = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.allowed_extensions is None:
            self.allowed_extensions = ['pdf', 'docx', 'txt']
        
        # Load from environment if available
        if not self.openai_api_key:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if os.getenv('CHROMA_DB_PATH'):
            self.chroma_db_path = os.getenv('CHROMA_DB_PATH')
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            chroma_db_path=os.getenv('CHROMA_DB_PATH', './chroma_db'),
        )
    
    def validate(self) -> bool:
        """Validate configuration."""
        # Check weights sum to 1.0
        total_weight = self.keyword_weight + self.skill_weight + self.similarity_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight}")
        
        # Check file size is positive
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
        
        return True


# Default configuration instance
default_config = Config()
