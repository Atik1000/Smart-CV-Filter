"""
LLM Provider Integration
Supports multiple LLM providers for CV analysis
"""

from typing import Optional, Dict
import os


class LLMProvider:
    """Manages multiple LLM provider integrations."""
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize LLM provider.
        
        Args:
            provider: LLM provider name (openai, anthropic, google, groq, ollama)
            api_key: API key for the provider
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client."""
        try:
            if self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.model = "gpt-3.5-turbo"
                
            elif self.provider == "anthropic":
                try:
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=self.api_key)
                    self.model = "claude-3-haiku-20240307"  # Fast and affordable
                except ImportError:
                    print("Install anthropic: pip install anthropic")
                    
            elif self.provider == "google":
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    self.client = genai.GenerativeModel('gemini-pro')
                    self.model = "gemini-pro"
                except ImportError:
                    print("Install google-generativeai: pip install google-generativeai")
                    
            elif self.provider == "groq":
                try:
                    from groq import Groq
                    self.client = Groq(api_key=self.api_key)
                    self.model = "mixtral-8x7b-32768"  # Fast and free tier available
                except ImportError:
                    print("Install groq: pip install groq")
                    
            elif self.provider == "ollama":
                # Ollama runs locally - no API key needed
                try:
                    import ollama
                    self.client = ollama
                    self.model = "llama2"  # Default model
                except ImportError:
                    print("Install ollama: pip install ollama")
                    
        except Exception as e:
            print(f"Error initializing {self.provider}: {str(e)}")
            self.client = None
    
    def generate_analysis(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate analysis using the configured LLM provider.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM generated text
        """
        if not self.client:
            return "LLM not available. Please check your configuration."
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert HR consultant providing CV feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            
            elif self.provider == "google":
                response = self.client.generate_content(prompt)
                return response.text
            
            elif self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert HR consultant providing CV feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            
            elif self.provider == "ollama":
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt
                )
                return response['response']
            
            else:
                return f"Provider {self.provider} not supported yet."
                
        except Exception as e:
            return f"Error generating analysis with {self.provider}: {str(e)}"
    
    @staticmethod
    def get_available_providers() -> Dict[str, Dict]:
        """
        Get information about available LLM providers.
        
        Returns:
            Dict of provider information
        """
        return {
            "openai": {
                "name": "OpenAI GPT",
                "models": ["gpt-3.5-turbo", "gpt-4"],
                "requires_api_key": True,
                "free_tier": False,
                "description": "Most popular, reliable AI"
            },
            "anthropic": {
                "name": "Anthropic Claude",
                "models": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
                "requires_api_key": True,
                "free_tier": True,
                "description": "Fast, accurate, free tier available"
            },
            "google": {
                "name": "Google Gemini",
                "models": ["gemini-pro"],
                "requires_api_key": True,
                "free_tier": True,
                "description": "Google's AI, generous free tier"
            },
            "groq": {
                "name": "Groq (Fast AI)",
                "models": ["mixtral-8x7b", "llama2-70b"],
                "requires_api_key": True,
                "free_tier": True,
                "description": "Extremely fast, free tier"
            },
            "ollama": {
                "name": "Ollama (Local)",
                "models": ["llama2", "mistral", "codellama"],
                "requires_api_key": False,
                "free_tier": True,
                "description": "100% free, runs locally, private"
            }
        }
