from dataclasses import asdict, dataclass
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from agents import ModelSettings


@dataclass
class LLMConfig:
    api_key: str | None = None
    generation_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.2
    top_p: float = 1.0
    max_output_tokens: int = 700
    history_token_budget: int = 2800
    max_turns: int = 4
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_dotenv()
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            generation_model=os.getenv("OPENAI_GENERATION_MODEL", cls.generation_model),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", cls.embedding_model),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", cls.temperature)),
            top_p=float(os.getenv("OPENAI_TOP_P", cls.top_p)),
            max_output_tokens=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", cls.max_output_tokens)),
            history_token_budget=int(os.getenv("OPENAI_HISTORY_TOKEN_BUDGET", cls.history_token_budget)),
            max_turns=int(os.getenv("OPENAI_MAX_TURNS", cls.max_turns)),
        )
        
    def public_dict(self) -> dict:
        """Configuración para mostrar con la key enmascarada"""
        
        data = asdict(self)
        data["api_key"] = f"{self.api_key[:5]}***" if self.api_key else None
        return data
    
    def client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key)
    
    def embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=self.api_key
        )
    
    def model_settings(self) -> ModelSettings:
        return ModelSettings(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
        )