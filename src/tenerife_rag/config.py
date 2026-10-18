"""Configuración del modelo leída del entorno (.env)."""

import os
from dataclasses import asdict, dataclass

from agents import ModelSettings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


@dataclass
class LLMConfig:
    """Modelos y parámetros de generación, con valores por defecto."""

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
        """Lee la configuración de las variables de entorno (y del .env)."""
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
        """Cliente de OpenAI con la key de la configuración."""
        return OpenAI(api_key=self.api_key)

    def embeddings(self) -> OpenAIEmbeddings:
        """Modelo de embeddings para el índice."""
        return OpenAIEmbeddings(model=self.embedding_model, api_key=self.api_key)

    def model_settings(self) -> ModelSettings:
        """Parámetros de generación para el agente."""
        return ModelSettings(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
        )
