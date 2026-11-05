"""Fábrica del chat model de LangChain según el proveedor configurado en el entorno.

Retoma el patrón de la pre-entrega 1 (`config_from_env` + `ModelConfig`): las variables de
entorno se validan con Pydantic y el resto del código no sabe qué proveedor hay detrás.
Se usan los chat models de LangChain (y no el cliente propio del M1) porque
`with_structured_output` solo existe en ellos.
"""

import logging
import os
from functools import cache
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel, Field, SecretStr, model_validator

logger = logging.getLogger(__name__)

Provider = Literal["gemini", "openai", "anthropic"]
ThinkingLevel = Literal["low", "medium", "high"]

# IDs reales y vigentes de cada API (ver tabla "Modelos por defecto" en el README).
DEFAULT_MODELS: dict[str, str] = {
    # Flash Lite: en el plan gratuito tiene 500 requests/día (los Flash, 20).
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
}

API_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


class LLMSettings(BaseModel):
    provider: Provider = "gemini"
    model: str | None = Field(default=None, description="Si es None se usa DEFAULT_MODELS[provider]")
    api_key: SecretStr | None = Field(default=None, repr=False)
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="None = valor por defecto del proveedor"
    )
    # Solo aplica a Gemini 3: los modelos que "piensan" gastan parte de max_tokens razonando.
    thinking_level: ThinkingLevel = "low"
    max_tokens: int = Field(default=2048, gt=0, le=32_000)
    timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(
        default=2, ge=0, le=10, description="Reintentos de la cadena (formato y errores transitorios); el SDK no reintenta"
    )
    # Cupo de requests por minuto (plan gratuito de Gemini: 15 en Flash Lite, 5 en Flash). None = sin límite.
    requests_per_minute: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def completar_y_verificar(self) -> "LLMSettings":
        if not self.model:
            self.model = DEFAULT_MODELS[self.provider]
        if self.api_key is None or not self.api_key.get_secret_value():
            nombres = " o ".join(API_KEY_ENVS[self.provider])
            raise ValueError("falta la API key de " + self.provider + ": definí " + nombres + " en el .env")
        return self


def settings_from_env(provider: str | None = None, **overrides: Any) -> LLMSettings:
    """Arma LLMSettings desde variables de entorno (ver .env.example); `overrides` tiene prioridad."""
    provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    api_key = next((os.getenv(n) for n in API_KEY_ENVS.get(provider, ()) if os.getenv(n)), None)
    data: dict[str, Any] = {
        "provider": provider,
        "model": os.getenv(provider.upper() + "_MODEL"),
        "api_key": api_key,
        "temperature": os.getenv("LLM_TEMPERATURE"),
        "thinking_level": os.getenv("LLM_THINKING_LEVEL"),
        "max_tokens": os.getenv("LLM_MAX_TOKENS"),
        "timeout": os.getenv("LLM_TIMEOUT"),
        "max_retries": os.getenv("LLM_MAX_RETRIES"),
        "requests_per_minute": os.getenv("LLM_REQUESTS_PER_MINUTE"),
    }
    data = {k: v for k, v in data.items() if v not in (None, "")}
    data.update(overrides)
    return LLMSettings.model_validate(data)


def _es_gemini_3(s: LLMSettings) -> bool:
    return s.provider == "gemini" and s.model.startswith("gemini-3")


def soporta_temperature(s: LLMSettings) -> bool:
    """Traducción de `temperature` por proveedor (ver README):

    - OpenAI y Gemini 2.x: se envía tal cual.
    - Gemini 3.x: Google la declaró deprecada e ignorada desde gemini-3.6-flash y
      gemini-3.5-flash-lite; se omite en toda la familia gemini-3*.
    - Anthropic (Claude Opus 4.7 en adelante, incluida la familia 5.5): la API responde 400.
    En los dos últimos casos el determinismo se busca con el prompt, no con sampling.
    """
    return s.provider == "openai" or (s.provider == "gemini" and not _es_gemini_3(s))


def reintentos_del_sdk(provider: str) -> int:
    """Valor de `max_retries` que desactiva los reintentos internos del SDK.

    Los reintentos los hace la cadena con `chain.es_reintentable`: el SDK de Gemini reintenta
    cualquier 429, incluido el de cupo diario, y lo hace sin pasar por el rate limiter.
    En langchain-google-genai `max_retries` es la cantidad total de intentos y la librería
    documenta que hay que pasar 1 (no 0) para que no reintente; en OpenAI y Anthropic cuenta
    reintentos, así que 0 = ninguno.
    """
    return 1 if provider == "gemini" else 0


@cache
def _rate_limiter(requests_per_minute: float) -> InMemoryRateLimiter:
    """Un único limitador por cupo: todos los modelos creados comparten el mismo balde de requests."""
    return InMemoryRateLimiter(requests_per_second=requests_per_minute / 60, max_bucket_size=1)


def get_chat_model(provider: str | None = None, **overrides: Any) -> BaseChatModel:
    """Devuelve ChatGoogleGenerativeAI, ChatOpenAI o ChatAnthropic según la configuración."""
    s = settings_from_env(provider, **overrides)

    # Parámetros que aceptan los tres chat models por igual. Sin reintentos del SDK: los hace la cadena.
    comunes: dict[str, Any] = {"timeout": s.timeout, "max_retries": reintentos_del_sdk(s.provider)}
    if s.requests_per_minute is not None:
        comunes["rate_limiter"] = _rate_limiter(s.requests_per_minute)

    sampling: dict[str, Any] = {}
    if s.temperature is not None:
        if soporta_temperature(s):
            sampling["temperature"] = s.temperature
        else:
            logger.warning("LLM_TEMPERATURE=%s no se envía: %s no admite temperature", s.temperature, s.model)

    # Imports diferidos: solo hace falta instalar el paquete del proveedor que se usa.
    match s.provider:
        case "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            # Gemini 3 razona antes de responder y ese razonamiento consume max_tokens.
            # Extraer entidades no requiere razonamiento profundo: "low" reduce latencia y truncamientos.
            thinking = {"thinking_level": s.thinking_level} if _es_gemini_3(s) else {}
            return ChatGoogleGenerativeAI(
                model=s.model,
                google_api_key=s.api_key,
                max_output_tokens=s.max_tokens,
                **comunes,
                **sampling,
                **thinking,
            )
        case "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=s.model, api_key=s.api_key, max_tokens=s.max_tokens, **comunes, **sampling)
        case "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=s.model, api_key=s.api_key, max_tokens=s.max_tokens, **comunes)
