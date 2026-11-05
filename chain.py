"""Cadena LCEL de extracción de entidades técnicas.

    RunnablePassthrough.assign(respuesta=
        prompt | (model.with_structured_output(RespuestaModelo, method="json_schema", include_raw=True)
                  | verificar_respuesta).with_retry(...)
    ) | anclar_al_texto

`include_raw=True` conserva el AIMessage original, necesario para leer `finish_reason`
y detectar truncamiento. Con esa opción LangChain no lanza los errores de parseo sino que
los devuelve en `parsing_error`; `verificar_respuesta` los convierte en excepciones para
que `.with_retry()` pueda reintentar. `anclar_al_texto` queda fuera del retry: descarta las
tecnologías que no aparecen en el texto y, si no queda ninguna, lanza SinTecnologiasError.
"""

import logging
import re
import time
from functools import cache
from typing import Any

from langchain_core.exceptions import ModelError, ModelRateLimitError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from llm import get_chat_model, settings_from_env
from prompt_template import extraction_prompt
from schemas import ExtraccionTecnica, RespuestaModelo

logger = logging.getLogger(__name__)

# Intentos por defecto (1 + LLM_MAX_RETRIES=2). El SDK no reintenta: todos los reintentos son
# de la cadena, así cada uno pasa por es_reintentable y por el rate limiter del modelo.
MAX_ATTEMPTS = 3
# Espera entre intentos: ~4 s y ~8 s (+ jitter de hasta 1 s). Da margen a los 503 por alta
# demanda; el cupo por minuto lo controla el rate limiter del modelo (ver llm.py).
BACKOFF = {"initial": 4, "max": 30, "exp_base": 2, "jitter": 1}
ERROR_MAX_CHARS = 200

# Valores con los que cada proveedor indica que cortó la respuesta por límite de tokens.
TRUNCATION_REASONS = {
    "length",  # OpenAI (finish_reason)
    "MAX_TOKENS",  # Gemini (finish_reason)
    "max_tokens",  # Anthropic (stop_reason)
}


class ExtractionError(Exception):
    """Base de los errores propios del pipeline."""


class TruncatedOutputError(ExtractionError):
    """El modelo cortó la respuesta por límite de tokens: el objeto quedaría incompleto."""


class SchemaValidationError(ExtractionError):
    """La respuesta no es JSON válido o no cumple el esquema Pydantic."""


class SinTecnologiasError(ExtractionError):
    """El texto no nombra ninguna tecnología concreta: no hay una extracción válida que devolver."""


def resumir_error(exc: BaseException) -> str:
    """Primera línea del error, recortada: los proveedores incluyen el JSON completo de la respuesta
    y LangChain agrega un link de troubleshooting; para el log alcanza con el comienzo."""
    linea = (str(exc).strip().splitlines() or [""])[0]
    return linea if len(linea) <= ERROR_MAX_CHARS else linea[:ERROR_MAX_CHARS] + "..."


def _finish_reason(raw: AIMessage) -> str | None:
    metadata = raw.response_metadata or {}
    return metadata.get("finish_reason") or metadata.get("stop_reason")


def verificar_respuesta(resultado: dict[str, Any]) -> RespuestaModelo:
    """Convierte la salida de include_raw en un RespuestaModelo o en una excepción tipada."""
    raw: AIMessage = resultado["raw"]
    finish_reason = _finish_reason(raw)

    # Se chequea antes que el parseo: un JSON cortado también fallaría al parsear,
    # pero la causa real es el límite de tokens y no conviene reintentarlo igual.
    if finish_reason in TRUNCATION_REASONS:
        logger.warning("Respuesta truncada por límite de tokens (finish_reason=%s)", finish_reason)
        raise TruncatedOutputError("respuesta truncada por límite de tokens: " + str(finish_reason))

    if (error := resultado.get("parsing_error")) is not None:
        logger.warning("Validación fallida: %s", resumir_error(error))
        raise SchemaValidationError(resumir_error(error)) from error

    parsed = resultado.get("parsed")
    if parsed is None:
        logger.warning("El modelo no devolvió un objeto estructurado (finish_reason=%s)", finish_reason)
        raise SchemaValidationError("el modelo no devolvió un objeto estructurado")

    logger.info("Validación OK (finish_reason=%s, tecnologias=%d)", finish_reason, len(parsed.tecnologias))
    return parsed


def _tokens(texto: str) -> set[str]:
    # Incluye . + # - para no partir nombres como "Node.js", "C++", "C#" o "scikit-learn";
    # el strip saca la puntuación de los bordes ("RDS." al final de una oración).
    return {t.strip(".-") for t in re.findall(r"[\w.+#-]+", texto.casefold())}


def aparece_en_texto(tecnologia: str, texto: str) -> bool:
    """La tecnología se menciona en el texto: completa o por alguna de sus palabras.

    Basta una palabra para aceptar variantes como "Amazon RDS" cuando el texto dice "RDS";
    lo que se busca evitar son tecnologías sin ninguna relación con el texto.
    """
    if tecnologia.casefold() in texto.casefold():
        return True
    del_texto = _tokens(texto)
    return any(len(p) >= 2 and p in del_texto for p in _tokens(tecnologia))


def anclar_al_texto(entrada: dict[str, Any]) -> ExtraccionTecnica:
    """Arma la salida final solo con tecnologías que aparecen en el texto original."""
    respuesta: RespuestaModelo = entrada["respuesta"]
    texto: str = entrada["texto"]
    ancladas = [t for t in respuesta.tecnologias if aparece_en_texto(t, texto)]
    if descartadas := [t for t in respuesta.tecnologias if t not in ancladas]:
        logger.warning("Tecnologías descartadas por no aparecer en el texto: %s", ", ".join(descartadas))
    if not ancladas:
        logger.warning("El texto no nombra ninguna tecnología concreta")
        raise SinTecnologiasError("el texto no nombra ninguna tecnología concreta")
    return ExtraccionTecnica(
        tecnologias=ancladas,
        nivel_de_criticidad=respuesta.nivel_de_criticidad,
        resumen_tecnico=respuesta.resumen_tecnico,
    )


def cuota_diaria_agotada(exc: BaseException) -> bool:
    """Un 429 por cupo diario (p. ej. Gemini free tier: 20 requests/día en los modelos Flash) no se resuelve
    esperando unos segundos, a diferencia del límite por minuto. Gemini lo indica con un quotaId
    "...PerDay..." dentro del mensaje."""
    return isinstance(exc, ModelRateLimitError) and "PerDay" in str(exc)


def es_reintentable(exc: BaseException) -> bool:
    """Política de reintento: errores de formato y fallos transitorios del proveedor.

    No se reintentan el truncamiento (con el mismo max_tokens volvería a cortar), la cuota
    diaria agotada ni los errores permanentes como autenticación o request inválido.
    """
    detalle = resumir_error(exc)
    if isinstance(exc, SchemaValidationError):
        reintentar = True
    elif cuota_diaria_agotada(exc):
        reintentar = False
        detalle = "cuota diaria del proveedor agotada; se renueva al día siguiente"
    elif isinstance(exc, ModelError):
        reintentar = exc.is_retryable
    else:
        reintentar = False
    logger.warning(
        "Intento fallido (%s: %s) -> %s",
        type(exc).__name__,
        detalle,
        "reintentable" if reintentar else "no reintentable",
    )
    return reintentar


def build_chain(
    model: BaseChatModel | None = None, intentos: int = MAX_ATTEMPTS
) -> Runnable[dict[str, str], ExtraccionTecnica]:
    """Arma la cadena LCEL. Recibe el modelo por parámetro para poder cambiarlo en pruebas."""
    model = model or get_chat_model()
    # json_schema = salida estructurada nativa del proveedor. Es el default en Gemini y OpenAI;
    # en Anthropic el default es tool calling, que con los modelos 5.5 no se puede forzar.
    extractor = model.with_structured_output(
        RespuestaModelo, method="json_schema", include_raw=True
    ) | RunnableLambda(verificar_respuesta)
    # El retry envuelve modelo + verificación: así un error de validación también se reintenta.
    # El prompt queda afuera porque formatearlo es determinístico.
    extraccion = extraction_prompt | extractor.with_retry(
        retry_if_exception_type=es_reintentable,
        stop_after_attempt=intentos,
        wait_exponential_jitter=True,
        exponential_jitter_params=BACKOFF,
    )
    # assign conserva {"texto"} junto a la respuesta para poder contrastarla. El anclaje queda
    # fuera del retry: reintentar no hace aparecer tecnologías que el texto no nombra.
    return RunnablePassthrough.assign(respuesta=extraccion) | RunnableLambda(anclar_al_texto)


@cache
def _default_chain() -> Runnable[dict[str, str], ExtraccionTecnica]:
    return build_chain(intentos=settings_from_env().max_retries + 1)


async def process_text(
    text: str, chain: Runnable[dict[str, str], ExtraccionTecnica] | None = None
) -> ExtraccionTecnica:
    """Extrae entidades técnicas de `text` de forma asíncrona.

    Lanza ValueError si el texto está vacío, TruncatedOutputError si el modelo cortó la
    respuesta, SinTecnologiasError si el texto no nombra ninguna tecnología, o la última
    excepción si se agotaron los reintentos.
    """
    if not text or not text.strip():
        raise ValueError("el texto a procesar no puede estar vacío")

    chain = chain or _default_chain()
    logger.info("Procesando texto (%d caracteres)", len(text))
    inicio = time.perf_counter()
    try:
        resultado = await chain.ainvoke({"texto": text.strip()})
    except Exception as exc:
        logger.error(
            "Extracción fallida tras %.2fs: %s: %s",
            time.perf_counter() - inicio,
            type(exc).__name__,
            resumir_error(exc),
        )
        raise
    logger.info(
        "Extracción completada en %.2fs (criticidad=%s)",
        time.perf_counter() - inicio,
        resultado.nivel_de_criticidad,
    )
    return resultado
