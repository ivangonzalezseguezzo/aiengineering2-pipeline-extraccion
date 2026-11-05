"""Contratos de datos del pipeline.

- `RespuestaModelo`: lo que se le pide al LLM. `tecnologias` puede venir vacía a propósito:
  exigirle al modelo al menos un elemento lo empujaba a inventar uno ("No especificada", "app").
- `ExtraccionTecnica`: el contrato final que devuelve el pipeline. Exige al menos una tecnología;
  `chain.anclar_al_texto` construye este objeto solo con tecnologías que aparecen en el texto.

Las `description` de cada campo se envían al modelo como parte del JSON Schema,
así que están redactadas como instrucciones para el LLM.
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

RESUMEN_MIN_CHARS = 20
RESUMEN_MAX_CHARS = 500

# Valores de relleno que un LLM puede devolver cuando el texto no menciona tecnologías
# (observado con Gemini). No son tecnologías: se descartan.
PLACEHOLDERS = {
    "no especificada", "no especificado", "no especificadas", "ninguna", "ninguno",
    "desconocida", "desconocido", "no aplica", "n/a", "na", "none", "unknown", "-",
}

# Palabras que aparecen en textos técnicos pero no nombran una tecnología concreta
# (observado: el modelo devolvió "app" para un texto sin tecnologías).
TERMINOS_GENERICOS = {
    "app", "apps", "aplicación", "aplicacion", "aplicaciones", "sistema", "sistemas",
    "servidor", "servidores", "servicio", "servicios", "base de datos", "bases de datos",
    "backend", "frontend", "componente", "componentes", "logs", "log", "web", "nube", "cloud",
}


class NivelCriticidad(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class RespuestaModelo(BaseModel):
    """Entidades técnicas extraídas de un log de error o una descripción de arquitectura."""

    tecnologias: list[str] = Field(
        description=(
            "Tecnologías, lenguajes, frameworks, bases de datos, servicios cloud o herramientas "
            "mencionados explícitamente en el texto, con su nombre propio (ej. 'PostgreSQL', 'FastAPI'). "
            "Sin duplicados. Si el texto no nombra ninguna tecnología concreta, devolvé una lista vacía: "
            "no inventes ni uses términos genéricos ('app', 'servidor', 'base de datos') o de relleno ('N/A')."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description=(
            "Impacto técnico de la situación descrita: 'alta' si hay caída del servicio, pérdida de datos "
            "o riesgo de seguridad; 'media' si hay degradación de rendimiento o errores parciales; "
            "'baja' si es informativo, una advertencia menor o una descripción sin problemas."
        ),
    )
    resumen_tecnico: str = Field(
        description=(
            "Resumen técnico en español de una o dos oraciones (20 a 500 caracteres): "
            "qué sistema es, qué ocurre y, si se deduce, la causa probable."
        ),
    )

    @field_validator("tecnologias")
    @classmethod
    def normalizar_tecnologias(cls, valores: list[str]) -> list[str]:
        """Quita espacios, vacíos, valores de relleno, términos genéricos y duplicados
        (sin distinguir mayúsculas), conservando el orden."""
        vistas: set[str] = PLACEHOLDERS | TERMINOS_GENERICOS
        normalizadas: list[str] = []
        for valor in valores:
            limpio = valor.strip()
            if limpio and limpio.casefold() not in vistas:
                vistas.add(limpio.casefold())
                normalizadas.append(limpio)
        return normalizadas

    # El largo se valida acá y no con Field(min_length/max_length) porque no todos los
    # proveedores aceptan esas restricciones de string en su JSON Schema.
    @field_validator("resumen_tecnico")
    @classmethod
    def validar_resumen(cls, valor: str) -> str:
        limpio = " ".join(valor.split())
        if not RESUMEN_MIN_CHARS <= len(limpio) <= RESUMEN_MAX_CHARS:
            raise ValueError(
                "el resumen debe tener entre %d y %d caracteres (tiene %d)"
                % (RESUMEN_MIN_CHARS, RESUMEN_MAX_CHARS, len(limpio))
            )
        return limpio


class ExtraccionTecnica(RespuestaModelo):
    """Salida validada del pipeline: siempre tiene al menos una tecnología."""

    # Se chequea después de normalizar (un validador "after"): ["  "] o ["N/A"] también quedan vacías.
    @model_validator(mode="after")
    def exigir_tecnologias(self) -> Self:
        if not self.tecnologias:
            raise ValueError("la extracción debe tener al menos una tecnología")
        return self
