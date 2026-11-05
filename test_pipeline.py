"""Mini script de prueba asíncrono del pipeline.

1. Procesa 4 textos en paralelo con asyncio.gather (criticidad alta, baja, media y un caso ambiguo).
2. Fuerza un límite de tokens muy bajo para demostrar la detección de truncamiento.

Uso:  python test_pipeline.py   (requiere un .env configurado, ver .env.example)
"""

import asyncio
import logging
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from chain import SinTecnologiasError, TruncatedOutputError, build_chain, process_text, resumir_error
from llm import get_chat_model, settings_from_env

TEXTOS: dict[str, str] = {
    "Log crítico": (
        "2026-11-03 14:02:11 ERROR [api-gateway] FastAPI worker 3: sqlalchemy.exc.TimeoutError: "
        "QueuePool limit of size 20 overflow 10 reached, connection timed out, timeout 30.00. "
        "PostgreSQL 16 en RDS no acepta nuevas conexiones. El clúster de Redis responde OK, pero el 78% "
        "de los requests a /checkout devuelven 503 desde hace 12 minutos. Kubernetes reinició 4 pods "
        "por liveness probe fallida."
    ),
    "Advertencia menor": (
        "DeprecationWarning en el job nocturno de reportes: pandas 2.2 avisa que DataFrame.applymap está "
        "deprecado y será reemplazado por DataFrame.map. El script de Python 3.12 terminó correctamente "
        "y generó todos los reportes a horario."
    ),
    "Arquitectura con degradación": (
        "Plataforma de e-commerce con microservicios en Go y Node.js desplegados en Kubernetes (EKS). "
        "La comunicación asíncrona usa Apache Kafka; los pedidos se guardan en PostgreSQL y el catálogo "
        "en Elasticsearch. Un frontend en Next.js consume una API GraphQL. Desde el último deploy, las "
        "búsquedas en Elasticsearch tardan 4 segundos en promedio (antes 300 ms), aunque no hay errores."
    ),
    "Ambiguo (prueba de estrés)": (
        "Desde la actualización del viernes algunos usuarios dicen que la app 'anda lenta' a veces, "
        "sobre todo a la tarde. No hay errores en los logs y nadie sabe bien qué componente cambió."
    ),
}


def configurar_logging() -> None:
    # La consola de Windows no usa UTF-8 por defecto y rompería las tildes.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Los clientes HTTP de los SDK loguean cada request; se silencian para leer solo el pipeline.
    for ruidoso in ("httpx", "httpcore", "google_genai", "openai", "anthropic"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


# No menciona tecnologías: lo correcto es que el pipeline lance SinTecnologiasError
# en vez de devolver una lista vacía o una tecnología inventada.
SIN_TECNOLOGIAS = {"Ambiguo (prueba de estrés)"}


async def probar_textos() -> int:
    """Devuelve la cantidad de textos cuyo resultado no fue el esperado."""
    resultados = await asyncio.gather(
        *(process_text(texto) for texto in TEXTOS.values()), return_exceptions=True
    )
    inesperados = 0
    for nombre, resultado in zip(TEXTOS, resultados):
        print("\n=== " + nombre + " ===")
        if nombre in SIN_TECNOLOGIAS:
            if isinstance(resultado, SinTecnologiasError):
                print("OK -> rechazado como se esperaba: " + resumir_error(resultado))
                continue
            inesperados += 1
            print("INESPERADO -> se esperaba SinTecnologiasError y se obtuvo:")
        elif isinstance(resultado, Exception):
            inesperados += 1
        if isinstance(resultado, Exception):
            print("ERROR -> " + type(resultado).__name__ + ": " + resumir_error(resultado))
        else:
            print(resultado.model_dump_json(indent=2))
    return inesperados


async def probar_truncamiento() -> None:
    print("\n=== Truncamiento forzado (max_tokens=16) ===")
    chain = build_chain(get_chat_model(max_tokens=16))
    try:
        await process_text(TEXTOS["Log crítico"], chain=chain)
        print("El modelo respondió completo dentro del límite: no hubo truncamiento.")
    except TruncatedOutputError as exc:
        print("OK -> truncamiento detectado y no reintentado: " + str(exc))
    except Exception as exc:  # p. ej. cuota agotada: se informa sin cortar el script
        print("No se pudo probar el truncamiento -> " + type(exc).__name__ + ": " + resumir_error(exc))


async def main() -> None:
    inesperados = await probar_textos()
    await probar_truncamiento()
    correctos = len(TEXTOS) - inesperados
    print("\nResumen: " + str(correctos) + "/" + str(len(TEXTOS)) + " textos con el resultado esperado.")


if __name__ == "__main__":
    load_dotenv()
    configurar_logging()
    try:
        settings = settings_from_env()
    except ValidationError as exc:
        sys.exit("Configuración inválida: " + exc.errors()[0]["msg"])
    if settings.requests_per_minute:
        logging.getLogger(__name__).info(
            "Rate limit activo: %s requests/min. Los textos se lanzan en paralelo, pero las llamadas "
            "al modelo se espacian; la corrida completa puede tardar uno o dos minutos.",
            settings.requests_per_minute,
        )
    asyncio.run(main())
