"""Tests offline del pipeline: no usan red ni API keys reales.

Un modelo falso (`FakeModel`) devuelve respuestas guionadas con la misma forma que
`with_structured_output(..., include_raw=True)`, para verificar la verificación de
truncamiento, la política de reintentos, el esquema y la configuración del modelo.

Uso:  python -m unittest test_offline -v
"""

import logging
import os
import unittest
from unittest.mock import patch

from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelRateLimitError,
    OutputParserException,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

import chain as chain_module
from chain import (
    SchemaValidationError,
    SinTecnologiasError,
    TruncatedOutputError,
    aparece_en_texto,
    build_chain,
    process_text,
)
from llm import get_chat_model, settings_from_env
from prompt_template import extraction_prompt
from schemas import ExtraccionTecnica, NivelCriticidad, RespuestaModelo

# Los fallos simulados generan warnings esperados; se silencian para que la salida sea legible.
logging.getLogger("chain").setLevel(logging.CRITICAL)

TEXTO = "Redis se cayó por OOM y la API de pagos devuelve 503."
CAMPOS = {"nivel_de_criticidad": "alta", "resumen_tecnico": "Caída del caché Redis por memoria agotada."}
# Lo que "devuelve el modelo" (RespuestaModelo) y lo que debe devolver el pipeline (ExtraccionTecnica).
VALIDO = RespuestaModelo(tecnologias=["Redis"], **CAMPOS)
ESPERADO = ExtraccionTecnica(tecnologias=["Redis"], **CAMPOS)


def respuesta(parsed=None, error=None, finish_reason="STOP") -> dict:
    raw = AIMessage(content="{}", response_metadata={"finish_reason": finish_reason})
    return {"raw": raw, "parsed": parsed, "parsing_error": error}


class FakeModel:
    """Imita `model.with_structured_output(..., include_raw=True)` con respuestas guionadas."""

    def __init__(self, *guion):
        self.guion = list(guion)
        self.llamadas = 0
        self.prompts = []

    def with_structured_output(self, schema, method=None, include_raw=False):
        assert schema is RespuestaModelo and method == "json_schema" and include_raw

        def paso(prompt_value):
            self.llamadas += 1
            self.prompts.append(prompt_value)
            item = self.guion.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        return RunnableLambda(paso)


class TestCadena(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Sin esperas entre reintentos: acá solo importa cuántas veces se llama al modelo.
        patcher = patch.dict(chain_module.BACKOFF, initial=0, max=0, jitter=0)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def ejecutar(self, *guion):
        modelo = FakeModel(*guion)
        return modelo, build_chain(modelo)

    async def test_respuesta_valida(self):
        modelo, cadena = await self.ejecutar(respuesta(VALIDO))
        resultado = await process_text(TEXTO, chain=cadena)
        self.assertEqual(resultado, ESPERADO)
        self.assertEqual(modelo.llamadas, 1)

    async def test_prompt_delimita_el_texto(self):
        modelo, cadena = await self.ejecutar(respuesta(VALIDO))
        await process_text(TEXTO, chain=cadena)
        system, human = modelo.prompts[0].to_messages()
        self.assertEqual(human.content, "<texto>\n" + TEXTO + "\n</texto>")
        self.assertIn("Formato de salida", system.content)

    async def test_error_de_parseo_se_reintenta(self):
        modelo, cadena = await self.ejecutar(
            respuesta(error=OutputParserException("json inválido")), respuesta(VALIDO)
        )
        self.assertEqual(await process_text(TEXTO, chain=cadena), ESPERADO)
        self.assertEqual(modelo.llamadas, 2)

    async def test_esquema_siempre_invalido_agota_reintentos(self):
        modelo, cadena = await self.ejecutar(
            *[respuesta(error=OutputParserException("x")) for _ in range(3)]
        )
        with self.assertRaises(SchemaValidationError):
            await process_text(TEXTO, chain=cadena)
        self.assertEqual(modelo.llamadas, 3)

    async def test_cantidad_de_intentos_configurable(self):
        # LLM_MAX_RETRIES=1 -> 2 intentos en total (process_text usa 1 + LLM_MAX_RETRIES).
        modelo = FakeModel(*[respuesta(error=OutputParserException("x")) for _ in range(2)])
        with self.assertRaises(SchemaValidationError):
            await process_text(TEXTO, chain=build_chain(modelo, intentos=2))
        self.assertEqual(modelo.llamadas, 2)

    async def test_parsed_vacio_es_error_de_esquema(self):
        modelo, cadena = await self.ejecutar(*[respuesta() for _ in range(3)])
        with self.assertRaises(SchemaValidationError):
            await process_text(TEXTO, chain=cadena)

    async def test_truncamiento_no_se_reintenta(self):
        for finish_reason in ("MAX_TOKENS", "length", "max_tokens"):
            with self.subTest(finish_reason=finish_reason):
                modelo, cadena = await self.ejecutar(
                    respuesta(error=OutputParserException("cortado"), finish_reason=finish_reason)
                )
                with self.assertRaises(TruncatedOutputError):
                    await process_text(TEXTO, chain=cadena)
                self.assertEqual(modelo.llamadas, 1)

    async def test_rate_limit_se_reintenta(self):
        modelo, cadena = await self.ejecutar(ModelRateLimitError("429"), respuesta(VALIDO))
        self.assertEqual(await process_text(TEXTO, chain=cadena), ESPERADO)
        self.assertEqual(modelo.llamadas, 2)

    async def test_cuota_diaria_no_se_reintenta(self):
        error = ModelRateLimitError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        modelo, cadena = await self.ejecutar(error)
        with self.assertRaises(ModelRateLimitError):
            await process_text(TEXTO, chain=cadena)
        self.assertEqual(modelo.llamadas, 1)

    async def test_autenticacion_no_se_reintenta(self):
        modelo, cadena = await self.ejecutar(ModelAuthenticationError("401"))
        with self.assertRaises(ModelAuthenticationError):
            await process_text(TEXTO, chain=cadena)
        self.assertEqual(modelo.llamadas, 1)

    async def test_descarta_tecnologias_que_no_estan_en_el_texto(self):
        inventadas = RespuestaModelo(tecnologias=["Redis", "AWS", "Kafka"], **CAMPOS)
        modelo, cadena = await self.ejecutar(respuesta(inventadas))
        self.assertEqual(await process_text(TEXTO, chain=cadena), ESPERADO)

    async def test_sin_tecnologias_lanza_error_sin_reintentar(self):
        vacia = RespuestaModelo(tecnologias=["app"], nivel_de_criticidad="baja", resumen_tecnico="x" * 30)
        modelo, cadena = await self.ejecutar(respuesta(vacia))
        with self.assertRaises(SinTecnologiasError):
            await process_text("La app anda lenta a la tarde.", chain=cadena)
        self.assertEqual(modelo.llamadas, 1)

    async def test_texto_vacio(self):
        for texto in ("", "   \n"):
            with self.subTest(texto=texto), self.assertRaises(ValueError):
                await process_text(texto)


class TestEsquema(unittest.TestCase):
    def test_normaliza_tecnologias(self):
        e = ExtraccionTecnica(
            tecnologias=[" Redis", "redis", "", "PostgreSQL "],
            nivel_de_criticidad="media",
            resumen_tecnico="  Latencia   alta en la API por conexiones agotadas.  ",
        )
        self.assertEqual(e.tecnologias, ["Redis", "PostgreSQL"])
        self.assertEqual(e.resumen_tecnico, "Latencia alta en la API por conexiones agotadas.")
        self.assertIs(e.nivel_de_criticidad, NivelCriticidad.MEDIA)

    def test_rellenos_y_genericos(self):
        # El modelo puede responder [] (RespuestaModelo), pero la salida final exige una tecnología.
        for valores in ([], ["  "], ["No especificada"], ["N/A", " ninguna "], ["app", "Servidor"]):
            with self.subTest(valores=valores):
                r = RespuestaModelo(tecnologias=valores, nivel_de_criticidad="baja", resumen_tecnico="x" * 30)
                self.assertEqual(r.tecnologias, [])
                with self.assertRaises(ValidationError):
                    ExtraccionTecnica(tecnologias=valores, nivel_de_criticidad="baja", resumen_tecnico="x" * 30)

    def test_aparece_en_texto(self):
        texto = "El API gateway en FastAPI usa PostgreSQL en RDS. El front es Node.js y hay un job en C++."
        for tecnologia in ("FastAPI", "postgresql", "Amazon RDS", "Node.js", "C++"):
            with self.subTest(tecnologia):
                self.assertTrue(aparece_en_texto(tecnologia, texto))
        for tecnologia in ("Redis", "Kubernetes", "Go"):
            with self.subTest(tecnologia):
                self.assertFalse(aparece_en_texto(tecnologia, texto))

    def test_rechaza_datos_invalidos(self):
        base = {"tecnologias": ["Redis"], "nivel_de_criticidad": "alta", "resumen_tecnico": "x" * 30}
        casos = {
            "tecnologías no es lista": {"tecnologias": "Redis"},
            "sin tecnologías": {"tecnologias": []},
            "criticidad fuera del enum": {"nivel_de_criticidad": "critica"},
            "resumen corto": {"resumen_tecnico": "corto"},
            "resumen largo": {"resumen_tecnico": "x" * 501},
        }
        for nombre, cambio in casos.items():
            with self.subTest(nombre), self.assertRaises(ValidationError):
                ExtraccionTecnica(**(base | cambio))

    def test_resumir_error_recorta(self):
        error = RuntimeError("429 RESOURCE_EXHAUSTED " + "x" * 1000 + "\nFor troubleshooting...")
        resumen = chain_module.resumir_error(error)
        self.assertTrue(resumen.startswith("429 RESOURCE_EXHAUSTED"))
        self.assertLessEqual(len(resumen), chain_module.ERROR_MAX_CHARS + 3)

    def test_prompt_solo_espera_texto(self):
        self.assertEqual(extraction_prompt.input_variables, ["texto"])


@patch.dict(os.environ, {}, clear=True)
class TestConfiguracion(unittest.TestCase):
    KEY = "clave-de-prueba"

    def test_falta_api_key(self):
        with self.assertRaisesRegex(ValidationError, "falta la API key de gemini"):
            settings_from_env()

    def test_modelos_por_defecto(self):
        esperados = {"gemini": "gemini-3.5-flash-lite", "openai": "gpt-4o-mini", "anthropic": "claude-sonnet-5-5"}
        for proveedor, modelo in esperados.items():
            with self.subTest(proveedor):
                self.assertEqual(settings_from_env(proveedor, api_key=self.KEY).model, modelo)

    def test_gemini_3_sin_temperature_y_con_thinking_low(self):
        with self.assertLogs("llm", level="WARNING"):
            modelo = get_chat_model("gemini", api_key=self.KEY, temperature=0.3)
        self.assertIsNone(modelo.temperature)
        self.assertEqual(modelo.thinking_level, "low")

    def test_gemini_2_recibe_temperature_sin_thinking(self):
        modelo = get_chat_model("gemini", api_key=self.KEY, model="gemini-2.5-flash", temperature=0.3)
        self.assertEqual(modelo.temperature, 0.3)
        self.assertIsNone(modelo.thinking_level)

    def test_openai_recibe_temperature(self):
        modelo = get_chat_model("openai", api_key=self.KEY, temperature=0.3)
        self.assertEqual(modelo.temperature, 0.3)

    def test_anthropic_nunca_recibe_temperature(self):
        with self.assertLogs("llm", level="WARNING"):
            modelo = get_chat_model("anthropic", api_key=self.KEY, temperature=0.3)
        self.assertIsNone(modelo.temperature)

    def test_sdk_sin_reintentos_internos(self):
        # Los reintentos son de la cadena. En langchain-google-genai max_retries cuenta intentos
        # totales (1 = sin reintentos); en OpenAI y Anthropic cuenta reintentos (0 = ninguno).
        self.assertEqual(get_chat_model("gemini", api_key=self.KEY).max_retries, 1)
        for proveedor in ("openai", "anthropic"):
            with self.subTest(proveedor):
                self.assertEqual(get_chat_model(proveedor, api_key=self.KEY).max_retries, 0)

    def test_rate_limiter_compartido(self):
        sin_limite = get_chat_model("gemini", api_key=self.KEY)
        a = get_chat_model("gemini", api_key=self.KEY, requests_per_minute=5)
        b = get_chat_model("gemini", api_key=self.KEY, requests_per_minute=5, max_tokens=16)
        self.assertIsNone(sin_limite.rate_limiter)
        self.assertIs(a.rate_limiter, b.rate_limiter)  # el modelo del truncamiento comparte el cupo
        self.assertAlmostEqual(a.rate_limiter.requests_per_second, 5 / 60)

    def test_valida_rangos(self):
        for invalido in ({"temperature": 3}, {"max_tokens": 0}, {"thinking_level": "minimal"}):
            with self.subTest(invalido), self.assertRaises(ValidationError):
                settings_from_env("gemini", api_key=self.KEY, **invalido)


if __name__ == "__main__":
    unittest.main()
