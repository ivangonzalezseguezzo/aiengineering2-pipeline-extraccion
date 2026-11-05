# Pipeline de Extracción de Entidades Técnicas

Pre-entrega 2 de **AI Engineering (Coderhouse)**: un pipeline de procesamiento validado con **LangChain (LCEL)** y **Pydantic**.

El pipeline recibe un texto técnico sin procesar, como un log de error o una descripción de arquitectura. Devuelve un objeto validado con las tecnologías mencionadas, el nivel de criticidad y un resumen técnico.

## Ejemplo de salida

Entrada (log de error):

```text
2026-11-03 14:02:11 ERROR [api-gateway] FastAPI worker 3: sqlalchemy.exc.TimeoutError: QueuePool limit
of size 20 overflow 10 reached... PostgreSQL 16 en RDS no acepta nuevas conexiones. El clúster de Redis
responde OK, pero el 78% de los requests a /checkout devuelven 503... Kubernetes reinició 4 pods...
```

Salida real de `gemini-3.5-flash-lite` (`ExtraccionTecnica.model_dump_json(indent=2)`):

```json
{
  "tecnologias": [
    "FastAPI",
    "SQLAlchemy",
    "PostgreSQL",
    "RDS",
    "Redis",
    "Kubernetes"
  ],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "Se registra una saturación en el pool de conexiones de SQLAlchemy hacia PostgreSQL en RDS, provocando un error de timeout y fallas masivas en el endpoint de checkout. Como consecuencia, Kubernetes ha reiniciado múltiples pods debido al fallo en las sondas de vivencia."
}
```

El log completo de esa corrida (2026-11-05, con el código final: 5 requests al modelo) está en [`evidencias/corrida_real_gemini-3.5-flash-lite.log`](evidencias/corrida_real_gemini-3.5-flash-lite.log). Incluye los 4 textos (el ambiguo termina en `SinTecnologiasError`, como se espera) y el truncamiento detectado sin reintento. Los 4 textos se validaron al primer intento: en esa corrida no hizo falta ningún reintento de la cadena.

## Estructura

```
├── schemas.py          # Contratos (Pydantic): RespuestaModelo, ExtraccionTecnica y NivelCriticidad
├── prompt_template.py  # ChatPromptTemplate (system + human) con variables, sin f-strings
├── llm.py              # Fábrica del chat model según LLM_PROVIDER (Gemini / OpenAI / Anthropic)
├── chain.py            # Cadena LCEL, verificación de truncamiento, retry y process_text()
├── test_pipeline.py    # Mini script de prueba asíncrono (contra la API real)
├── test_offline.py     # 28 tests sin red ni API keys (unittest)
├── evidencias/         # Log de una corrida real contra Gemini
├── requirements.txt
└── .env.example
```

Cada módulo tiene una sola responsabilidad. `chain.py` es el único que conoce a los demás, así que cambiar de proveedor solo toca `llm.py` y ajustar el prompt solo toca `prompt_template.py`.

## Flujo de la cadena

```
process_text(texto)
   │  valida que no esté vacío, loguea inicio y mide latencia
   ▼
{"texto": ...} ──► RunnablePassthrough.assign(respuesta=...)  (conserva el texto)
                     │
                     ▼
                   extraction_prompt ──► ChatPromptValue (system + human)
                     │
   ┌──────────── .with_retry(stop_after_attempt=3) ────────────┐
   │ model.with_structured_output(RespuestaModelo,             │
   │                              method="json_schema",        │
   │                              include_raw=True)            │
   │   → {"raw": AIMessage, "parsed": ..., "parsing_error"}    │
   │                     ▼                                     │
   │ verificar_respuesta                                       │
   │   ├─ finish_reason de truncamiento → TruncatedOutputError │
   │   ├─ parsing_error / parsed vacío  → SchemaValidationError│
   │   └─ OK → RespuestaModelo                                 │
   └───────────────────────────────────────────────────────────┘
                     ▼
   anclar_al_texto({"texto", "respuesta"})   (fuera del retry)
     ├─ descarta tecnologías que no aparecen en el texto
     ├─ ninguna tecnología → SinTecnologiasError
     └─ OK → ExtraccionTecnica (validado, al menos una tecnología)
```

```python
extraccion = extraction_prompt | (
    model.with_structured_output(RespuestaModelo, method="json_schema", include_raw=True)
    | RunnableLambda(verificar_respuesta)
).with_retry(retry_if_exception_type=es_reintentable, stop_after_attempt=3)

chain = RunnablePassthrough.assign(respuesta=extraccion) | RunnableLambda(anclar_al_texto)
```

## Decisiones de diseño

### Contrato de datos (`schemas.py`)
- **`nivel_de_criticidad`** es un `StrEnum` (`baja`, `media`, `alta`). El JSON Schema se lo informa al modelo como `enum`, y cualquier otro valor se rechaza.
- **Dos esquemas para `tecnologias`**: la salida del pipeline (`ExtraccionTecnica`) **siempre tiene al menos una tecnología y todas aparecen en el texto**. Para lograrlo sin que el modelo invente, se separan dos contratos:
  - `RespuestaModelo` es lo que se le pide al LLM, y ahí la lista **puede venir vacía**. En las primeras corridas reales se le exigía al menos un elemento, y el modelo inventaba uno para cumplir: `"No especificada"` en una corrida y `"app"` en otra.
  - `anclar_al_texto` (en `chain.py`) contrasta cada tecnología con el texto original y descarta las que no aparecen. Se acepta la mención completa o alguna de sus palabras, para admitir variantes como `"Amazon RDS"` cuando el texto dice `"RDS"`. Si no queda ninguna, lanza `SinTecnologiasError`.
  - `ExtraccionTecnica` hereda de `RespuestaModelo` y exige al menos una tecnología con un `model_validator`.

  Un `field_validator` común a los dos esquemas saca espacios y elimina duplicados sin distinguir mayúsculas (`"Redis"` y `"redis "` quedan una sola vez). También descarta valores de relleno (`"N/A"`) y términos genéricos que no son tecnologías (`"app"`, `"servidor"`, `"base de datos"`).
- **`resumen_tecnico`** debe tener entre 20 y 500 caracteres. Se valida con un `field_validator` y no con `Field(max_length=...)` porque no todos los proveedores aceptan esas restricciones en su JSON Schema.
- **Las `description`** de los campos están escritas como instrucciones para el modelo, porque viajan en el JSON Schema.

### `with_structured_output` en lugar de `PydanticOutputParser`
Con `method="json_schema"` se usa la salida estructurada nativa de cada proveedor: `response_json_schema` en Gemini, `response_format` en OpenAI y `output_config.format` en Anthropic. Así el modelo queda forzado a respetar el esquema desde la generación, no solo al parsear después. El método se pasa explícito porque en Anthropic el default es tool calling, y con los modelos 5.5 LangChain no puede forzar esa llamada.

### Detección de truncamiento (`finish_reason`)
Con `include_raw=True` se conserva el `AIMessage` original. `verificar_respuesta` lee `response_metadata` y compara contra los valores de corte de cada proveedor:

| Proveedor | Campo | Valor de truncamiento |
|---|---|---|
| OpenAI | `finish_reason` | `length` |
| Gemini | `finish_reason` | `MAX_TOKENS` |
| Anthropic | `stop_reason` | `max_tokens` |

El truncamiento se chequea **antes** que el parseo. Un JSON cortado también fallaría al parsear, pero la causa real es el límite de tokens.

### Política de reintentos
`.with_retry()` envuelve **modelo + verificación**, así un error de validación también dispara un reintento. El prompt queda afuera porque formatearlo es determinístico. La función `es_reintentable` decide qué reintentar:

| Error | ¿Se reintenta? | Motivo |
|---|---|---|
| `SchemaValidationError` (JSON inválido o fuera de esquema) | Sí | El modelo puede responder bien en el siguiente intento |
| Rate limit, timeout, conexión, error 5xx | Sí | Fallos transitorios (`ModelError.is_retryable`) |
| `TruncatedOutputError` | No | Con el mismo `max_tokens` volvería a cortar |
| Rate limit por cuota **diaria** agotada | No | Se renueva al día siguiente: esperar segundos no sirve |
| Autenticación, permisos, request inválido | No | Errores permanentes: reintentar solo gasta tiempo y dinero |

La espera entre intentos usa backoff exponencial con jitter: unos 4 s y luego unos 8 s, para dar margen a los 503 por alta demanda. Son 1 intento + `LLM_MAX_RETRIES` (2 por defecto, o sea 3 intentos).

**Los reintentos son solo de la cadena, no del SDK.** El SDK de Gemini reintenta cualquier 429, incluido el de cupo diario, sin distinguirlos y sin pasar por el rate limiter. Por eso `llm.py` crea los chat models sin reintentos internos (`reintentos_del_sdk`: `max_retries=1` en Gemini, que en esa librería significa un único intento, y `0` en OpenAI y Anthropic). Así cada reintento pasa por `es_reintentable` y descuenta del cupo por minuto. Es la misma política que las pre-entregas 3, 4 y 5.

### Cupo de requests (`LLM_REQUESTS_PER_MINUTE`)
El plan gratuito de Gemini limita las requests por minuto y por día, y cada modelo tiene su propio cupo:

| Modelos | Requests/min | Tokens/min | Requests/día |
|---|---|---|---|
| Flash (`gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3-flash`, `gemini-2.5-flash`) | 5 | 250K | 20 |
| Flash Lite (`gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`) | 15 | 250K | 500 |

*Valores del panel de AI Studio en noviembre de 2026. Google aclara que pueden cambiar; los de cada proyecto se ven en [aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit).*

Por eso el modelo por defecto es **`gemini-3.5-flash-lite`**. Las primeras corridas se hicieron con `gemini-3.8-flash`, y su cupo de 20 por día se agotó en una tarde de pruebas.

Sin control, lanzar 4 textos en paralelo con 3 intentos cada uno supera el límite por minuto en segundos. Además, reintentar a los pocos segundos un 429 no sirve, porque Google pide esperar casi un minuto.

Por eso `llm.py` usa el `InMemoryRateLimiter` de LangChain:
- Es un único limitador por cupo, compartido por todos los modelos que se crean. El modelo de la prueba de truncamiento también descuenta del mismo cupo.
- El paralelismo de `asyncio.gather` se mantiene: las corrutinas corren a la vez, pero las llamadas al modelo se espacian. Con `LLM_REQUESTS_PER_MINUTE=12` sale una cada 5 segundos. Para un modelo Flash conviene bajarlo a 4.
- Si el plan es pago, se borra la variable y no hay límite.

Limitación: el limitador es por proceso. Dos ejecuciones simultáneas del script comparten el mismo cupo de la API key y se pisan.

El **cupo diario** se reinicia a la medianoche del Pacífico. Una corrida completa de `test_pipeline.py` usa unos 5 requests si no hay reintentos. Un 429 por cupo diario no se reintenta y genera una sola request: `cuota_diaria_agotada` en `chain.py` lo detecta por el `quotaId` `...PerDay...` que envía Gemini.

### Sin f-strings en el prompt
El texto del usuario entra como variable de template (`{texto}`) y delimitado entre etiquetas `<texto>`. Las instrucciones de formato se inyectan con `.partial()`. Los logs usan formato `%s` del módulo `logging`, que solo arma el mensaje si el nivel está activo.

### Proveedor
`llm.py` retoma el patrón de configuración de la pre-entrega 1: variables de entorno validadas con un modelo Pydantic (`LLMSettings`) y un proveedor elegible por `LLM_PROVIDER`. Soporta `ChatGoogleGenerativeAI` (por defecto), `ChatOpenAI` y `ChatAnthropic`. La consigna menciona ChatOpenAI o ChatAnthropic: los dos están implementados y se eligen con `LLM_PROVIDER=openai` o `LLM_PROVIDER=anthropic`. El valor por defecto es Gemini porque es el único de los tres con un plan gratuito, que es con el que se hicieron las pruebas reales. Se usan los chat models de LangChain y no el cliente propio del Módulo 1 porque `with_structured_output` solo existe en ellos.

#### Modelos por defecto
Son identificadores reales de cada API (verificados en noviembre de 2026) y se pueden cambiar en el `.env`:

| Proveedor | Modelo por defecto | Estado | Fuente |
|---|---|---|---|
| Gemini | `gemini-3.5-flash-lite` | Estable. Es el de mayor cupo gratuito; `gemini-3.8-flash` también funciona | [Modelos de Gemini](https://aistudio.google.com/docs/models) |
| OpenAI | `gpt-4o-mini` | Deprecado, se da de baja el 14/01/2027 (reemplazo: `gpt-6-luna`) | [Deprecaciones de OpenAI](https://developers.openai.com/api/docs/deprecations) |
| Anthropic | `claude-sonnet-5-5` | Vigente | [Modelos de Claude](https://platform.claude.com/docs/en/about-claude/models/overview) |

Solo se probó contra la API real el proveedor por defecto (Gemini).

#### `temperature` entre proveedores
No todos los proveedores aceptan este parámetro. Para que la configuración no tenga efectos distintos sin aviso, `llm.py` aplica esta regla (`soporta_temperature`):

| Proveedor / modelo | Qué pasa con `LLM_TEMPERATURE` |
|---|---|
| OpenAI, Gemini 2.x | Se envía tal cual |
| Gemini 3.x | No se envía: Google lo declaró deprecado e ignorado desde `gemini-3.6-flash` y `gemini-3.5-flash-lite`, y se omite en toda la familia `gemini-3*` |
| Anthropic (Claude Opus 4.7 en adelante, incluida la familia 5.5) | No se envía: la API responde 400 |

En los dos últimos casos se registra un `WARNING`. El determinismo se busca con el prompt (reglas explícitas y esquema), que es lo que recomiendan Google y Anthropic.

#### Razonamiento de Gemini 3 (`LLM_THINKING_LEVEL`)
El razonamiento interno consume del mismo `max_tokens`, y extraer entidades no requiere pensar a fondo. Por eso el valor por defecto es `low`: baja la latencia y el riesgo de truncamiento. Se probó con `gemini-3.5-flash-lite` y `gemini-3.8-flash`. Si otro modelo lo rechaza, se cambia `LLM_THINKING_LEVEL`: por ejemplo, según la documentación de Google, `gemini-3.6-flash` acepta solo `medium` o `high`.

### Errores que puede lanzar `process_text`
Todas las excepciones propias heredan de `ExtractionError` (en `chain.py`). Las del proveedor son subclases de `langchain_core.exceptions.ModelError`, que es igual en los tres proveedores.

| Excepción | Cuándo ocurre | ¿Se reintentó? | Cómo reaccionar |
|---|---|---|---|
| `ValueError` | El texto está vacío | No llega al modelo | Validar la entrada |
| `TruncatedOutputError` | `finish_reason` indica límite de tokens | No | Subir `LLM_MAX_TOKENS` o acortar el texto |
| `SchemaValidationError` | La respuesta no cumplió el esquema en ninguno de los intentos | Sí (3 intentos por defecto) | Revisar el texto o el prompt |
| `SinTecnologiasError` | El texto no nombra ninguna tecnología concreta | No: otro intento no las haría aparecer | Pedir un texto más específico |
| `ModelRateLimitError` con `...PerDay...` en el mensaje | Se agotó el cupo diario del modelo | No (una sola request) | Esperar al reinicio diario o cambiar `GEMINI_MODEL` |
| `ModelError` con `is_retryable=True` (rate limit por minuto, timeout, 5xx) | Falla transitoria del proveedor | Sí (3 intentos por defecto) | Reintentar más tarde |
| `ModelError` con `is_retryable=False` (API key, permisos, request inválido, modelo inexistente) | Error de configuración | No | Corregir el `.env` |

```python
from chain import ExtractionError, TruncatedOutputError, process_text
from langchain_core.exceptions import ModelError

async def extraer(texto: str):
    try:
        return await process_text(texto)
    except TruncatedOutputError:
        ...  # subir LLM_MAX_TOKENS
    except ExtractionError:
        ...  # esquema inválido tras 3 intentos (SchemaValidationError) o texto sin tecnologías (SinTecnologiasError)
    except ModelError as exc:
        ...  # exc.is_retryable indica si conviene reintentar más tarde
```

### Logs
El pipeline registra estos eventos:
- inicio de cada extracción
- resultado de cada validación (OK / fallida, con el motivo)
- cada intento fallido y si es reintentable
- truncamiento detectado
- tecnologías descartadas por no aparecer en el texto, y textos sin ninguna tecnología
- latencia total y criticidad resultante

## Instalación

Requiere Python 3.12 o superior.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # y completá la API key del proveedor elegido
```

## Tests offline

```bash
python -m unittest test_offline -v
```

Son 28 tests que no usan red ni API keys: un modelo falso devuelve respuestas guionadas. Cubren:
- **la cadena**: respuesta válida, descarte de tecnologías que no están en el texto, `SinTecnologiasError` sin reintento, delimitación del texto en el prompt, reintento ante error de parseo y ante rate limit, reintentos agotados, cantidad de intentos configurable, truncamiento sin reintento en los tres proveedores, cuota diaria agotada y error de autenticación (ambos sin reintento), y texto vacío
- **el esquema**: normalización de tecnologías, rellenos y términos genéricos, búsqueda de tecnologías en el texto (`aparece_en_texto`), rechazo de datos inválidos y recorte de mensajes de error para el log
- **la configuración**: modelos por defecto, regla de `temperature` por proveedor, `thinking_level`, rate limiter compartido, SDK sin reintentos internos, API key faltante y rangos inválidos

## Ejecución contra la API real

```bash
python test_pipeline.py
```

El script:
1. Procesa 4 textos **en paralelo** con `asyncio.gather`: un log crítico, una advertencia menor, una arquitectura con degradación de rendimiento y un texto ambiguo como prueba de estrés.
2. Fuerza `max_tokens=16` para demostrar que el truncamiento se detecta y no se reintenta.

El texto ambiguo no menciona tecnologías concretas, y lo correcto es que el pipeline lance `SinTecnologiasError`: no devuelve una lista vacía ni una tecnología inventada. El script lo cuenta como resultado esperado. Si en cambio devuelve un objeto, lo marca como `INESPERADO`.

## Uso desde código

```python
import asyncio
from dotenv import load_dotenv
from chain import process_text

load_dotenv()
resultado = asyncio.run(process_text("Redis devolvió OOM command not allowed y la API de pagos en Django falló"))
print(resultado.model_dump_json(indent=2))
```
