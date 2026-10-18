# Guía Tenerife

Asistente turístico conversacional sobre una guía de Tenerife en PDF. Responde preguntas sobre lugares, planes y
restaurantes citando los fragmentos de la guía, consulta la predicción del tiempo de Open-Meteo y recuerda la
conversación dentro de un presupuesto de tokens. Se usa desde un notebook o desde una interfaz web con Gradio.

## Qué hace

- **RAG con chunks estructurales**: la guía se trocea por su jerarquía (zona › localidad › lugar), cada chunk se indexa
  con su ruta como cabecera y las respuestas citan los fragmentos (`[S1]`).
- **Dos herramientas** (Agents SDK de OpenAI): `search_tenerife_guide`, con filtros opcionales de zona y categoría, y
  `get_weather` (Open-Meteo, sin API key), con errores devueltos como datos.
- **Fotos de la guía**: la app muestra, en una galería al lado del chat, las fotos de los lugares citados (como mucho 4).
- **Paso a paso**: debajo de las fotos, la app muestra en directo cada paso del turno: la memoria, cada ronda del
  modelo (tokens y qué decide), las búsquedas en la guía (filtros, fragmentos y puntuación), el tiempo, las citas y las
  fotos.
- **Memoria** de conversación sin citas ni salidas de herramientas, recortada por tokens.
- **Streaming**, un log JSONL por turno (herramientas, tokens y latencia) y mensajes de error útiles.
- **Evaluación reproducible**: hit@k y MRR de la recuperación, y casos del asistente (incluidos ataques de
  *prompt injection*) ejecutados varias veces.

## Requisitos

- Python 3.12 o superior y [uv](https://docs.astral.sh/uv/).
- Una API key de OpenAI.

## Instalación

```bash
uv sync
cp .env.example .env        # PowerShell: Copy-Item .env.example .env
```

Edita `.env` y pon tu `OPENAI_API_KEY`. El resto de variables (`OPENAI_GENERATION_MODEL`, `OPENAI_TEMPERATURE`,
`OPENAI_HISTORY_TOKEN_BUDGET`…) son opcionales y tienen valores por defecto en `src/tenerife_rag/config.py`.

## Ejecución

```bash
uv run python app.py                                   # interfaz web en http://127.0.0.1:7860
uv run --with jupyterlab jupyter lab src/tenerife_rag/notebooks/tenerife_rag.ipynb
uv run pytest                                          # 127 tests, sin red ni API
uv run ruff check src tests app.py
```

Para ejecutar el notebook de principio a fin sin abrirlo:

```bash
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 src/tenerife_rag/notebooks/tenerife_rag.ipynb
```

La ejecución completa llama a la API (la evaluación del asistente son 30 conversaciones) y tarda unos 4 minutos.

Si falla al arrancar con `FileNotFoundError`, el kernel `python3` registrado en tu usuario apunta a un entorno que ya no
existe (`jupyter kernelspec list` muestra dónde está). Bórralo con `jupyter kernelspec remove python3` y se usará el del
proyecto.

## Certificados (proxy o antivirus con inspección TLS)

Si `uv` falla con `UnknownIssuer` al descargar paquetes, añade `--system-certs` (por ejemplo `uv sync --system-certs`).

En tiempo de ejecución, `app.py` y la primera celda del notebook llaman a `truststore.inject_into_ssl()`, que hace que
Python use el almacén de certificados del sistema. Sin esa llamada, Open-Meteo falla con `SSLError` en esas redes. Un
script propio que use el paquete tiene que hacer la misma llamada al arrancar.

## Estructura

```
app.py                          interfaz web (Gradio)
app.css                         estilos de la interfaz web
src/tenerife_rag/
  config.py                     configuración desde .env
  extraction.py                 texto y líneas del PDF (posición, negrita, viñeta)
  chunking.py                   Chunk, chunks estructurales y ventanas fijas
  store.py                      índice vectorial en memoria y búsqueda
  pipeline.py                   build_index: el índice de la guía
  weather.py                    WeatherService (Open-Meteo) y lugares de la isla
  tools.py                      herramientas del agente y log de llamadas
  memory.py                     memoria con presupuesto de tokens
  citations.py                  gramática de las citas [S1]
  trace.py                      Step: los pasos de un turno para el visor
  assistant.py                  TenerifeAssistant: agente, streaming y log por turno
  evaluation.py                 hit@k, MRR y evaluación del asistente
  eval/                         preguntas y casos de evaluación (JSON)
  data/TENERIFE.pdf             la guía
  notebooks/tenerife_rag.ipynb  recorrido completo con resultados
tests/                          tests con modelo, embeddings y HTTP simulados
INFORME.md                      diseño, decisiones, resultados y limitaciones
```

Los logs se escriben en `logs/` (relativo al directorio de trabajo): `tool_calls.log` con cada llamada a herramienta y
cada reintento fallido de Open-Meteo, y `turns.jsonl` con un registro por turno.
