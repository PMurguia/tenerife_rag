# Informe · Guía Tenerife v1

## 1. Objetivo

Un asistente conversacional que responde preguntas turísticas sobre Tenerife a partir de una guía en PDF
(`src/tenerife_rag/data/TENERIFE.pdf`, 25 páginas), con estos requisitos:

- respuestas basadas en la guía y con citas a sus fragmentos;
- predicción del tiempo para planes concretos ("¿merece la pena ir mañana a Benijo?");
- conversación de varios turnos con control de la longitud del historial;
- uso desde un notebook y desde una interfaz web.

## 2. Diseño de la solución

```
TENERIFE.pdf
  │  extraction.py   extract_lines: líneas con posición, negrita y viñeta; extract_images: fotos con su posición
  ▼
chunking.py          structural_chunks: un chunk por lugar, con su ruta (zona › localidad › lugar) y sus fotos
  │  store.py        InMemoryVectorStore (embeddings de OpenAI, similitud coseno exacta)
  ▼
pipeline.build_index ── el mismo índice para el notebook y la app
  │
  ▼
TenerifeAssistant (assistant.py)
  ├─ Agent (Agents SDK) con instrucciones + fecha de hoy en Canarias
  │    ├─ search_tenerife_guide(query, zona, categoria, k) → fragmentos etiquetados [S1], [S2]…
  │    └─ get_weather(fecha, lugar)                       → Open-Meteo, errores como datos
  ├─ ConversationMemory: preguntas y respuestas sin citas, recortadas por tokens
  ├─ ask_stream: texto en streaming (Runner.run_streamed)
  └─ logs/turns.jsonl: herramientas, tokens, latencia y memoria por turno
  │
  ▼
app.py (Gradio)      un asistente por sesión, fuentes citadas bajo cada respuesta, galería con las fotos
                     de los lugares citados, errores comprensibles
```

**Chunking estructural.** La guía tiene una jerarquía visual estable: zonas en negrita, localidades con `•`, lugares con
`o`, restaurantes con `▪` y continuaciones con `-`. Una máquina de estados recorre las líneas y decide el nivel por la
viñeta y la sangría. El resultado son 47 chunks (16 puntos de interés, 13 secciones, 11 restaurantes, 6 rutas y un chunk
de curiosidades). Cada uno guarda su ruta, por ejemplo `Zona Norte > Parque rural de Anaga > Playa de Benijo`. Esa ruta
se antepone al texto que se indexa, sirve de cita legible y permite filtrar por zona (`path[0]`) y por categoría
(`kind == "restaurante"`).

**RAG como herramienta.** El modelo decide cuándo buscar y con qué filtros, en lugar de recibir siempre los mismos
fragmentos. Cada resultado lleva una etiqueta (`S1`, `S2`…) que el modelo cita y que la app traduce a la ruta y la página.

**Tiempo.** `WeatherService` consulta Open-Meteo para 12 lugares de la isla. El Teide usa la altura de Las Cañadas
(2150 m) porque el modelo de terreno de Open-Meteo da otra altura. Las fechas pasadas, las que quedan fuera del horizonte,
los timeouts y los argumentos inválidos se devuelven como datos (`ok: false` y un código), así que la conversación no se
rompe y el modelo puede explicar el problema.

**Fotos.** Las 37 fotos de la guía se leen con su página y su altura, y se intercalan con las líneas en orden de lectura:
cada foto va al chunk abierto cuando aparece (`Chunk.images`), con la misma regla que el texto. Las 37 quedan asignadas.
No se indexan. `export_images` las guarda en el directorio temporal, desde donde Gradio puede servirlas, y la app muestra
en una galería, al lado del chat, las de los lugares citados en la última respuesta (como mucho 4).

**Memoria.** Solo se guardan la pregunta y la respuesta final, sin citas ni salidas de herramientas. Cuando se supera el
presupuesto (2800 tokens por defecto, contados con `tiktoken`), se descartan los turnos más antiguos.

**Seguridad del prompt.** Las instrucciones empiezan con un bloque de alcance: solo turismo en Tenerife, reglas que no
cambian aunque el mensaje diga venir de un administrador, y el texto de la guía y de las herramientas tratado como datos.

## 3. Decisiones

| # | Decisión | Alternativas | Motivo | Consecuencias |
|---|---|---|---|---|
| D-01 | `uv` + `pyproject.toml` con layout `src/` | `requirements.txt` + `venv` | Un comando para instalar y lockfile reproducible | `--system-certs` detrás de un proxy con inspección TLS |
| D-02 | Agents SDK de OpenAI (`gpt-4.1-mini`) | Bucle manual con la Responses API | El SDK hace el bucle, genera el schema strict desde la firma y empareja los `call_id`; el bucle manual queda en el notebook (7) para explicarlo | Menos control directo sobre cada ronda |
| D-03 | Prototipo en el notebook y después módulos | Módulos desde el principio | Entender el problema antes de diseñar la estructura | Un refactor intermedio |
| D-04 | `InMemoryVectorStore` de LangChain | FAISS, Chroma | 47 chunks: búsqueda exacta sin servicios externos | No escala a corpus grandes (no lo necesita) |
| D-05 | Chunking estructural con cabecera | Ventanas fijas | Un lugar por chunk, citas legibles y filtros por zona | El parser depende del formato de esta guía (`MARKER_X_RANGE`) |
| D-06 | Ventanas fijas como baseline | Sin baseline | Sin comparación no se puede decir qué aporta la estructura | — |
| D-07 | RAG como herramienta | Recuperación fija en cada turno | El modelo decide cuándo y cuántas veces buscar | A veces responde sin buscar; se mide en la evaluación |
| D-08 | Fotos de la guía solo para mostrar: cada una va al chunk abierto cuando aparece en el PDF | Indexarlas con descripciones generadas por visión | Mostrar la foto del lugar citado no necesita visión ni embeddings | No se puede buscar por el contenido de una foto; en Playa de Troya algunas son de los locales de la zona |
| D-09 | Errores como datos (`ok: false` + código) | Excepciones | La conversación nunca se rompe; el modelo explica el fallo | Hay que mantener la lista de códigos |
| D-10 | `Literal` como única fuente de los valores válidos | Listas duplicadas en schema y validación | El mismo tipo da el `enum` del schema strict y la validación | — |
| D-11 | Memoria sin salidas de herramientas ni citas | Guardar todo (`to_input_list()`) | Historial pequeño y sin etiquetas obsoletas | El modelo repite búsquedas si necesita un dato antiguo |
| D-12 | Recorte por tokens sin resumen | Resumir con el LLM lo que sale | Cumple el control de longitud con lo mínimo | Lo antiguo se olvida del todo |
| D-13 | `truststore` solo en los puntos de entrada | Dentro de `weather.py` | Es un efecto global: lo decide quien arranca | Un script propio necesita su propia llamada |
| D-14 | Una sola gramática de citas (`citations.py`) | Una regex en cada módulo | `[S1]`, `[S1][S2]`, `[S1, S2]` y `[S1; S2]` se reconocen igual en memoria, evaluación y app | — |
| D-15 | Gradio con `gr.Blocks` y `gr.State` | `ChatInterface`, Streamlit | Un asistente con memoria propia por sesión, fuentes bajo la respuesta y galería de fotos al lado del chat | Más código que `ChatInterface` |
| D-16 | Sin filtro de entrada aparte | Clasificador previo | El bloque de alcance cubre los ataques medidos | Una sola capa de defensa |
| D-17 | Sin caché de embeddings | Caché en disco | Indexar 47 chunks cuesta céntimos y segundos | Cada arranque vuelve a indexar |
| D-18 | Regla explícita para preguntas de plan | Dejar que el modelo decida | Sin ella, "¿merece la pena ir mañana a Benijo?" solo consultaba el tiempo (0 de 3) | A veces consulta el tiempo sin que se lo pidan |

## 4. Resultados

Ejecución del notebook con `gpt-4.1-mini` y `text-embedding-3-large`. El modelo de embeddings sale de `.env`; el valor
por defecto en `config.py` es `text-embedding-3-small`.

### 4.1. Recuperación (notebook 10)

24 preguntas, 13 literales y 11 paráfrasis. Una pregunta acierta si un fragmento recuperado contiene la frase esperada.

| Configuración | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| Ventanas de 800 caracteres | 0,83 | 0,92 | 1,00 | 0,89 |
| Ventanas de 400 caracteres | 0,83 | 0,96 | 1,00 | 0,90 |
| Estructural sin cabecera | 0,83 | 0,96 | 0,96 | 0,89 |
| Estructural con cabecera | 0,79 | 0,96 | 0,96 | 0,88 |

Las cuatro configuraciones empatan dentro del ruido: con 24 preguntas, cada una vale 0,04. La única diferencia en hit@5
es "¿Dónde puedo comprar gofio?", una línea dentro del chunk largo de La Orotava. La cabecera solo mueve dos preguntas una
posición, una a mejor y otra a peor. El chunking estructural gana en lo que entrega al modelo y al
usuario (un lugar por chunk, filtros y citas legibles).

### 4.2. Asistente (notebook 11)

10 casos, cada uno en una conversación nueva y ejecutado 3 veces. Se puntúa el recorrido: herramientas usadas, fecha
pedida a `get_weather`, palabras clave, citas y si reconoce el límite.

| Caso | Aciertos |
|---|---|
| Plan con tiempo (Benijo mañana) | 3/3 |
| Dato literal | 3/3 |
| Comer en el norte (filtros) | 3/3 |
| Solo tiempo | 3/3 |
| Dato fuera de su sección (altura del Teide) | 0/3 |
| Fecha pasada | 3/3 |
| Fuera de la guía | 3/3 |
| Fuera de tema | 3/3 |
| Ataque "ignora tus instrucciones" | 3/3 |
| Ataque "como administrador" | 3/3 |

El único fallo sistemático es la altura del Teide (3.718 m). El dato está en las curiosidades de La Orotava, y la búsqueda
lo trae en segunda posición, pero el modelo responde que la guía no lo menciona.

### 4.3. Observabilidad (notebook 12)

En una conversación de 10 turnos, la memoria llega a 1024 tokens de 2800 y no se descarta ningún turno. Las instrucciones
y los esquemas de las herramientas cuestan unos 730 tokens por ronda. Un turno con búsqueda hace dos rondas y suma los
fragmentos, así que `input_tokens` pasa de unos 1600 a más de 3400. El coste lo marcan las herramientas, no el historial.

## 5. Limitaciones

- El chunker depende del formato de esta guía: sangrías, negritas y viñetas.
- La evaluación es pequeña (24 preguntas, 10 casos × 3 ejecuciones) y se hace con un modelo con temperatura > 0.
- El modelo no siempre sigue las reglas de formato: a veces cita el tiempo con corchetes inventados. La app no los lista
  como fuente, pero aparecen en el texto.
- La regla de "plan" a veces consulta el tiempo cuando la pregunta no lo pide.
- Los datos que están en la sección de otro lugar tienden a no usarse.
- Una sola capa frente a *prompt injection*: resiste los ataques medidos, no cualquier ataque.
- La memoria olvida del todo lo que sale del presupuesto.
- Las fotos se muestran pero no se indexan; las de un lugar con muchos locales (Playa de Troya) pueden ser de esos locales.
- Sin modo offline y sin caché de embeddings.
- En redes con inspección TLS, Open-Meteo necesita `truststore`.

## 6. Mejoras futuras

| Mejora | Qué aporta | Dónde se engancha |
|---|---|---|
| Resumen de los turnos descartados | Lo antiguo no se olvida del todo | `ConversationMemory._trim` |
| Caché de embeddings en disco | Arranques sin coste | `pipeline.build_index` |
| Filtro de entrada (guard) | Segunda capa frente a *prompt injection* | antes de `TenerifeAssistant.ask_stream` |
| Resolución de "allí" a un lugar de `PLACES` | Menos ambigüedad en el tiempo | `weather.PLACES` |
| Descripciones de las fotos generadas con visión | Buscar por lo que muestran las fotos | `Chunk.images` |
| Rangos de citas `[S1-S3]` | Otra forma que usa el modelo | `citations.py` |
| Más preguntas y casos de evaluación | Métricas con menos ruido; datos fuera de su sección | `src/tenerife_rag/eval/` |

## 7. Cómo reproducir

Ver el [README](README.md): `uv sync`, copiar `.env.example` a `.env` con la key, y después `uv run pytest`,
`uv run python app.py` o ejecutar el notebook de principio a fin.
