from collections.abc import Callable
from datetime import date
from agents import MaxTurnsExceeded

from agents import Agent, Model, Runner, RunResult

from tenerife_rag.chunking import Chunk
from tenerife_rag.config import LLMConfig
from tenerife_rag.tools import TurnContext, get_weather, search_tenerife_guide
from tenerife_rag.weather import canary_today

TURN_LIMIT_MESSAGE = "No he podido completar la respuesta. ¿Puedes concretar un poco más la pregunta?"

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

INSTRUCTIONS = """Eres un guía turístico de Tenerife. Responde en español.
Usa search_tenerife_guide antes de responder sobre lugares, playas, rutas o comida, y responde solo con lo que devuelva.
Usa get_weather cuando pregunten por el tiempo.
Cita cada dato de la guía con el id del fragmento entre corchetes, por ejemplo [S1].
Si la guía no contiene la respuesta, dilo."""

def instructions_for(today: date) -> str:
    return INSTRUCTIONS + f"\nHoy es {DIAS[today.weekday()]} {today.isoformat()}. Convierte fechas relativas a AAAA-MM-DD."

def build_agent(config: LLMConfig, today_fn: Callable[[], date] = canary_today, model: str | Model | None = None) -> Agent[TurnContext]:
    return Agent[TurnContext](
        name="GuiaTenerife",
        instructions= lambda _ctx, _agent: instructions_for(today_fn()),
        model=model or config.generation_model,
        model_settings=config.model_settings(),
        tools=[search_tenerife_guide, get_weather],
    )
    
class TenerifeAssistant:
    
    def __init__(self, config: LLMConfig, store, weather, *, model=None, today_fn=canary_today):
        self.config, self.store, self.weather = config, store, weather
        self.agent = build_agent(config, today_fn, model)
        self.turn = TurnContext(store, weather)
        self.last_result: RunResult | None = None
        
    @property
    def chunks_by_label(self) -> dict[str, Chunk]:
        return self.turn.chunks_by_label
    
    async def ask(self, question: str) -> str:
        self.turn = TurnContext(self.store, self.weather)
        
        try:  
            self.last_result = await Runner.run(
                self.agent,
                question,
                context=self.turn, max_turns=self.config.max_turns
            )
        except MaxTurnsExceeded:
            return TURN_LIMIT_MESSAGE
        
        return self.last_result.final_output