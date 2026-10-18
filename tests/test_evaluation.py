from datetime import date

import pytest
from agents.testing import ScriptedModel, assistant_message, function_call
from conftest import FakeWeather

from tenerife_rag.assistant import TenerifeAssistant
from tenerife_rag.chunking import Chunk
from tenerife_rag.evaluation import (
    EVAL_DIR,
    evaluate,
    evaluate_assistant,
    expected_date,
    first_relevant_rank,
    load_cases,
    pass_rates,
    score_case,
)
from tenerife_rag.store import build_store

TODAY = date(2026, 10, 9)
PLAN = {
    "id": "plan",
    "pregunta": "¿Atardecer en Benijo mañana?",
    "herramientas": ["search_tenerife_guide", "get_weather"],
    "fecha": "+1",
    "palabras": ["Benijo"],
    "cita": True,
}


def test_first_relevant_rank_ignores_case_and_line_breaks():
    chunks = [Chunk(id="1", text="Otra playa."), Chunk(id="2", text="La Tejita es playa\nNUDISTA.")]

    assert first_relevant_rank(chunks, "playa nudista") == 2
    assert first_relevant_rank(chunks, "volcán") is None


def test_evaluate_computes_hit_at_k_and_mrr(embeddings):
    store = build_store(
        [
            Chunk(id="1", text="playa de arena negra en benijo"),
            Chunk(id="2", text="pizzeria en la orotava"),
        ],
        embeddings,
    )
    questions = [
        {"pregunta": "playa de arena negra", "esperado": "benijo"},  # rank 1
        {"pregunta": "playa de arena negra", "esperado": "pizzeria"},  # rank 2
    ]

    assert evaluate(store, questions, ks=(1, 2)) == {"hit@1": 0.5, "hit@2": 1.0, "MRR": 0.75}


def test_expected_date_is_relative_to_today_or_literal():
    assert expected_date("+1", TODAY) == "2026-10-10"
    assert expected_date("2024-01-01", TODAY) == "2024-01-01"


def test_score_case_checks_tools_date_words_and_citation():
    calls = [("search_tenerife_guide", {"query": "Benijo"}), ("get_weather", {"fecha": "2026-10-10", "lugar": "anaga"})]

    checks = score_case(PLAN, "Sí, Benijo tiene atardeceres espectaculares [S1, S2].", calls, TODAY)

    assert checks == {"herramientas": True, "fecha": True, "palabras": True, "cita": True}


def test_score_case_fails_wrong_date_and_missing_tool():
    checks = score_case(PLAN, "Benijo [S1].", [("get_weather", {"fecha": "2026-10-09"})], TODAY)

    assert not checks["herramientas"]
    assert not checks["fecha"]


@pytest.mark.parametrize("answer", ["Solo puedo ayudarte con Tenerife.", "Esa información no aparece en la guía."])
def test_score_case_recognises_a_limit(answer):
    assert score_case({"limite": True}, answer, [], TODAY) == {"limite": True}


def test_eval_sets_are_well_formed():
    questions = load_cases("questions.json")
    cases = load_cases("assistant_cases.json")

    assert len(questions) >= 20
    assert all({"id", "pregunta", "esperado"} <= q.keys() for q in questions)
    assert len({c["id"] for c in cases}) == len(cases) >= 8
    assert (EVAL_DIR / "questions.json").exists()


async def test_evaluate_assistant_runs_each_case_in_a_new_conversation(config, store):
    model = ScriptedModel(
        [
            [function_call("get_weather", {"fecha": "2026-10-10", "lugar": "teide"}, call_id="call_1")],
            [assistant_message("Despejado en el Teide.")],
            [assistant_message("No lo sé.")],
        ]
    )
    case = {"id": "teide", "pregunta": "¿Tiempo mañana en el Teide?", "herramientas": ["get_weather"], "fecha": "+1"}
    assistants = []

    def make_assistant():
        assistants.append(TenerifeAssistant(config, store, FakeWeather(), model=model, today_fn=lambda: TODAY))
        return assistants[-1]

    results = await evaluate_assistant(make_assistant, [case], runs=2, today_fn=lambda: TODAY)

    assert [r["ok"] for r in results] == [True, False]
    assert pass_rates(results) == {"teide": 0.5}
    assert len(assistants) == 2
    assert all(len(a.memory.turns) == 1 for a in assistants)
