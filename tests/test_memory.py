from datetime import date

import tiktoken
from agents.testing import ScriptedModel, assistant_message

from conftest import FakeWeather
from tenerife_rag.assistant import TenerifeAssistant
from tenerife_rag.config import LLMConfig
from tenerife_rag.memory import ConversationMemory

ENCODING = tiktoken.get_encoding("o200k_base")


def turn_tokens(user: str, assistant: str) -> int:
    return len(ENCODING.encode(user + assistant))


TURNS = [
    ("¿Cómo es Benijo?", "Playa de arena negra con atardeceres."),
    ("¿Y qué tiempo hará allí mañana?", "Despejado, 24 °C."),
    ("¿Y en el sur?", "Soleado en Costa Adeje, 27 °C."),
]


def test_empty_memory_has_no_input_and_no_tokens():
    memory = ConversationMemory()

    assert memory.as_input() == []
    assert memory.tokens() == 0


def test_as_input_alternates_user_and_assistant_in_order():
    memory = ConversationMemory()
    for user, assistant in TURNS[:2]:
        memory.add(user, assistant)

    assert memory.as_input() == [
        {"role": "user", "content": TURNS[0][0]},
        {"role": "assistant", "content": TURNS[0][1]},
        {"role": "user", "content": TURNS[1][0]},
        {"role": "assistant", "content": TURNS[1][1]},
    ]


def test_add_strips_citations_from_answer():
    memory = ConversationMemory()

    memory.add("¿Cómo es Benijo?", "Arena negra [S1] y atardeceres [S12].")

    assert memory.turns[0].assistant == "Arena negra y atardeceres."


def test_add_strips_grouped_citations():
    memory = ConversationMemory()

    memory.add("¿Cómo es Benijo?", "Arena negra [S1, S2] y atardeceres [S3; S4].")

    assert memory.turns[0].assistant == "Arena negra y atardeceres."


def test_tokens_sums_every_turn():
    memory = ConversationMemory()
    for user, assistant in TURNS:
        memory.add(user, assistant)

    assert memory.tokens() == sum(turn_tokens(u, a) for u, a in TURNS)


def test_keeps_everything_while_under_budget():
    memory = ConversationMemory(budget_tokens=10_000)
    for user, assistant in TURNS:
        memory.add(user, assistant)

    assert len(memory.turns) == 3
    assert memory.dropped == 0


def test_add_drops_oldest_turns_when_over_budget():
    memory = ConversationMemory(budget_tokens=turn_tokens(*TURNS[1]) + turn_tokens(*TURNS[2]))
    for user, assistant in TURNS:
        memory.add(user, assistant)

    assert [t.user for t in memory.turns] == [TURNS[1][0], TURNS[2][0]]
    assert memory.dropped == 1
    assert memory.tokens() <= memory.budget


def test_keeps_last_turn_even_if_it_exceeds_budget_alone():
    memory = ConversationMemory(budget_tokens=1)
    for user, assistant in TURNS:
        memory.add(user, assistant)

    assert [t.user for t in memory.turns] == [TURNS[2][0]]
    assert memory.dropped == 2


def test_assistant_uses_budget_from_config(store):
    assistant = TenerifeAssistant(LLMConfig(history_token_budget=1234), store, FakeWeather())

    assert assistant.memory.budget == 1234


async def test_ask_sends_history_and_stores_only_question_and_answer(config, store):
    model = ScriptedModel([
        [assistant_message("Playa de arena negra [S1].")],
        [assistant_message("Despejado, 24 °C.")],
    ])
    assistant = TenerifeAssistant(config, store, FakeWeather(), model=model, today_fn=lambda: date(2026, 10, 9))

    await assistant.ask("¿Cómo es Benijo?")
    await assistant.ask("¿Y qué tiempo hará allí mañana?")

    messages = [(i["role"], i["content"]) for i in model.last_call.input if "role" in i]
    assert messages == [
        ("user", "¿Cómo es Benijo?"),
        ("assistant", "Playa de arena negra."),  # previous turn's labels are not resent
        ("user", "¿Y qué tiempo hará allí mañana?"),
    ]
    assert assistant.memory.as_input()[-1] == {"role": "assistant", "content": "Despejado, 24 °C."}
