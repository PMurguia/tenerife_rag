from datetime import date
from typing import Any

import pytest
import requests

from tenerife_rag.weather import WeatherService

TODAY = date(2026, 10, 9)


class FakeResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self.payload = payload

    def json(self) -> Any:
        return self.payload


class FakeHttpGet:
    """Returns (or raises) the queued outcomes in order and records each call."""

    def __init__(self, *outcomes: FakeResponse | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def forecast_payload(day: str) -> dict[str, Any]:
    return {
        "daily": {
            "time": [day],
            "weather_code": [2],
            "temperature_2m_max": [24.1],
            "temperature_2m_min": [18.3],
            "precipitation_probability_max": [10],
            "precipitation_sum": [0.0],
            "wind_speed_10m_max": [21.5],
        }
    }


def make_service(http_get: FakeHttpGet) -> WeatherService:
    return WeatherService(http_get=http_get, today_fn=lambda: TODAY)


def test_parses_successful_response():
    http_get = FakeHttpGet(FakeResponse(200, forecast_payload("2026-10-10")))

    result = make_service(http_get).get_weather("2026-10-10", "santa_cruz")

    assert result == {
        "ok": True,
        "fuente": "Open-Meteo",
        "modo": "real",
        "fecha": "2026-10-10",
        "lugar_id": "santa_cruz",
        "lugar": "Santa Cruz de Tenerife",
        "estado": "parcialmente nublado",
        "weather_code": 2,
        "temp_max_c": 24.1,
        "temp_min_c": 18.3,
        "prob_precipitacion_pct": 10,
        "precipitacion_mm": 0.0,
        "viento_max_kmh": 21.5,
    }
    assert http_get.calls[0]["params"]["start_date"] == "2026-10-10"


def test_rejects_past_date_without_calling_api():
    http_get = FakeHttpGet()

    result = make_service(http_get).get_weather("2026-10-08", "santa_cruz")

    assert result["ok"] is False
    assert result["error"] == "fecha_pasada"
    assert http_get.calls == []


@pytest.mark.parametrize("fecha", ["2026-10-25", "2026-11-08"])
def test_rejects_date_beyond_forecast_horizon(fecha):
    http_get = FakeHttpGet()

    result = make_service(http_get).get_weather(fecha, "santa_cruz")

    assert result["ok"] is False
    assert result["error"] == "fuera_de_horizonte"
    assert "2026-10-24" in result["message"]
    assert http_get.calls == []


def test_retries_after_timeout_and_succeeds():
    http_get = FakeHttpGet(
        requests.Timeout("slow"),
        FakeResponse(200, forecast_payload("2026-10-10")),
    )

    result = make_service(http_get).get_weather("2026-10-10", "teide")

    assert result["ok"] is True
    assert len(http_get.calls) == 2


def test_returns_timeout_error_when_retries_are_exhausted():
    http_get = FakeHttpGet(requests.Timeout("slow"), requests.Timeout("slow"))

    result = make_service(http_get).get_weather("2026-10-10", "teide")

    assert result["ok"] is False
    assert result["error"] == "timeout"
    assert len(http_get.calls) == 2


def test_bad_request_returns_api_reason_without_retrying():
    http_get = FakeHttpGet(FakeResponse(400, {"error": True, "reason": "Parameter 'daily' is invalid"}))

    result = make_service(http_get).get_weather("2026-10-10", "masca")

    assert result["ok"] is False
    assert result["error"] == "error_api"
    assert "Parameter 'daily' is invalid" in result["message"]
    assert len(http_get.calls) == 1
