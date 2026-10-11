"""Predicción del tiempo para Tenerife con Open-Meteo (sin API key).

``get_weather(fecha, lugar)`` devuelve siempre un diccionario serializable:
``ok=True`` con la predicción, u ``ok=False`` con un código de error y un mensaje
que el modelo puede explicar al usuario sin inventar datos.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal, get_args
from zoneinfo import ZoneInfo

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

TIMEZONE = "Atlantic/Canary"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_FORECAST_DAYS = 15  # Open-Meteo da hasta 16 días contando hoy
ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class Place:
    """Punto de la guía con coordenadas aproximadas."""

    label: str
    latitude: float
    longitude: float
    elevation: float | None = None  # para el Teide: corrige la temperatura a 2.150 m
    section: str = ""  # encabezado de la localidad en la guía (enlaza lugares citados con su zona)
    aliases: tuple[str, ...] = ()  # nombres extra con los que se menciona la zona en las respuestas


PLACES: dict[str, Place] = {
    "santa_cruz": Place(
        "Santa Cruz de Tenerife", 28.4682, -16.2546, section="Santa Cruz de Tenerife", aliases=("Santa Cruz",)
    ),
    "la_laguna": Place("San Cristóbal de La Laguna", 28.4853, -16.3201, section="San Cristóbal de La Laguna"),
    "anaga": Place("Parque rural de Anaga (Benijo)", 28.5705, -16.1856, section="Parque rural de Anaga"),
    "la_orotava": Place("La Orotava", 28.3905, -16.5230, section="La Orotava"),
    "puerto_de_la_cruz": Place("Puerto de la Cruz", 28.4142, -16.5484, section="Puerto de La Cruz"),
    "isla_baja": Place("Isla Baja (Garachico)", 28.3734, -16.7637, section="Isla Baja"),
    "teide": Place(
        "Parque Nacional del Teide (Las Cañadas)",
        28.2224,
        -16.6401,
        elevation=2150,
        section="El Teide",
        aliases=(
            "Parque Nacional del Teide",
            "Las Cañadas",
            "Mirador de La Ruleta",
            "Parador",
            "subir al Teide",
            "subida al Teide",
            "Teide de noche",
            "cima del Teide",
        ),
    ),
    "costa_adeje": Place("Costa Adeje", 28.0880, -16.7360, section="Costa Adeje"),
    "los_gigantes": Place("Los Gigantes", 28.2457, -16.8418, section="Los Gigantes"),
    "masca": Place("Masca", 28.3060, -16.8410, section="Masca"),
    "el_medano": Place("El Médano", 28.0448, -16.5370, section="El Médano / La Tejita"),
    "los_cristianos": Place("Los Cristianos", 28.0510, -16.7170, section="Los Cristianos"),
}
DEFAULT_PLACE = "santa_cruz"
PlaceId = Literal[
    "santa_cruz", "la_laguna", "anaga", "la_orotava", "puerto_de_la_cruz", "isla_baja",
    "teide", "costa_adeje", "los_gigantes", "masca", "el_medano", "los_cristianos",
]  # fmt: skip
if set(get_args(PlaceId)) != set(PLACES):  # el Literal (schema/validación) y las coordenadas deben coincidir
    raise RuntimeError(f"PlaceId y PLACES no coinciden: {set(get_args(PlaceId)) ^ set(PLACES)}")

WMO_CODES = {
    0: "despejado", 1: "mayormente despejado", 2: "parcialmente nublado", 3: "cubierto",
    45: "niebla", 48: "niebla con escarcha",
    51: "llovizna débil", 53: "llovizna moderada", 55: "llovizna intensa",
    56: "llovizna helada débil", 57: "llovizna helada intensa",
    61: "lluvia débil", 63: "lluvia moderada", 65: "lluvia fuerte",
    66: "lluvia helada débil", 67: "lluvia helada fuerte",
    71: "nevada débil", 73: "nevada moderada", 75: "nevada fuerte", 77: "granos de nieve",
    80: "chubascos débiles", 81: "chubascos moderados", 82: "chubascos violentos",
    85: "chubascos de nieve débiles", 86: "chubascos de nieve fuertes",
    95: "tormenta", 96: "tormenta con granizo débil", 97: "tormenta fuerte", 99: "tormenta con granizo fuerte",
}  # fmt: skip


class WeatherArgs(BaseModel):
    """Argumentos de ``get_weather`` (validación con Pydantic)."""

    fecha: str = Field(description="Fecha en formato ISO AAAA-MM-DD.")
    lugar: PlaceId | None = Field(default=None, description="Zona de la guía; None = Santa Cruz.")

    @field_validator("fecha")
    @classmethod
    def iso_date(cls, value: str) -> str:
        """Exige el formato AAAA-MM-DD y que la fecha exista."""
        if not ISO_DATE_RE.fullmatch(value):
            raise ValueError("formato AAAA-MM-DD")
        date.fromisoformat(value)  # valida además que el día exista (p. ej. 2026-02-30)
        return value


def canary_today() -> date:
    """Fecha actual en Canarias."""
    return datetime.now(ZoneInfo(TIMEZONE)).date()


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": code, "message": message, **extra}


class WeatherService:
    """Consulta Open-Meteo (modo ``real``) o genera datos deterministas (modo ``simulado``)."""

    def __init__(
        self,
        mode: Literal["real", "simulado"] = "real",
        http_get: Callable[..., Any] = requests.get,
        timeout: float = 8.0,
        retries: int = 1,
        today_fn: Callable[[], date] = canary_today,
    ) -> None:
        """Configura el modo, el cliente HTTP, el timeout, los reintentos y la fecha de referencia."""
        self.mode = mode
        self.http_get = http_get
        self.timeout = timeout
        self.retries = retries
        self.today_fn = today_fn

    # ------------------------------------------------------------------ API pública

    def get_weather(self, fecha: str, lugar: str | None = None) -> dict[str, Any]:
        """Predicción diaria para una fecha y una zona de Tenerife."""
        try:
            args = WeatherArgs(fecha=fecha, lugar=lugar)
        except ValidationError as exc:
            field = exc.errors()[0]["loc"][0]
            if field == "lugar":
                return _error(
                    "lugar_desconocido",
                    f"'{lugar}' no es una zona soportada. Opciones: {', '.join(PLACES)}.",
                    lugar=lugar,
                )
            return _error("fecha_invalida", f"'{fecha}' no es una fecha válida en formato AAAA-MM-DD.", fecha=fecha)

        target = date.fromisoformat(args.fecha)
        today = self.today_fn()
        if target < today:
            return _error("fecha_pasada", f"{args.fecha} ya ha pasado; solo hay predicciones desde hoy.", fecha=fecha)
        if target > today + timedelta(days=MAX_FORECAST_DAYS):
            limit = (today + timedelta(days=MAX_FORECAST_DAYS)).isoformat()
            return _error(
                "fuera_de_horizonte",
                f"Solo hay predicción hasta {limit} (unos 15 días vista).",
                fecha=fecha,
            )

        place_id = args.lugar or DEFAULT_PLACE
        place = PLACES[place_id]
        if self.mode == "simulado":
            return self._simulated(target, place_id, place)
        return self._real(target, place_id, place)

    # ------------------------------------------------------------------ modos

    def _real(self, target: date, place_id: str, place: Place) -> dict[str, Any]:
        params = {
            "latitude": place.latitude,
            "longitude": place.longitude,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max,precipitation_sum,wind_speed_10m_max",
            "timezone": TIMEZONE,
            "start_date": target.isoformat(),
            "end_date": target.isoformat(),
        }
        if place.elevation is not None:
            params["elevation"] = place.elevation

        last_error: dict[str, Any] | None = None
        for attempt in range(1, self.retries + 2):
            try:
                response = self.http_get(FORECAST_URL, params=params, timeout=self.timeout)
            except requests.Timeout:
                last_error = _error("timeout", "Open-Meteo no respondió a tiempo.")
            except requests.RequestException as exc:
                last_error = _error("error_red", f"No se pudo conectar con Open-Meteo: {type(exc).__name__}.")
            else:
                if response.status_code == 400:
                    reason = _safe_json(response).get("reason", "petición no válida")
                    return _error("error_api", f"Open-Meteo rechazó la petición: {reason}")
                if response.status_code >= 500:
                    last_error = _error("error_api", f"Open-Meteo devolvió HTTP {response.status_code}.")
                elif response.status_code != 200:
                    return _error("error_api", f"Open-Meteo devolvió HTTP {response.status_code}.")
                else:
                    return self._parse(_safe_json(response), target, place_id, place)
            logger.warning("Intento %d de Open-Meteo fallido: %s", attempt, last_error["message"])
        return last_error

    def _parse(self, payload: dict, target: date, place_id: str, place: Place) -> dict[str, Any]:
        daily = payload.get("daily") or {}
        try:
            index = daily["time"].index(target.isoformat())
            code = daily["weather_code"][index]
            result = {
                "ok": True,
                "fuente": "Open-Meteo",
                "modo": "real",
                "fecha": target.isoformat(),
                "lugar_id": place_id,
                "lugar": place.label,
                "estado": WMO_CODES.get(code, f"código WMO {code}"),
                "weather_code": code,
                "temp_max_c": daily["temperature_2m_max"][index],
                "temp_min_c": daily["temperature_2m_min"][index],
                "prob_precipitacion_pct": daily["precipitation_probability_max"][index],
                "precipitacion_mm": daily["precipitation_sum"][index],
                "viento_max_kmh": daily["wind_speed_10m_max"][index],
            }
        except (KeyError, ValueError, IndexError, TypeError):
            return _error("respuesta_inesperada", "La respuesta de Open-Meteo no tiene el formato esperado.")
        return result

    @staticmethod
    def _simulated(target: date, place_id: str, place: Place) -> dict[str, Any]:
        """Datos deterministas y plausibles (útil sin conexión o en tests)."""
        seed = int(hashlib.md5(f"{target}{place_id}".encode()).hexdigest(), 16)
        code = [0, 1, 2, 3, 61, 80][seed % 6]
        base = 12 if place_id == "teide" else 23
        return {
            "ok": True,
            "fuente": "simulación local (no es una predicción real)",
            "modo": "simulado",
            "fecha": target.isoformat(),
            "lugar_id": place_id,
            "lugar": place.label,
            "estado": WMO_CODES[code],
            "weather_code": code,
            "temp_max_c": base + seed % 5,
            "temp_min_c": base - 6 + seed % 3,
            "prob_precipitacion_pct": 70 if code >= 61 else (seed % 25),
            "precipitacion_mm": 2.5 if code >= 61 else 0.0,
            "viento_max_kmh": 10 + seed % 25,
        }


def _safe_json(response: Any) -> dict:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}
