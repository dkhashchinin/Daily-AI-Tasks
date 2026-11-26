"""FastMCP-сервер rates-store: три инструмента для планировщика.

  fetch_rates()            — получение данных: курсы ЦБ РФ прямо сейчас;
  save_snapshot(payload)   — сохранение: пишет снимок в SQLite;
  read_snapshots(hours)    — чтение накопленного за период в JSON.

Схемы инструментов FastMCP строит сам из сигнатур и docstring'ов.
Все инструменты возвращают JSON-строки — их одинаково удобно читать и
планировщику, и LLM-агенту.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

from . import store
from .rates import RatesError, fetch_latest_rates

load_dotenv(ROOT / ".env")

mcp = FastMCP("rates-store")


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


@mcp.tool()
def fetch_rates(currencies: str = "USD,EUR,CNY,GBP") -> str:
    """Получить актуальные курсы ЦБ РФ по выбранным валютам.

    Args:
        currencies: коды валют ЦБ через запятую, до 8 штук,
            например "USD,EUR" или "USD,JPY,KZT". Пусто — USD, EUR, CNY, GBP.
            Коды, которых нет в справочнике ЦБ, вернутся в поле unknown.
    """
    try:
        return _dumps(fetch_latest_rates(currencies))
    except RatesError as exc:
        return _dumps({"error": str(exc)})


@mcp.tool()
def save_snapshot(payload: str) -> str:
    """Сохранить снимок данных в SQLite (один запуск планировщика = одна строка).

    Args:
        payload: JSON-строка, обычно результат fetch_rates
            ({"fetched_at": ..., "rates": {"USD": ..., "EUR": ...}}).
            Поля rates.USD/EUR/CNY/GBP раскладываются по колонкам,
            весь payload целиком сохраняется в колонке payload.
    """
    try:
        data = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        return _dumps({"error": f"payload должен быть JSON-строкой: {exc}"})
    if not isinstance(data, dict):
        return _dumps({"error": "payload должен быть JSON-объектом, а не массивом/скаляром"})
    saved = store.save_snapshot(data)
    saved["saved"] = True
    # В подтверждении возвращаем все валюты из payload, а не только
    # фиксированные колонки: набор может меняться от запуска к запуску.
    saved["rates"] = data.get("rates") or {}
    return _dumps(saved)


@mcp.tool()
def read_snapshots(hours: float = 24.0) -> str:
    """Прочитать накопленные снимки за период и вернуть их JSON-списком.

    Args:
        hours: глубина периода в часах; 0 — вся накопленная история.
            Возвращаются снимки от старых к новым: id, ts, usd, eur, cny, gbp.
    """
    rows = store.read_snapshots(hours)
    return _dumps({"period_hours": hours, "count": len(rows), "snapshots": rows})


def main() -> None:
    store.init_db()
    mcp.run()
