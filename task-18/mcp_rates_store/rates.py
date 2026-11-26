"""Получение данных: актуальные курсы ЦБ РФ через публичный JSON-API cbr-xml-daily.ru.

Отдельный модуль (не инструмент), чтобы сервер мог переиспользовать его в
fetch_rates. httpx уже идёт транзитивной зависимостью MCP SDK.
"""
import httpx

CBR_DAILY_URL = "https://www.cbr-xml-daily.ru/daily_json.js"

# Валюты по умолчанию, если параметр currencies не задан.
DEFAULT_CURRENCIES = ("USD", "EUR", "CNY", "GBP")

# Ограничение на количество валют в одном снимке (колонки в отчётах и UI).
_MAX_CURRENCIES = 8

_TIMEOUT = 15.0


class RatesError(RuntimeError):
    """Не удалось получить курсы."""


def parse_currencies(spec: str | None) -> list[str]:
    """Разобрать строку вида 'usd, eur ,JPY' -> ['USD', 'EUR', 'JPY'].

    Пустая строка/None -> валюты по умолчанию. Дубликаты отбрасываются,
    количество ограничено _MAX_CURRENCIES.
    """
    if spec is None or not str(spec).strip():
        return list(DEFAULT_CURRENCIES)
    codes: list[str] = []
    seen: set[str] = set()
    for chunk in str(spec).replace(";", ",").split(","):
        code = chunk.strip().upper()
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes[:_MAX_CURRENCIES]


def fetch_latest_rates(currencies: str | list[str] | None = None) -> dict:
    """Скачать дневной справочник ЦБ и вернуть плоский dict с курсами.

    currencies — строка с кодами через запятую или список; запрашиваются
    только они. Неизвестные ЦБ коды попадут в поле unknown, известные —
    в rates. Если ни одной валюты найти не удалось, бросает RatesError.
    """
    codes = parse_currencies(
        ",".join(currencies) if isinstance(currencies, (list, tuple)) else currencies
    )
    try:
        response = httpx.get(CBR_DAILY_URL, timeout=_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise RatesError(f"HTTP-ошибка при запросе курсов ЦБ: {exc}") from exc
    except ValueError as exc:
        raise RatesError(f"ЦБ вернул не-JSON ответ: {exc}") from exc

    valute = data.get("Valute") or {}
    rates: dict[str, float] = {}
    unknown: list[str] = []
    for code in codes:
        item = valute.get(code)
        if item and item.get("Value") is not None:
            rates[code] = round(float(item["Value"]), 4)
        else:
            unknown.append(code)
    if not rates:
        raise RatesError(f"в ответе ЦБ нет ни одной из валют: {', '.join(codes)}")

    return {
        "fetched_at": data.get("Date", ""),
        "cbr_date": data.get("PreviousDate", ""),
        "rates": rates,
        "unknown": unknown,
    }
