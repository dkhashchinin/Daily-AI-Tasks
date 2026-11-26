"""MCP-сервер rates-store: данные -> хранилище -> чтение за период (транспорт stdio).

Запуск: python -m mcp_rates_store

Три инструмента, которые дёргает планировщик:
  - fetch_rates    — получение данных: актуальные курсы ЦБ РФ (USD, EUR, CNY, ...);
  - save_snapshot  — сохранение: один запуск = один снимок в SQLite (data/rates.db);
  - read_snapshots — чтение накопленного за период (N часов) в виде JSON.

Путь к БД: переменная окружения RATES_DB_PATH, по умолчанию <корень проекта>/data/rates.db.
"""

from .server import mcp, main

__all__ = ["mcp", "main"]
