"""MCP-сервер вокруг T-Invest API (Т-Банк Инвестиции), транспорт stdio.

Запуск: python -m mcp_tbank_invest
Токен берётся из переменной окружения INVEST_TOKEN (или из .env рядом с проектом).
Контур: INVEST_TARGET=sandbox (по умолчанию) или prod.

Proto-файлы сгенерированы из официального репозитория Tinkoff/invest-python
(Apache-2.0, https://github.com/Tinkoff/invest-python).
"""
