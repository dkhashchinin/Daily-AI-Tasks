# LLM Chat Demo

Минимальный проект для общения с LLM через API (OpenAI-совместимый эндпоинт).
Интерфейс: **Web**.

## Структура

```
task-01/
├── app/               # основное приложение (пакет)
│   ├── config.py      # настройки из переменных окружения
│   ├── llm.py         # обёртка над OpenAI-клиентом (функция chat)
│   └── web.py         # Flask-интерфейс
├── .env.example       # шаблон настроек
├── .env               # реальные настройки (не коммитится)
├── .gitignore
└── requirements.txt
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Настройка

Скопируйте `.env.example` в `.env` и укажите свой ключ:

```
GPUSTACK_API_KEY=...
GPUSTACK_BASE_URL=https://gpustack.data.lmru.tech/v1
GPUSTACK_MODEL=deepseek-v4.1-flash
```

## Запуск Web

```bash
python -m app.web
```

Откройте <http://localhost:5000>
