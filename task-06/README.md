# Простой LLM агент

Агент принимает запрос пользователя, отправляет его в LLM через API и выводит ответ в веб-интерфейсе.
Запросы фронтенда выполняются через HTTP-клиент (`fetch`), бэкенд общается с LLM через OpenAI-совместимый эндпоинт.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Настройка

Скопируйте `.env.example` в `.env` и укажите свой ключ:

```
GPUSTACK_API_KEY=...
GPUSTACK_BASE_URL=https://gpustack.data.lmru.tech/v1
GPUSTACK_MODEL=deepseek-v4.1-flash
```

## Запуск

```bash
python -m app.web
```

Откройте <http://localhost:5000>
