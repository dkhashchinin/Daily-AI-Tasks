"""Агрегация накопленных снимков в отчёт за период.

Чистые функции без зависимостей: планировщик скармливает им JSON из
MCP-инструмента read_snapshots, получает dict отчёта и markdown-текст.
Отчёт всегда считается по ВСЕМ снимкам периода, а не по последнему.
Набор валют берётся из самих снимков — что собирали, то и агрегируется.
"""
import json
from datetime import datetime

# Приоритет порядка колонок: основные валюты вперёд, остальное по алфавиту.
_PRIORITY = ("usd", "eur", "cny", "gbp")

# Сколько последних снимков показывать таблицей в markdown-отчёте.
_TABLE_ROWS_MAX = 30


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def currencies_from_rows(rows: list[dict]) -> list[str]:
    """Собрать список валют, встречающихся в снимках, в стабильном порядке."""
    codes: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key in ("id", "ts") or not isinstance(value, (int, float)):
                continue
            if key not in codes:
                codes.append(key)
    codes.sort(key=lambda c: (_PRIORITY.index(c) if c in _PRIORITY else len(_PRIORITY), c))
    return codes


def _series_stats(rows: list[dict], key: str) -> dict | None:
    """min/max/avg/first/last и изменение за период по одной валюте."""
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return None
    first, last = values[0], values[-1]
    change = last - first
    return {
        "first": first,
        "last": last,
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 4),
        "change": round(change, 4),
        "change_pct": round(change / first * 100, 2) if first else 0.0,
        "points": len(values),
    }


def aggregate(rows: list[dict], period_hours: float, generated_at: str | None = None) -> dict:
    """Собрать отчёт из списка снимков (как их вернул read_snapshots)."""
    rows = sorted(rows, key=lambda r: (r.get("ts") or "", r.get("id") or 0))
    now = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    codes = currencies_from_rows(rows)
    report = {
        "generated_at": now,
        "period_hours": period_hours,
        "currencies": codes,
        "snapshots_count": len(rows),
        "first_ts": rows[0]["ts"] if rows else None,
        "last_ts": rows[-1]["ts"] if rows else None,
        "series": {},
        "snapshots": rows,
    }
    for code in codes:
        stats = _series_stats(rows, code)
        if stats is not None:
            report["series"][code.upper()] = stats
    return report


def format_report_md(report: dict) -> str:
    """Markdown-представление отчёта (файл в data/reports/)."""
    count = report["snapshots_count"]
    period = ("всю накопленную историю"
              if not report["period_hours"]
              else f"{report['period_hours']:g} ч")
    # Ключи в снимках — в нижнем регистре (usd, jpy...), в заголовке —
    # в верхнем. Храним оба варианта: иначе row.get("USD") всегда мимо.
    raw_codes = list(report.get("currencies") or report["series"].keys())
    codes = [code.upper() for code in raw_codes]
    lines = [
        f"# Сводка по курсам ЦБ за {period}",
        "",
        f"- Сформирован: {report['generated_at']}",
        f"- Снимков в отчёте: **{count}**",
        f"- Валюты: {', '.join(codes) if codes else '-'}",
        f"- Период: {report['first_ts'] or '-'} … {report['last_ts'] or '-'}",
    ]
    if not count:
        lines += ["", "Данных за период нет — планировщик ещё ничего не собрал."]
        return "\n".join(lines) + "\n"

    lines += ["", "## Агрегаты по валютам", "",
              "| Валюта | Первая | Последняя | Изменение | Мин | Макс | Средняя | Точек |",
              "|---|---|---|---|---|---|---|---|"]
    for code in codes:
        s = report["series"][code]
        lines.append(
            f"| {code} | {s['first']:.4f} | {s['last']:.4f} "
            f"| {s['change']:+.4f} ({s['change_pct']:+.2f}%) "
            f"| {s['min']:.4f} | {s['max']:.4f} | {s['avg']:.4f} | {s['points']} |"
        )

    shown = report["snapshots"][-_TABLE_ROWS_MAX:]
    head = " | ".join(codes) if codes else "-"
    lines += ["", f"## Последние запуски ({len(shown)} из {count})", "",
              f"| # | Время | {head} |",
              f"|---|---|{'---|' * len(codes)}"]
    for row in shown:
        cells = " | ".join(
            str(row.get(code, "-")) if row.get(code) is not None else "-"
            for code in raw_codes
        )
        lines.append(f"| {row['id']} | {row['ts']} | {cells} |")
    lines.append("")
    lines.append(
        '"-" — валюта не собиралась в этом запуске: набор валют задаётся при старте '
        "планировщика и может меняться. Агрегат по валюте считается только по точкам, "
        "где она была — см. колонку «Точек»."
    )
    if count > len(shown):
        lines.append("")
        lines.append(f"…и ещё {count - len(shown)} более ранних снимков (см. JSON-версию отчёта).")
    return "\n".join(lines) + "\n"


def parse_read_snapshots(text: str) -> tuple[list[dict], float | None, str | None]:
    """Разобрать ответ MCP-инструмента read_snapshots -> (rows, hours, error)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], None, f"read_snapshots вернул не-JSON: {exc}"
    if isinstance(data, dict) and data.get("error"):
        return [], None, str(data["error"])
    if not isinstance(data, dict) or "snapshots" not in data:
        return [], None, f"неожиданный формат ответа read_snapshots: {text[:200]}"
    return data.get("snapshots") or [], data.get("period_hours"), None
