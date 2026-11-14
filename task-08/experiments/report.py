"""Turn a dialog_cost run directory into a markdown report.

    python experiments/report.py experiments/results/stub-20261114-123456

Reads summary.json and turns_*.csv, writes report.md and cost_growth.svg into
the same directory. The chart is the same one the /lab page shows, so the web
and the CLI never disagree about what was measured.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.lab import render_chart  # noqa: E402


def fmt_int(value) -> str:
    return f"{int(value):,}".replace(",", " ")


def fmt_money(value) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".") or "0"


def load_rows(run_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(run_dir.glob("turns_*.csv")):
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            for key in ("turn", "steps", "tool_calls", "prompt_tokens", "completion_tokens",
                        "history_tokens", "request_tokens", "reply_tokens", "pred_first_call",
                        "db_messages", "http_status"):
                row[key] = int(row[key])
            for key in ("cost", "cum_cost", "latency_s"):
                row[key] = float(row[key])
        out[path.stem.removeprefix("turns_")] = rows
    return out


def table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    head = "| " + " | ".join(title for _, title, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        cells = []
        for key, _, kind in columns:
            value = row.get(key, "")
            if kind == "int":
                cells.append(fmt_int(value))
            elif kind == "money":
                cells.append(fmt_money(value))
            elif kind == "float":
                cells.append(f"{float(value):.2f}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def build_report(run_dir: Path) -> Path:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = load_rows(run_dir)
    by_name = {s["scenario"]: s for s in summary["scenarios"]}

    lines: list[str] = []
    lines.append("# Как растут токены и стоимость диалога\n")
    lines.append(f"- режим: **{summary['mode']}**")
    lines.append(f"- запуск: {summary['started_at']}")
    lines.append(f"- потрачено: **{fmt_money(summary['spent_rub'])} ₽** "
                 f"из капа {summary['cap_rub']} ₽")
    lines.append(f"- модель цены: вход {summary['price_in_per_m']:.0f} ₽/1M, "
                 f"выход {summary['price_out_per_m']:.0f} ₽/1M, "
                 f"накладные {summary['overhead_tokens']} токенов на вызов")
    lines.append(f"- заявленный лимит контекста: {fmt_int(summary['context_limit'])} токенов\n")
    if summary["mode"] == "stub":
        lines.append("> Режим `stub`: модель заменена заглушкой, которая считает токены тем же "
                     "локальным счётчиком и имитирует лимит контекста. Маршруты, хранилище и "
                     "обработка ошибок — настоящий код приложения. Пайплайн проверяется "
                     "бесплатно; реальные цифры даст режим `real`.\n")

    lines.append("## Сводка по сценариям\n")
    lines.append(table(
        [{"scenario": n, **by_name[n]} for n in by_name],
        [("scenario", "сценарий", "str"), ("turns_done", "ходов", "int"),
         ("first_prompt", "промпт 1-го хода", "int"), ("peak_prompt", "пик промпта", "int"),
         ("total_prompt", "вход всего", "int"), ("total_completion", "выход всего", "int"),
         ("total_cost", "стоимость, ₽", "money"), ("status", "статус", "str")],
    ))
    lines.append("")
    for name, info in by_name.items():
        if info.get("note"):
            lines.append(f"- **{info.get('title', name)}** — {info['note']}")
    lines.append("")

    if "short" in rows:
        lines.append("## Короткий диалог: все ходы\n")
        lines.append(table(
            rows["short"],
            [("turn", "ход", "int"), ("history_tokens", "история", "int"),
             ("request_tokens", "запрос", "int"), ("prompt_tokens", "промпт (API)", "int"),
             ("reply_tokens", "ответ", "int"), ("cost", "цена, ₽", "money"),
             ("cum_cost", "накоплено, ₽", "money"), ("latency_s", "сек", "float")],
        ))
        lines.append("")

    if "long" in rows:
        long_rows = [r for r in rows["long"] if r["status"] == "ok"]
        sampled = [r for r in long_rows if r["turn"] % 5 == 0 or r is long_rows[-1]]
        lines.append("## Длинный диалог: каждый пятый ход\n")
        lines.append(table(
            sampled,
            [("turn", "ход", "int"), ("history_tokens", "история", "int"),
             ("request_tokens", "запрос", "int"), ("prompt_tokens", "промпт (API)", "int"),
             ("cost", "цена, ₽", "money"), ("cum_cost", "накоплено, ₽", "money"),
             ("latency_s", "сек", "float")],
        ))
        lines.append("")
        if len(long_rows) >= 4:
            half = len(long_rows) // 2
            n_half, n_full = long_rows[half - 1], long_rows[-1]
            ratio = n_full["cum_cost"] / n_half["cum_cost"] if n_half["cum_cost"] else 0
            lines.append("### Проверка квадратичности\n")
            lines.append(f"Стоимость на ходе {n_half['turn']}: "
                         f"**{fmt_money(n_half['cum_cost'])} ₽**, на ходе {n_full['turn']}: "
                         f"**{fmt_money(n_full['cum_cost'])} ₽** — отношение **{ratio:.2f}**.")
            lines.append("При чисто квадратичном росте удвоение числа ходов даёт ×4; постоянная "
                         "добавка на накладные расходы слегка занижает это число. "
                         f"Промпт первого хода {fmt_int(by_name['long']['first_prompt'])} токенов, "
                         f"последнего — {fmt_int(n_full['prompt_tokens'])}.\n")

    if "overflow" in rows:
        info = by_name["overflow"]
        lines.append("## Переполнение: что ломается\n")
        lines.append(f"История засеяна напрямую в SQLite: "
                     f"{fmt_int(info['seeded_tokens'])} токенов "
                     f"({info['seeded_messages']} сообщений), "
                     f"лимит {fmt_int(summary['context_limit'])}.\n")
        lines.append(table(
            rows["overflow"],
            [("turn", "ход", "int"), ("status", "статус", "str"),
             ("http_status", "HTTP", "int"), ("history_tokens", "история", "int"),
             ("db_messages", "сообщений в БД", "int"), ("error", "ошибка", "str")],
        ))
        lines.append("")
        failure = summary.get("failure") or {}
        if failure:
            lines.append("Что из этого следует:\n")
            lines.append(f"- `/api/chat` отвечает **HTTP {failure['http_status']}** — ошибка "
                         f"отдаётся как успех, отличить отказ на уровне транспорта нельзя;")
            lines.append(f"- в базе {failure['db_before']} сообщений до и "
                         f"{failure['db_after']} после: упавший ход **не сохранён**, "
                         f"UI и БД расходятся;")
            lines.append(f"- попыток подряд: {failure['attempts']} — каждая падает одинаково, "
                         f"потому что история не обрезается: **диалог заклинило**, "
                         f"лечится только «Сбросить».\n")
        if info.get("error"):
            lines.append("Текст ошибки, который видит пользователь:\n")
            lines.append("```")
            lines.append(info["error"][:600])
            lines.append("```\n")

    lines.append("## График\n")
    lines.append("![Рост токенов и стоимости](cost_growth.svg)\n")
    lines.append("Верхняя панель — токены промпта на каждый ход: линейный рост. Нижняя — "
                 "накопленная стоимость: квадратичный. Пунктиром наложена модель `a·n²`, "
                 "построенная по последней точке: совпадение означает, что рост объясняется "
                 "именно повторной отправкой всей истории.\n")

    lines.append("## Выводы\n")
    lines.append("1. **Плата за ход пропорциональна длине истории, а плата за диалог — квадрату "
                 "числа ходов.** Каждый ход отправляет всю переписку заново, поэтому "
                 "`Σ ≈ k·n²/2` входных токенов.")
    overhead = by_name.get("baseline", {}).get("first_prompt") or 0
    source = "нулевого замера"
    if not overhead:
        # No baseline scenario: turn 1 of a short dialog is the same measurement.
        for name in ("short", "long", "long_tools"):
            first_ok = next((r for r in rows.get(name, []) if r["status"] == "ok"), None)
            if first_ok:
                overhead = (first_ok["prompt_tokens"] - first_ok["history_tokens"]
                            - first_ok["request_tokens"])
                source = f"первого хода сценария «{name}»"
                break
    if overhead:
        lines.append(f"2. **Постоянная добавка.** Даже пустой диалог стоит "
                     f"~{fmt_int(overhead)} токенов на запрос (по данным {source}): схемы "
                     f"инструментов и обвязка. Локальный счётчик её не видит, поэтому "
                     f"предохранитель по локальным токенам будет ошибаться.")
    else:
        lines.append("2. **Постоянная добавка.** Каждый запрос несёт схемы инструментов и "
                     "обвязку, которых локальный счётчик не видит, — запустите нулевой замер, "
                     "чтобы измерить её точно.")
    lines.append("3. **Переполнение заклинивает диалог.** История только дописывается, обрезки "
                 "нет: следующее сообщение снова отправит ту же переросшую историю и снова "
                 "упадёт.")
    lines.append("4. **Упавший ход не сохраняется.** `save_messages` вызывается только на "
                 "успешном пути, поэтому UI и БД расходятся, а результаты уже выполненных "
                 "инструментов теряются.")
    lines.append("5. **Ошибка отдаётся с HTTP 200** (`/api/chat` ловит исключение и кладёт текст "
                 "в поле `error`), поэтому отличить успех от отказа на уровне транспорта нельзя.")

    report = run_dir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "cost_growth.svg").write_text(
        render_chart(rows_to_flat(rows)), encoding="utf-8")
    return report


def rows_to_flat(rows: dict[str, list[dict]]) -> list[dict]:
    return [row for scenario_rows in rows.values() for row in scenario_rows]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    run_dir = Path(sys.argv[1])
    report = build_report(run_dir)
    print(f"report: {report}")
    print(f"chart:  {run_dir / 'cost_growth.svg'}")


if __name__ == "__main__":
    main()
