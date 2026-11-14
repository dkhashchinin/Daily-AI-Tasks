"""Run the token lab from the command line and save the results.

The interactive version of the same engine lives in the web app:

    python -m app.web   ->   http://localhost:5000/lab

This CLI is for reproducible runs: it writes ``turns_*.csv``, ``summary.json``
and ``cost_growth.svg`` into a results directory, which
``experiments/report.py`` turns into a markdown report.

    python experiments/dialog_cost.py --mode stub
    python experiments/dialog_cost.py --mode real --cap 45 --scenarios short,long
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.lab import (  # noqa: E402
    CONTEXT_LIMIT,
    DEFAULT_SCENARIOS,
    OVERHEAD_TOKENS,
    PRICE_IN,
    PRICE_OUT,
    ROW_FIELDS,
    SCENARIOS,
    LabRun,
    render_chart,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("stub", "real"), default="stub")
    parser.add_argument("--cap", type=float, default=45.0, help="жёсткий предел расходов, ₽")
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--long-turns", type=int, default=0,
                        help="переопределить число ходов в длинном диалоге")
    parser.add_argument("--limit", type=int, default=CONTEXT_LIMIT,
                        help="лимит контекста для заглушки")
    parser.add_argument("--out", default=None, help="каталог результатов")
    parser.add_argument("--db-dir", default=None, help="каталог баз данных сценариев")
    args = parser.parse_args()

    names = [n.strip() for n in args.scenarios.split(",") if n.strip()]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error("unknown scenarios: " + ", ".join(unknown))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(args.out) if args.out
               else ROOT / "experiments" / "results" / f"{args.mode}-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = {"long": {"turns": args.long_turns}} if args.long_turns >= 2 else {}
    run = LabRun(mode=args.mode, cap=args.cap, scenarios=names, limit=args.limit,
                 overrides=overrides, db_dir=args.db_dir)
    print(f"режим {args.mode}, бюджет {args.cap} ₽, сценарии: {', '.join(names)}")
    print(f"базы сценариев: {run.db_dir}")
    run.start()

    last = None
    while run.running:
        snap = run.snapshot()
        progress = snap["progress"]
        key = (progress["scenario"], progress["turn"])
        if key != last and progress["scenario"]:
            print(f"  [{progress['scenario']}] ход {progress['turn']}/{progress['turns']} · "
                  f"потрачено {snap['spent']} ₽", flush=True)
            last = key
        time.sleep(0.2)
    run.join()
    snap = run.snapshot()

    by_scenario: dict[str, list[dict]] = {}
    for row in snap["rows"]:
        by_scenario.setdefault(row["scenario"], []).append(row)
    for name, rows in by_scenario.items():
        with (out_dir / f"turns_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    (out_dir / "cost_growth.svg").write_text(render_chart(snap["rows"]), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({
        "mode": args.mode,
        "cap_rub": args.cap,
        "started_at": snap["started_at"],
        "finished_at": snap["finished_at"],
        "state": snap["state"],
        "spent_rub": snap["spent"],
        "db_dir": str(run.db_dir),
        "price_in_per_m": PRICE_IN * 1_000_000,
        "price_out_per_m": PRICE_OUT * 1_000_000,
        "overhead_tokens": OVERHEAD_TOKENS,
        "context_limit": args.limit,
        "scenarios": snap["summaries"],
        "failure": snap["failure"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    for summary in snap["summaries"]:
        print(f"[{summary['scenario']}] {summary['status']}: ходов {summary['turns_done']}"
              f"/{summary['turns_planned']}, пик промпта {summary['peak_prompt']}, "
              f"стоимость {summary['total_cost']} ₽")
        if summary["error"]:
            print(f"    ошибка: {summary['error'][:160]}")
    print(f"\nитого {snap['spent']} ₽ из {args.cap} ₽ · состояние: {snap['state']}")
    print(f"результаты: {out_dir}")


if __name__ == "__main__":
    main()
