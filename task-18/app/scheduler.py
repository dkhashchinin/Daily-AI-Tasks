"""Планировщик: фоновый процесс, который по расписанию дёргает
MCP-инструменты сервера rates-store и собирает сводку за период.

Запуск:  python scheduler.py            (бесконечный цикл по расписанию)
         python scheduler.py --once     (один сбор — для внешнего cron/Task Scheduler)
         python scheduler.py --report   (отчёт прямо сейчас из накопленного)

Расписание (env или CLI): SCHED_INTERVAL_SEC / SCHED_REPORT_INTERVAL_SEC /
SCHED_REPORT_HOURS. Каждый запуск пишется в SQLite через MCP-инструмент
save_snapshot, поэтому данные накапливаются и между перезапусками процесса.
Состояние для веб-панели — data/scheduler_status.json, журнал — data/scheduler.log,
отчёты — data/reports/report_*.md|.json.
"""
import argparse
import atexit
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from .mcp_bridge import McpBridge, _server_command
from .reports import aggregate, format_report_md, parse_read_snapshots

ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_INTERVAL = 30.0
_DEFAULT_REPORT_INTERVAL = 120.0
_DEFAULT_REPORT_HOURS = 24.0
_DEFAULT_CURRENCIES = "USD, EUR, CNY, GBP"


class SchedulerConfig:
    """Параметры расписания и сбора; env -> CLI/веб-переопределения."""

    def __init__(self, args: argparse.Namespace | None = None) -> None:
        def num(value, default):
            try:
                v = float(value)
                return v if v > 0 else default
            except (TypeError, ValueError):
                return default

        def num_or_zero(value, default):
            """Как num, но 0 допустим (для глубины отчёта 0 = вся история)."""
            if value is not None and str(value).strip() != "":
                try:
                    v = float(value)
                    return v if v >= 0 else default
                except (TypeError, ValueError):
                    return default
            return default

        self.interval = num(
            args.interval if args and args.interval else os.getenv("SCHED_INTERVAL_SEC"),
            _DEFAULT_INTERVAL,
        )
        self.report_interval = num(
            args.report_interval if args and args.report_interval else os.getenv("SCHED_REPORT_INTERVAL_SEC"),
            _DEFAULT_REPORT_INTERVAL,
        )
        self.report_hours = num_or_zero(
            args.hours if args and args.hours is not None else os.getenv("SCHED_REPORT_HOURS"),
            _DEFAULT_REPORT_HOURS,
        )
        currencies = (
            args.currencies if args and args.currencies
            else os.getenv("SCHED_CURRENCIES")
        )
        # Нормализация в одном месте: "usd, eur ,JPY" -> "USD, EUR, JPY".
        from mcp_rates_store.rates import parse_currencies

        self.currencies = ", ".join(parse_currencies(currencies))
        self.data_dir = Path(
            os.getenv("SCHED_DATA_DIR") or (ROOT / "data")
        )
        self.reports_dir = self.data_dir / "reports"
        self.status_file = self.data_dir / "scheduler_status.json"
        self.log_file = self.data_dir / "scheduler.log"
        self.max_loops = int(args.max_loops) if args and args.max_loops else 0


class Scheduler:
    """Держит MCP-сессию, исполняет задания по расписанию, пишет статус/журнал.

    mode: "cli" — запущен из терминала/cron, "web" — из кнопки в веб-панели.
    """

    def __init__(self, config: SchedulerConfig, mode: str = "cli"):
        self.cfg = config
        self.mode = mode
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.bridge: McpBridge | None = None
        self.collects_total = 0
        self.errors_total = 0
        self.last_collect_at: str | None = None
        self.last_collect_summary: str | None = None
        self.last_report_at: str | None = None
        self.last_report_file: str | None = None
        self.next_collect_at: str | None = None
        self.next_report_at: str | None = None

    # ------------------------------------------------------------- инфраструктура

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        try:
            self.cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cfg.log_file.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def write_status(self, running: bool) -> None:
        status = {
            "running": running,
            "mode": self.mode,
            "pid": os.getpid(),
            "started_at": getattr(self, "started_at", None),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "collect_interval_sec": self.cfg.interval,
            "report_interval_sec": self.cfg.report_interval,
            "report_hours": self.cfg.report_hours,
            "currencies": self.cfg.currencies,
            "collects_total": self.collects_total,
            "errors_total": self.errors_total,
            "last_collect_at": self.last_collect_at,
            "last_collect_summary": self.last_collect_summary,
            "last_report_at": self.last_report_at,
            "last_report_file": self.last_report_file,
            "next_collect_at": self.next_collect_at,
            "next_report_at": self.next_report_at,
        }
        try:
            self.cfg.status_file.parent.mkdir(parents=True, exist_ok=True)
            self.cfg.status_file.write_text(
                json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            self.log(f"не удалось записать статус: {exc}")

    @staticmethod
    def _iso(monotonic_deadline: float | None = None) -> str:
        if monotonic_deadline is None:
            return datetime.now().astimezone().isoformat(timespec="seconds")
        delta = max(0.0, monotonic_deadline - time.monotonic())
        return (datetime.now().astimezone() + timedelta(seconds=delta)).isoformat(
            timespec="seconds"
        )

    def _call(self, tool: str, args: dict | None = None) -> str:
        return self.bridge.call_tool(tool, args or {})

    def request_stop(self) -> None:
        """Попросить цикл завершиться (кнопка в вебе / остановка приложения)."""
        self.stop_event.set()

    def _sleep_until(self, deadline: float) -> None:
        """Спать до дедлайна, но просыпаться раньше, если пришла команда стоп."""
        while not self.stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.stop_event.wait(min(0.25, remaining))

    # ------------------------------------------------------------------- задания

    def job_collect(self) -> str:
        """Задание сбора: fetch_rates -> save_snapshot. Возвращает строку-итог."""
        rates_text = self._call("fetch_rates", {"currencies": self.cfg.currencies})
        try:
            rates = json.loads(rates_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"fetch_rates вернул не-JSON: {exc}") from exc
        if rates.get("error"):
            raise RuntimeError(f"fetch_rates: {rates['error']}")

        saved_text = self._call("save_snapshot", {"payload": rates_text})
        saved = json.loads(saved_text)
        if saved.get("error"):
            raise RuntimeError(f"save_snapshot: {saved['error']}")

        rates_str = ", ".join(f"{k} {v}" for k, v in (saved.get("rates") or {}).items())
        unknown = rates.get("unknown") or []
        note = f"; пропущены unknown-коды: {', '.join(unknown)}" if unknown else ""
        return f"снимок #{saved['id']} сохранён ({rates_str or 'без курсов'}){note}"

    def job_report(self) -> str:
        """Задание отчёта: read_snapshots -> агрегат -> md/json в data/reports."""
        hours = self.cfg.report_hours
        rows, _, error = parse_read_snapshots(self._call("read_snapshots", {"hours": hours}))
        if error:
            raise RuntimeError(error)

        report = aggregate(rows, hours)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = self.cfg.reports_dir / f"report_{stamp}.md"
        json_path = self.cfg.reports_dir / f"report_{stamp}.json"
        md_path.write_text(format_report_md(report), encoding="utf-8")
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.last_report_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.last_report_file = md_path.name

        series = report.get("series") or {}
        parts = [f"{code}: {s['first']:.2f} -> {s['last']:.2f} ({s['change_pct']:+.2f}%)"
                 for code, s in series.items() if s.get("points")]
        period = "всю историю" if not hours else f"{hours:g} ч"
        return (
            f"отчёт за {period} по {report['snapshots_count']} снимкам "
            f"-> {md_path.name}"
            + ("; " + "; ".join(parts) if parts else "")
        )

    # --------------------------------------------------------------------- циклы

    def run_forever(self) -> None:
        """Основной режим: бесконечный цикл по расписанию."""
        self.cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.log(
            f"планировщик запущен: сбор каждые {self.cfg.interval:g} с "
            f"(валюты: {self.cfg.currencies}), отчёт каждые {self.cfg.report_interval:g} с "
            f"(период {self.cfg.report_hours:g} ч, БД и отчёты в {self.cfg.data_dir})"
        )
        self.bridge = McpBridge(_server_command("mcp_rates_store"))
        self.log(f"MCP rates-store подключён, инструменты: {', '.join(self.bridge.tool_names)}")
        self.write_status(running=True)

        next_collect = time.monotonic()          # первый сбор выполняем сразу
        next_report = time.monotonic() + self.cfg.report_interval
        try:
            while not self.stop_event.is_set() and (
                self.cfg.max_loops <= 0 or self.collects_total < self.cfg.max_loops
            ):
                now = time.monotonic()
                if now >= next_collect:
                    self._run_job("collect")
                    self.collects_total += 1
                    while next_collect <= now:
                        next_collect += self.cfg.interval
                now = time.monotonic()
                if now >= next_report:
                    self._run_job("report")
                    while next_report <= now:
                        next_report += self.cfg.report_interval

                self.next_collect_at = self._iso(next_collect)
                self.next_report_at = self._iso(next_report)
                self.write_status(running=True)
                wake = min(next_collect, next_report)
                self._sleep_until(wake)
            # Выход из цикла: по команде стоп либо по лимиту --max-loops.
            if self.stop_event.is_set():
                self.log(f"планировщик остановлен по запросу ({self.mode})")
            elif time.monotonic() >= next_report:
                # Отчёт уже был должен по расписанию, но не успел — собираем.
                self._run_job("report")
        except KeyboardInterrupt:
            self.log("остановлен вручную (Ctrl+C)")
        finally:
            self.write_status(running=False)
            if self.bridge is not None:
                self.bridge.close()
            self.log(f"планировщик остановлен [{self.mode}]")

    def _run_job(self, kind: str) -> bool:
        """Выполнить задание; True — успешно, False — с ошибкой."""
        try:
            summary = self.job_collect() if kind == "collect" else self.job_report()
            self.log(f"ЗАДАНИЕ {kind}: {summary}")
            if kind == "collect":
                self.last_collect_at = datetime.now().astimezone().isoformat(timespec="seconds")
                self.last_collect_summary = summary
            return True
        except Exception as exc:
            self.errors_total += 1
            self.log(f"ЗАДАНИЕ {kind}: ошибка: {exc}")
            self.log(traceback.format_exc(limit=3))
            return False

    def run_once(self) -> int:
        """Режим для внешнего планировщика (cron / Task Scheduler): один сбор."""
        self.cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = McpBridge(_server_command("mcp_rates_store"))
        try:
            ok = self._run_job("collect")
            self.collects_total += 1
            return 0 if ok else 1
        finally:
            if self.bridge is not None:
                self.bridge.close()

    def run_report_now(self) -> int:
        """Собрать отчёт немедленно из уже накопленных данных."""
        self.cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = McpBridge(_server_command("mcp_rates_store"))
        try:
            return 0 if self._run_job("report") else 1
        finally:
            if self.bridge is not None:
                self.bridge.close()


# --------------------------------------------------- управление из веб-панели

_background: Scheduler | None = None
_background_lock = threading.Lock()
_atexit_registered = False


def _ensure_atexit() -> None:
    """При выходе из процесса веб-приложения корректно остановить планировщик."""
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(stop_background)
        _atexit_registered = True


def start_background(interval: float | None = None,
                     report_interval: float | None = None,
                     hours: float | None = None,
                     currencies: str | None = None) -> tuple[Scheduler, bool]:
    """Запустить планировщик в фоновом потоке веб-приложения.

    Использует тот же класс Scheduler, что и CLI: те же MCP-инструменты,
    статус, журнал и БД. Возвращает (экземпляр, удалось_ли_запустить);
    повторный запуск, пока предыдущий работает, считается ошибкой.
    """
    global _background
    with _background_lock:
        current = _background
        if current is not None and current.thread is not None and current.thread.is_alive():
            return current, False

        args = argparse.Namespace(
            interval=interval, report_interval=report_interval,
            hours=hours, currencies=currencies, max_loops=0,
        )
        scheduler = Scheduler(SchedulerConfig(args), mode="web")
        # Пишем статус синхронно, чтобы веб-панель сразу увидела новые
        # параметры (валюты, интервалы), не дожидаясь первого цикла потока.
        scheduler.write_status(running=True)
        scheduler.thread = threading.Thread(
            target=scheduler.run_forever, name="scheduler-web", daemon=True
        )
        scheduler.thread.start()
        _background = scheduler
        _ensure_atexit()
        return scheduler, True


def stop_background(timeout: float = 15.0) -> tuple[bool, bool]:
    """Остановить фоновый планировщик; (был_ли_запущен, остановился_ли).

    Текущее задание (MCP-вызов) даём мягко завершить, дальше цикл выходит
    сам: статус-файл получает running=False, MCP-сессия закрывается.
    """
    global _background
    with _background_lock:
        scheduler = _background
    if scheduler is None or scheduler.thread is None or not scheduler.thread.is_alive():
        return False, True
    scheduler.request_stop()
    scheduler.thread.join(timeout=timeout)
    return True, not scheduler.thread.is_alive()


def get_background() -> Scheduler | None:
    with _background_lock:
        return _background


def _read_status_file(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def scheduler_snapshot(max_snapshots: int = 15) -> dict:
    """Состояние планировщика для веб-панели: статус + снимки + отчёты.

    Читает файлы, которые пишет запущенный процесс-планировщик: статус
    (data/scheduler_status.json), SQLite (только чтение) и каталог отчётов.
    """
    from mcp_rates_store import store

    cfg = SchedulerConfig()
    status = _read_status_file(cfg.status_file)
    running = False
    if status:
        grace = float(status.get("collect_interval_sec") or cfg.interval) * 3 + 15
        running = bool(status.get("running")) and _is_fresh(
            status.get("updated_at"), grace
        )
    # Планировщик, запущенный кнопкой из этого же процесса, живее любого файла.
    background = get_background()
    bg_alive = (
        background is not None
        and background.thread is not None
        and background.thread.is_alive()
    )
    running = running or bg_alive
    if bg_alive and status is not None:
        status = dict(status, mode=background.mode)

    reports: list[str] = []
    if cfg.reports_dir.exists():
        reports = sorted(
            (p.name for p in cfg.reports_dir.glob("report_*.md")), reverse=True
        )
    latest_report = None
    if reports:
        try:
            latest_report = (cfg.reports_dir / reports[0]).read_text(encoding="utf-8")
        except OSError:
            latest_report = None

    return {
        "status": status,
        "running": running,
        "snapshots": store.read_recent(max_snapshots),
        "snapshots_total": store.count_all(),
        "reports": reports,
        "latest_report": latest_report,
    }


def _is_fresh(iso_ts: str | None, grace_sec: float) -> bool:
    if not iso_ts:
        return False
    try:
        updated = datetime.fromisoformat(iso_ts)
    except ValueError:
        return False
    now = datetime.now(updated.tzinfo) if updated.tzinfo else datetime.now()
    return (now - updated).total_seconds() <= grace_sec


def main(argv: list[str] | None = None) -> None:
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Планировщик MCP-заданий")
    parser.add_argument("--interval", type=float, help="период сбора, сек (SCHED_INTERVAL_SEC)")
    parser.add_argument("--report-interval", type=float, help="период отчёта, сек")
    parser.add_argument("--hours", type=float, help="глубина отчёта, часов (0 = вся история)")
    parser.add_argument("--currencies", help="валюты ЦБ через запятую, до 8 (SCHED_CURRENCIES)")
    parser.add_argument("--max-loops", type=int, help="остановиться после N сборов (0 = бесконечно)")
    parser.add_argument("--once", action="store_true", help="выполнить один сбор и выйти")
    parser.add_argument("--report", action="store_true", help="собрать отчёт сейчас и выйти")
    args = parser.parse_args(argv)

    cfg = SchedulerConfig(args)
    scheduler = Scheduler(cfg)

    if args.report:
        raise SystemExit(scheduler.run_report_now())
    if args.once:
        raise SystemExit(scheduler.run_once())
    scheduler.run_forever()


if __name__ == "__main__":
    main()
