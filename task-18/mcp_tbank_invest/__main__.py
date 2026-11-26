"""Инструменты MCP-сервера вокруг T-Invest API (Т-Банк Инвестиции).

Каждая функция с декоратором @mcp.tool() регистрируется как MCP-инструмент:
аннотации типов становятся JSON-схемой входных параметров, docstring -
описанием для модели, возвращаемая строка - результатом вызова.
"""
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grpc
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from tinkoff.invest.grpc import (
    common_pb2,
    instruments_pb2,
    marketdata_pb2,
    operations_pb2,
    orders_pb2,
    sandbox_pb2,
    users_pb2,
)
from tinkoff.invest.grpc import (
    instruments_pb2_grpc,
    marketdata_pb2_grpc,
    operations_pb2_grpc,
    orders_pb2_grpc,
    sandbox_pb2_grpc,
    users_pb2_grpc,
)

load_dotenv(ROOT / ".env")

PROD_HOST = "invest-public-api.tinkoff.ru:443"
SANDBOX_HOST = "sandbox-invest-public-api.tinkoff.ru:443"

_CANDLE_INTERVALS = {
    "1min": "CANDLE_INTERVAL_1_MIN",
    "2min": "CANDLE_INTERVAL_2_MIN",
    "3min": "CANDLE_INTERVAL_3_MIN",
    "5min": "CANDLE_INTERVAL_5_MIN",
    "10min": "CANDLE_INTERVAL_10_MIN",
    "15min": "CANDLE_INTERVAL_15_MIN",
    "30min": "CANDLE_INTERVAL_30_MIN",
    "hour": "CANDLE_INTERVAL_HOUR",
    "2hour": "CANDLE_INTERVAL_2_HOUR",
    "4hour": "CANDLE_INTERVAL_4_HOUR",
    "day": "CANDLE_INTERVAL_DAY",
    "week": "CANDLE_INTERVAL_WEEK",
    "month": "CANDLE_INTERVAL_MONTH",
}

# Ограничения API на длину запрашиваемого периода свечей.
_MAX_DAYS_BY_INTERVAL = {
    "1min": 7, "2min": 7, "3min": 7, "5min": 31, "10min": 31,
    "15min": 31, "30min": 31,
}

_INSTRUMENT_KINDS = {
    "bond": "INSTRUMENT_TYPE_BOND",
    "share": "INSTRUMENT_TYPE_SHARE",
    "currency": "INSTRUMENT_TYPE_CURRENCY",
    "etf": "INSTRUMENT_TYPE_ETF",
    "future": "INSTRUMENT_TYPE_FUTURES",
    "sp": "INSTRUMENT_TYPE_SP",
    "option": "INSTRUMENT_TYPE_OPTION",
}

mcp = FastMCP("tbank-invest")


# ---------------------------------------------------------------- служебное

def _target() -> str:
    return os.environ.get("INVEST_TARGET", "sandbox").strip().lower()


def _host() -> str:
    return PROD_HOST if _target() == "prod" else SANDBOX_HOST


def _meta() -> list[tuple[str, str]]:
    token = os.environ.get("INVEST_TOKEN", "")
    return [
        ("authorization", f"Bearer {token}"),
        ("x-app-name", "mcp-tbank-invest"),
    ]


def _quotation_to_str(quotation) -> str:
    units = quotation.units
    nano = abs(quotation.nano)
    text = f"{units}.{nano:09d}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


def _float_to_quotation(value: float):
    units = int(value)
    nano = round((value - units) * 1e9)
    return common_pb2.Quotation(units=units, nano=nano)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# T-Invest API использует сертификаты Минцифры (Russian Trusted Root CA),
# которых нет в стандартном наборе корневых сертификатов gRPC.
_CERTS_DIR = Path(__file__).resolve().parent / "certs"


def _ssl_credentials():
    ca_path = os.environ.get("INVEST_CA_CERT", "")
    pem_files = (
        [Path(ca_path)] if ca_path else sorted(_CERTS_DIR.glob("*.pem"))
    )
    roots = b"".join(p.read_bytes() for p in pem_files if p.is_file())
    return (
        grpc.ssl_channel_credentials(root_certificates=roots)
        if roots
        else grpc.ssl_channel_credentials()
    )


@contextmanager
def _services():
    """Открывает защищённый канал и отдаёт заглушки всех сервисов API."""
    channel = grpc.secure_channel(_host(), _ssl_credentials())
    try:
        yield channel, {
            "users": users_pb2_grpc.UsersServiceStub(channel),
            "instruments": instruments_pb2_grpc.InstrumentsServiceStub(channel),
            "market": marketdata_pb2_grpc.MarketDataServiceStub(channel),
            "operations": operations_pb2_grpc.OperationsServiceStub(channel),
            "orders": orders_pb2_grpc.OrdersServiceStub(channel),
            "sandbox": sandbox_pb2_grpc.SandboxServiceStub(channel),
        }
    finally:
        channel.close()


def _error_text(exc: Exception) -> str:
    if isinstance(exc, grpc.RpcError):
        return f"Ошибка T-Invest API ({exc.code().name}): {exc.details()}"
    return f"Ошибка T-Invest API: {exc}"


def _first_account_id(stubs) -> str:
    accounts = stubs["users"].GetAccounts(
        users_pb2.GetAccountsRequest(), metadata=_meta()
    )
    if accounts.accounts:
        return accounts.accounts[0].id
    if _target() == "sandbox":
        # У свежего sandbox-токена счёта ещё нет — создаём автоматически.
        opened = stubs["sandbox"].OpenSandboxAccount(
            sandbox_pb2.OpenSandboxAccountRequest(), metadata=_meta()
        )
        return opened.account_id
    raise RuntimeError("не найдено ни одного счёта")


def _resolve_account(stubs, account_id: str) -> str:
    return account_id.strip() or _first_account_id(stubs)


def _sandbox_only() -> str | None:
    if _target() != "sandbox":
        return (
            "Ошибка: инструмент доступен только в песочнице. "
            "Установите INVEST_TARGET=sandbox."
        )
    return None


# -------------------------------------------------------------- инструменты

@mcp.tool()
def find_instrument(query: str, kind: str = "") -> str:
    """Найти инструмент по тикеру, FIGI, ISIN или названию.

    Args:
        query: поисковый запрос, например 'SBER' или 'Сбербанк'.
        kind: тип инструмента (share, bond, etf, currency, future, option).
            Пустая строка означает любой тип.
    """
    try:
        kind_name = _INSTRUMENT_KINDS.get(kind.strip().lower())
        request = instruments_pb2.FindInstrumentRequest(query=query)
        if kind_name:
            request.instrument_kind = getattr(common_pb2, kind_name)
        with _services() as (_, stubs):
            response = stubs["instruments"].FindInstrument(request, metadata=_meta())
        lines = [
            f"{i.name} ({i.ticker}, {i.class_code}): figi={i.figi}, тип={i.instrument_type}"
            for i in response.instruments[:10]
        ]
        if len(response.instruments) > 10:
            lines.append(f"... и ещё {len(response.instruments) - 10}")
        return "\n".join(lines) or "Ничего не найдено"
    except Exception as exc:
        return _error_text(exc)


@mcp.tool()
def get_last_price(figi: str) -> str:
    """Получить последнюю цену инструмента по его FIGI."""
    try:
        with _services() as (_, stubs):
            response = stubs["market"].GetLastPrices(
                marketdata_pb2.GetLastPricesRequest(figi=[figi]), metadata=_meta()
            )
        if not response.last_prices:
            return f"Цена для {figi} не найдена"
        return f"{figi}: {_quotation_to_str(response.last_prices[0].price)}"
    except Exception as exc:
        return _error_text(exc)


@mcp.tool()
def get_candles(figi: str, interval: str = "day", days: int = 5) -> str:
    """Получить историю свечей по FIGI за последние N дней.

    Args:
        figi: идентификатор инструмента.
        interval: шаг свечей (1min, 5min, 10min, 15min, 30min, hour, 2hour,
            4hour, day, week, month). По умолчанию day.
        days: сколько дней назад начинать, по умолчанию 5.
    """
    try:
        key = interval.strip().lower()
        if key not in _CANDLE_INTERVALS:
            return "Ошибка: интервал должен быть одним из " + ", ".join(_CANDLE_INTERVALS)
        candle_interval = getattr(marketdata_pb2, _CANDLE_INTERVALS[key])
        days = max(1, min(days, _MAX_DAYS_BY_INTERVAL.get(key, 366)))
        now = _utc_now()
        request = marketdata_pb2.GetCandlesRequest(
            **{
                "figi": figi,
                "from": now - timedelta(days=days),
                "to": now,
                "interval": candle_interval,
            }
        )
        with _services() as (_, stubs):
            response = stubs["market"].GetCandles(request, metadata=_meta())
        lines = [
            (
                f"{c.time.ToDatetime().strftime('%Y-%m-%d %H:%M')}: "
                f"O={_quotation_to_str(c.open)} H={_quotation_to_str(c.high)} "
                f"L={_quotation_to_str(c.low)} C={_quotation_to_str(c.close)} "
                f"V={c.volume}"
            )
            for c in response.candles[-60:]
            if c.is_complete
        ]
        return "\n".join(lines) or "Свечей не найдено"
    except Exception as exc:
        return _error_text(exc)


@mcp.tool()
def get_portfolio(account_id: str = "") -> str:
    """Получить портфель счёта: позиции, количество, средняя и текущая цены.

    Args:
        account_id: id счёта; пустая строка - взять первый доступный счёт.
    """
    try:
        with _services() as (_, stubs):
            resolved = _resolve_account(stubs, account_id)
            portfolio = stubs["operations"].GetPortfolio(
                operations_pb2.PortfolioRequest(account_id=resolved), metadata=_meta()
            )
        lines = [f"Счёт: {portfolio.account_id}"]
        for p in portfolio.positions:
            lines.append(
                f"{p.figi} ({p.instrument_type}): qty={_quotation_to_str(p.quantity)}, "
                f"avg={_quotation_to_str(p.average_position_price)}, "
                f"current={_quotation_to_str(p.current_price)}"
            )
        lines.append(
            "Всего в портфеле: " + _quotation_to_str(portfolio.total_amount_portfolio)
        )
        return "\n".join(lines)
    except Exception as exc:
        return _error_text(exc)


@mcp.tool()
def get_operations(account_id: str = "", days: int = 30) -> str:
    """Получить историю операций счёта за последние N дней.

    Args:
        account_id: id счёта; пустая строка - взять первый доступный счёт.
        days: глубина истории в днях, по умолчанию 30.
    """
    try:
        now = _utc_now()
        request = operations_pb2.OperationsRequest(
            **{
                "from": now - timedelta(days=max(1, days)),
                "to": now,
            }
        )
        with _services() as (_, stubs):
            request.account_id = _resolve_account(stubs, account_id)
            response = stubs["operations"].GetOperations(request, metadata=_meta())
        lines = [
            (
                f"{op.date.ToDatetime().strftime('%Y-%m-%d %H:%M')} "
                f"{operations_pb2.OperationType.Name(op.operation_type)} "
                f"qty={op.quantity} сумма={_quotation_to_str(op.payment)} "
                f"{op.currency} figi={op.figi}"
            )
            for op in response.operations[-50:]
        ]
        return "\n".join(lines) or "Операций не найдено"
    except Exception as exc:
        return _error_text(exc)


@mcp.tool()
def place_order(
    figi: str, direction: str = "buy", quantity: int = 1, order_type: str = "market", price: str = ""
) -> str:
    """Выставить заявку в песочнице (реальные деньги не используются).

    Args:
        figi: идентификатор инструмента.
        direction: buy или sell.
        quantity: количество лотов, по умолчанию 1.
        order_type: market или limit. Для limit обязателен price.
        price: цена лимитной заявки, например '105.5'.
    """
    sandbox_error = _sandbox_only()
    if sandbox_error:
        return sandbox_error
    try:
        direction_enum = getattr(orders_pb2, f"ORDER_DIRECTION_{direction.strip().upper()}")
        order_type_enum = getattr(orders_pb2, f"ORDER_TYPE_{order_type.strip().upper()}")
        request = orders_pb2.PostOrderRequest(
            figi=figi,
            quantity=quantity,
            direction=direction_enum,
            order_type=order_type_enum,
            order_id=uuid.uuid4().hex,
        )
        if order_type_enum == orders_pb2.ORDER_TYPE_LIMIT:
            if not price.strip():
                return "Ошибка: для лимитной заявки укажите price"
            request.price.CopyFrom(_float_to_quotation(float(price.replace(",", "."))))
        with _services() as (_, stubs):
            response = stubs["orders"].PostOrder(request, metadata=_meta())
        return (
            f"Заявка принята: order_id={response.order_id}, "
            f"статус={orders_pb2.OrderExecutionReportStatus.Name(response.execution_report_status)}, "
            f"сообщение={response.message or '-'}"
        )
    except AttributeError:
        return "Ошибка: допустимые значения direction=buy|sell, order_type=market|limit"
    except ValueError as exc:
        return f"Ошибка: некорректный параметр ({exc})"
    except Exception as exc:
        return _error_text(exc)


@mcp.tool()
def sandbox_top_up(amount: float) -> str:
    """Пополнить песочный счёт виртуальными рублями.

    Args:
        amount: сумма пополнения, например 100000.
    """
    sandbox_error = _sandbox_only()
    if sandbox_error:
        return sandbox_error
    try:
        with _services() as (_, stubs):
            account_id = _resolve_account(stubs, "")
            response = stubs["sandbox"].SandboxPayIn(
                sandbox_pb2.SandboxPayInRequest(
                    account_id=account_id,
                    amount=common_pb2.MoneyValue(currency="rub", units=int(amount), nano=0),
                ),
                metadata=_meta(),
            )
        return f"Баланс после пополнения: {_quotation_to_str(response.balance)} RUB"
    except Exception as exc:
        return _error_text(exc)


if __name__ == "__main__":
    mcp.run()
