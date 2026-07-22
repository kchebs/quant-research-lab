"""Single-asset market simulator with commissions and market impact.

The simulator converts a series of share trades (positive = buy, negative =
sell) into daily portfolio values. Costs modeled:

- ``commission``: flat dollar fee charged on every executed trade.
- ``impact``: fractional price slippage against the trader; buys execute at
  ``price * (1 + impact)`` and sells at ``price * (1 - impact)``.
"""

from __future__ import annotations

import pandas as pd


def affordable_shares(prices: pd.Series, cash: float) -> int:
    """Largest whole-share position purchasable with `cash` at the first price.

    Using this as the position limit keeps strategies and benchmarks unlevered
    and therefore comparable across windows with very different price levels.
    """
    return int(cash // prices.iloc[0])


def positions_to_trades(positions: pd.Series, max_shares: int = 1000) -> pd.Series:
    """Convert a target-position signal in {-1, 0, +1} to share trades.

    ``+1`` targets ``max_shares`` long, ``-1`` targets ``max_shares`` short,
    ``0`` targets flat. The returned series holds the share delta to execute
    each day.
    """
    target_shares = positions.clip(-1, 1) * max_shares
    trades = target_shares.diff()
    trades.iloc[0] = target_shares.iloc[0]
    return trades


def run_backtest(
    trades: pd.Series,
    prices: pd.Series,
    start_cash: float = 100_000.0,
    commission: float = 1.0,
    impact: float = 0.001,
) -> pd.DataFrame:
    """Simulate trades against a price series.

    Parameters
    ----------
    trades : share deltas indexed by date (subset of the price index).
    prices : adjusted close prices.
    start_cash : starting cash balance.
    commission, impact : transaction cost model (see module docstring).

    Returns
    -------
    DataFrame indexed like ``prices`` with columns ``shares`` (position held),
    ``cash``, and ``value`` (total portfolio value marked at close).
    """
    trades = trades.reindex(prices.index).fillna(0.0)

    shares = 0.0
    cash = start_cash
    records = []
    for date, price in prices.items():
        delta = trades.loc[date]
        if delta != 0:
            fill_price = price * (1 + impact) if delta > 0 else price * (1 - impact)
            cash -= delta * fill_price
            cash -= commission
            shares += delta
        records.append((date, shares, cash, cash + shares * price))

    return pd.DataFrame(
        records, columns=["date", "shares", "cash", "value"]
    ).set_index("date")


def buy_and_hold_benchmark(
    prices: pd.Series,
    start_cash: float = 100_000.0,
    shares: int = 1000,
    commission: float = 1.0,
    impact: float = 0.001,
) -> pd.DataFrame:
    """Buy a fixed number of shares on day one and hold to the end."""
    trades = pd.Series(0.0, index=prices.index)
    trades.iloc[0] = shares
    return run_backtest(
        trades, prices, start_cash=start_cash, commission=commission, impact=impact
    )


def run_orders(
    orders: pd.DataFrame,
    prices: pd.DataFrame,
    start_cash: float = 100_000.0,
    commission: float = 1.0,
    impact: float = 0.001,
) -> pd.Series:
    """Simulate a multi-symbol order log against a price panel.

    Parameters
    ----------
    orders : DataFrame with columns ``Symbol``, ``Order`` (BUY/SELL), ``Shares``,
        indexed by trade date (may have multiple rows per date).
    prices : DataFrame of adjusted closes, columns = symbols (plus optional extras).
    start_cash, commission, impact : same cost model as :func:`run_backtest`.

    Returns
    -------
    Series of daily portfolio values indexed like ``prices``.
    """
    orders = orders.copy()
    orders.index = pd.to_datetime(orders.index)
    orders = orders.sort_index()
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)

    symbols = sorted(set(orders["Symbol"].astype(str)))
    missing = [s for s in symbols if s not in prices.columns]
    if missing:
        raise KeyError(f"Missing price columns for symbols: {missing}")

    holdings = {s: 0.0 for s in symbols}
    cash = float(start_cash)
    values: list[tuple[pd.Timestamp, float]] = []

    # Group orders by date for same-day multi-fill handling.
    orders_by_date = {
        d: grp for d, grp in orders.groupby(level=0, sort=True)
    }

    for date, row in prices.iterrows():
        if date in orders_by_date:
            for _, order in orders_by_date[date].iterrows():
                sym = str(order["Symbol"])
                side = str(order["Order"]).upper()
                shares = float(order["Shares"])
                px = float(row[sym])
                if side == "BUY":
                    fill = px * (1 + impact)
                    cash -= shares * fill + commission
                    holdings[sym] += shares
                elif side == "SELL":
                    fill = px * (1 - impact)
                    cash += shares * fill - commission
                    holdings[sym] -= shares
                else:
                    raise ValueError(f"Unknown order side: {side}")
        equity = cash + sum(holdings[s] * float(row[s]) for s in symbols)
        values.append((date, equity))

    return pd.Series(
        [v for _, v in values],
        index=pd.DatetimeIndex([d for d, _ in values]),
        name="value",
    )
