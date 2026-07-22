from pathlib import Path

from quantlab.abides_agent import inventory_skewed_prices, momentum_signal
from quantlab.abides_runner import (
    abides_available,
    abides_root,
    agent_description,
    agent_module_path,
    ensure_abides_on_path,
)


def test_momentum_signal_buy():
    # Rising series → fast EMA above slow
    mids = list(range(1, 40))
    assert momentum_signal(mids, fast_span=5, slow_span=20) == "buy"


def test_momentum_signal_sell():
    mids = list(range(40, 0, -1))
    assert momentum_signal(mids, fast_span=5, slow_span=20) == "sell"


def test_inventory_skew_long_prefers_sell():
    buy_q, _, sell_q, _ = inventory_skewed_prices(
        mid=100, spread_std=2, cash=10_000, shares=50, ask=101, bid=99
    )
    assert buy_q == 0
    assert sell_q > 0


def test_abides_vendored():
    assert abides_available()
    assert abides_root().joinpath("LICENSE.txt").is_file()
    assert agent_module_path().is_file()
    assert "momentum" in agent_description().lower()
    root = ensure_abides_on_path()
    assert Path(root).is_dir()
