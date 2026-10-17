"""Backtest of the setup rules: one account, one setup at a time, fixed risk per trade."""

import numpy as np
import pandas as pd
import pytest

from futurecast import cli
from futurecast.backtest import BacktestConfig, format_report, run_backtest
from futurecast.features.setups import SetupSpec

# Round trip on a 1% stop: a win 0.04R (maker both ways), a stop 0.10R (maker, then taker + slippage)
SPEC = SetupSpec(ltf_minutes=1, htf_minutes=15, targets=(3.0,), maker_bps=2.0, taker_bps=5.0, slippage_bps=3.0)


def _history():
    """Eight A+/B setups with known results, on 100 one-minute candles."""
    index = pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC")
    #            idx grade  fill         fill_idx order_end outcome    r     end
    rows = [(0, "A+", "filled", 2, -1, "won", 3.0, 10),        # taken
            (5, "A+", "filled", 6, -1, "won", 3.0, 9),         # skipped: the first trade is still open
            (12, "A+", "expired", -1, 20, "", np.nan, -1),     # an order that never fills: blocks until 20
            (15, "A+", "filled", 16, -1, "won", 3.0, 18),      # skipped: that order is still waiting
            (25, "B", "filled", 26, -1, "won", 3.0, 28),       # skipped: wrong grade
            (30, "A+", "filled", 31, -1, "lost", -1.0, 35),    # taken
            (40, "A+", "filled", 41, -1, "lost", -1.0, 45),    # taken
            (50, "A+", "pending", -1, -1, "", np.nan, -1)]     # left out: not resolved yet
    setups = pd.DataFrame({"idx": [r[0] for r in rows], "side": 1, "grade": [r[1] for r in rows],
                           "entry": 100.0, "stop": 99.0, "risk": 0.01}, index=index[[r[0] for r in rows]])
    outcomes = pd.DataFrame({"fill": [r[2] for r in rows], "fill_idx": [r[3] for r in rows],
                             "order_end": [r[4] for r in rows], "outcome_3": [r[5] for r in rows],
                             "win_3": [1.0 if r[5] == "won" else (0.0 if r[5] else np.nan) for r in rows],
                             "r_3": [r[6] for r in rows], "end_3": [r[7] for r in rows]}, index=setups.index)
    return setups, outcomes, index


def test_one_setup_at_a_time_with_fixed_risk():
    setups, outcomes, index = _history()
    result = run_backtest(setups, outcomes, index, SPEC, BacktestConfig(target=3.0, risk_pct=1.0))
    s, trades = result["summary"], result["trades"]

    assert s["orders"] == 4 and s["trades"] == 3 and s["fill_rate"] == 0.75
    assert list(trades["setup_at"].str[11:16]) == ["00:01", "00:31", "00:41"]  # setup candles 0, 30, 40 (close times)
    assert list(trades["net_r"].round(4)) == [2.96, -1.1, -1.1]
    assert (s["wins"], s["losses"], s["win_rate"]) == (1, 2, round(1 / 3, 4))
    assert s["total_net_r"] == pytest.approx(0.76)
    assert s["profit_factor"] == pytest.approx(2.96 / 2.2, abs=1e-3)
    assert s["break_even_win_rate"] == pytest.approx(1.1 / (3 - 0.04 + 1.1), abs=1e-4)  # p (3 - 0.04) = (1 - p) 1.1
    # 1% risk per trade, compounding: +2.96%, then -1.1% twice
    assert s["total_return_pct"] == pytest.approx((1.0296 * 0.989 * 0.989 - 1) * 100, abs=0.01)
    assert s["max_drawdown_pct"] == pytest.approx((1 - 0.989 ** 2) * 100, abs=0.01)
    assert s["max_drawdown_r"] == pytest.approx(2.2) and s["longest_losing_streak"] == 2
    assert trades["hold_hours"].iat[0] == pytest.approx(9 / 60)  # fill candle 2 through candle 10

    assert trades["equity_after"].to_numpy() == pytest.approx([102.96, 102.96 * 0.989, 102.96 * 0.989 ** 2])
    assert result["sides"]["long"]["trades"] == 3
    assert result["monthly"] == [{"month": "2026-01", "trades": 3, "win_rate": round(1 / 3, 4), "net_r": 0.76,
                                  "return_pct": round((1.0296 * 0.989 * 0.989 - 1) * 100, 2)}]

    # Taking every grade adds the B setup, which is free by then
    every = run_backtest(setups, outcomes, index, SPEC, BacktestConfig(grades=("A+", "B")))["summary"]
    assert every["trades"] == 4

    report = format_report("B-TEST_USDT", result)
    assert "Win rate          33.3%" in report and "Only 3 trades" in report
    assert "fees 0.02% maker / 0.05% taker, 0.03% stop slippage (0.04% round trip on a win, 0.1% on a stop)" in report


def test_each_trade_pays_the_costs_of_how_it_ended():
    setups, outcomes, index = _history()
    # The last taken trade times out at +0.5R instead: closed with a market order, so taker but no slippage
    outcomes.loc[outcomes.index[6], ["outcome_3", "r_3", "win_3"]] = ["timeout", 0.5, 0.0]
    result = run_backtest(setups, outcomes, index, SPEC, BacktestConfig(target=3.0))
    trades, s = result["trades"], result["summary"]
    assert list(trades["outcome"]) == ["won", "lost", "timeout"]
    assert list(trades["cost_r"].round(4)) == [0.04, 0.1, 0.07]
    assert list(trades["net_r"].round(4)) == [2.96, -1.1, 0.43]
    assert s["avg_cost_r"] == pytest.approx(0.07, abs=1e-4)
    # Break-even still counts every non-win as a stop
    assert s["break_even_win_rate"] == pytest.approx(1.1 / 4.06, abs=1e-4)


def test_leverage_cap_shrinks_the_risk_of_tight_stops():
    setups, outcomes, index = _history()
    setups.loc[setups.index[5], "risk"] = 0.001  # the second taken trade has a 0.1% stop: 1% risk needs 10x
    free = SetupSpec(ltf_minutes=1, htf_minutes=15, targets=(3.0,), maker_bps=0.0, taker_bps=0.0, slippage_bps=0.0)
    result = run_backtest(setups, outcomes, index, free, BacktestConfig(target=3.0, max_leverage=5.0))
    trades, s = result["trades"], result["summary"]
    assert list(trades["leverage"]) == [1.0, 5.0, 1.0]
    assert list(trades["account_risk_pct"]) == pytest.approx([1.0, 0.5, 1.0])
    assert list(trades["net_r"]) == [3.0, -1.0, -1.0]  # results in R are unchanged ...
    assert s["capped_trades"] == 1  # ... but the capped loss costs the account half as much
    assert s["total_return_pct"] == pytest.approx((1.03 * 0.995 * 0.99 - 1) * 100, abs=0.01)
    assert "1 of 3 trades risked less than 1%: their stop was too tight for 5x" in format_report("B-TEST_USDT", result)

    uncapped = run_backtest(setups, outcomes, index, free, BacktestConfig(target=3.0))["summary"]
    assert uncapped["capped_trades"] == 0
    assert uncapped["total_return_pct"] == pytest.approx((1.03 * 0.99 * 0.99 - 1) * 100, abs=0.01)


def test_no_trades_is_reported_not_invented():
    setups, outcomes, index = _history()
    result = run_backtest(setups.iloc[:0], outcomes.iloc[:0], index, SPEC)
    assert result["summary"]["trades"] == 0 and result["monthly"] == [] and "win_rate" not in result["summary"]
    assert "No filled trades." in format_report("B-TEST_USDT", result)


class _FakeClient:
    """Random-walk candles for any interval, as the Binance client returns them."""

    def fetch_candle_history(self, symbol, interval, bars, **_):
        minutes = SetupSpec.for_charts(interval, interval).ltf_minutes
        rng = np.random.default_rng(minutes)
        ts = pd.date_range(end="2026-06-01", periods=bars, freq=f"{minutes}min", tz="UTC")
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002 * np.sqrt(minutes), bars)))
        open_ = np.r_[close[0], close[:-1]]
        wick = np.abs(rng.normal(0, 0.001 * np.sqrt(minutes), bars))
        return pd.DataFrame({"timestamp": ts, "open": open_, "high": np.maximum(open_, close) * (1 + wick),
                             "low": np.minimum(open_, close) * (1 - wick), "close": close,
                             "volume": rng.lognormal(0, 0.5, bars)}), "fake"

    def close(self):
        pass


def test_backtest_command(monkeypatch, capsys, tmp_path):
    from futurecast.publisher import multi_manager

    monkeypatch.setattr(multi_manager, "_binance_client", lambda assets: _FakeClient())
    csv = tmp_path / "trades.csv"
    cli.backtest("b-test_usdt", strategy="fvg", entry="15m", context="1h", days=120, target=3.0, grade="all", risk=1.0,
                 max_leverage=None, breakeven=True, maker=1.0, taker=4.0, slippage=None, csv=str(csv))
    out = capsys.readouterr().out
    assert out.startswith("B-TEST_USDT · 15m entries in 1h context · A+/B setups · 3R target")
    # Fees not given keep SetupSpec's defaults
    assert f"fees 0.01% maker / 0.04% taker, {SetupSpec().slippage_bps / 100:g}% stop slippage" in out
    assert "setups found" in out
    if "No filled trades." not in out:
        assert "Win rate" in out and pd.read_csv(csv)["outcome"].isin(["won", "lost", "timeout"]).all()


def test_wedge_backtest_command(monkeypatch, capsys, tmp_path):
    from futurecast.publisher import multi_manager

    monkeypatch.setattr(multi_manager, "_binance_client", lambda assets: _FakeClient())
    csv = tmp_path / "trades.csv"
    cli.backtest("b-test_usdt", strategy="wedge", entry=None, context="1h", days=400, target=None, grade="all",
                 risk=1.0, max_leverage=None, breakeven=True, maker=None, taker=None, slippage=None, csv=str(csv))
    out = capsys.readouterr().out
    # One chart (1h by default), a 2R target, leverage capped at 5x and the stop moved at 1.5R unless told otherwise
    assert out.startswith("B-TEST_USDT · wedge breakouts on 1h · A+/B setups · 2R target · 1% risk per trade, "
                          "at most 5x leverage\nmarket entries, stop to break-even at 1.5R")
    assert " wedges, " in out and " broke out: " in out
    if "No filled trades." not in out:
        assert pd.read_csv(csv)["outcome"].isin(["won", "lost", "breakeven", "timeout"]).all()

    with pytest.raises(Exception, match="needs --strategy wedge"):
        cli.backtest("b-test_usdt", strategy="fvg", entry=None, context="1h", days=30, target="measured", grade="A+",
                     risk=1.0, max_leverage=None, breakeven=True, maker=None, taker=None, slippage=None, csv="")
