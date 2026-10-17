"""History loading: downloads and training on separate one-coin queues."""

import asyncio
import threading
import time

from deltarisk.publisher.multi_manager import MultiCoinStreamManager, SymbolWorker


def test_next_coin_downloads_while_the_previous_one_trains(monkeypatch):
    events = []
    lock = threading.Lock()

    def log(what, worker):
        with lock:
            events.append((what, worker.symbol, time.monotonic()))

    def download(self, client=None):
        log("download start", self)
        time.sleep(0.05)
        self.status = "training"
        log("download end", self)

    def prepare(self):
        log("train start", self)
        time.sleep(0.2)
        self.status = "ready"
        log("train end", self)

    monkeypatch.setattr(SymbolWorker, "download_history", download)
    monkeypatch.setattr(SymbolWorker, "prepare_models", prepare)
    manager = MultiCoinStreamManager(persistence_path=None, live=True)
    coins = ["B-AAA_USDT", "B-BBB_USDT", "B-CCC_USDT"]
    for sym in coins:
        manager.add_symbol(sym, persist=False)

    async def load_all():
        await asyncio.gather(*(manager.load_symbol_history(s) for s in coins))

    asyncio.run(load_all())
    at = {(what, sym): t for what, sym, t in events}
    assert all(manager.workers[s].status == "ready" for s in coins)
    # B downloads while A trains, and C while B trains
    assert at[("download start", "B-BBB_USDT")] < at[("train end", "B-AAA_USDT")]
    assert at[("download start", "B-CCC_USDT")] < at[("train end", "B-BBB_USDT")]
    # Still one at a time within each queue
    for queue in ("download", "train"):
        spans = sorted((at[(f"{queue} start", s)], at[(f"{queue} end", s)]) for s in coins)
        assert all(end <= next_start for (_, end), (next_start, _) in zip(spans, spans[1:]))


def test_a_failed_download_skips_training(monkeypatch):
    def download(self, client=None):
        self.status = "failed"
        raise RuntimeError("no market")

    trained = []
    monkeypatch.setattr(SymbolWorker, "download_history", download)
    monkeypatch.setattr(SymbolWorker, "prepare_models", lambda self: trained.append(self.symbol))
    manager = MultiCoinStreamManager(persistence_path=None, live=True)
    manager.add_symbol("B-NOPE_USDT", persist=False)
    asyncio.run(manager.load_symbol_history("B-NOPE_USDT"))
    assert trained == [] and manager.workers["B-NOPE_USDT"].status == "failed"
