from datetime import datetime
from typing import List, Tuple
import numpy as np
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy
from lumibot.brokers.alpaca import Alpaca
from lumibot.traders import Trader
from config import AlpacaConfig


from main import TrendFollowingStrategy


class TrendFollowerTurkey(TrendFollowingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_market("XIST")
        self.symbols = ["AKBNK.IS", "AKGRT.IS", "MAVI.IS", "TUPRS.IS"]
        self.set_symbols(self.symbols)


if __name__ == "__main__":
    trade = False  # Set to True to run the strategy live
    TrendStrategy = TrendFollowerTurkey
    if trade:
        # implement an api for turkish brokers
        print("Not implemented yet")
    else:
        # Specify the start and ending times
        backtesting_start = datetime(2023, 8, 21)
        backtesting_end = datetime(2023, 10, 31)

        # Run the backtest
        backtest = TrendStrategy.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=100000,
            benchmark_asset="XU100.IS",
            # The benchmark asset to use for the backtest to compare to.
        )
