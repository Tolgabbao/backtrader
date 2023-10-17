from datetime import datetime
from typing import List, Tuple
import numpy as np
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy
from lumibot.brokers.alpaca import Alpaca
from lumibot.traders import Trader
from config import AlpacaConfig


class TrendFollowingStrategy(Strategy):
    def __init__(self, *args: Tuple, **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)
        self.symbols: List[str] = [
            'YKBNK.IS', 'AKBNK.IS', 'THYAO.IS', 'TUPRS.IS', 'TCELL.IS', 'GARAN.IS', 'VAKBN.IS', 'HALKB.IS', 'EREGL.IS'
            , 'KCHOL.IS', 'SAHOL.IS', ]
        # Replace with your desired symbols
        self.sma1: int = 50
        self.sma2: int = 250
        self.risk_per_trade: float = 0.03
        self.atr_period: int = 14
        self.atr_multiplier: int = 2
        self.stop_loss: float = 0

    def on_trading_iteration(self) -> None:
        """
        This method is called on each trading iteration.
        It checks if a buy or sell order should be placed for each symbol.
        """
        if self.broker.is_market_open():
            for symbol in self.symbols:
                if not self.has_position(symbol):
                    if self.should_place_buy_order(symbol):
                        self.place_buy_order(symbol)
                elif self.has_position(symbol):
                    if self.stop_loss is None:
                        self.calculate_stop_loss(symbol)
                    if self.should_place_sell_order(symbol):
                        self.place_sell_order(symbol)

    def has_position(self, symbol: str) -> bool:
        """
        This method checks if a position exists for a symbol.
        """
        position = self.get_position(asset=symbol)
        return position is not None and position.quantity > 0

    def should_place_buy_order(self, symbol: str) -> bool:
        """
        This method checks if a buy order should be placed for a symbol.
        """
        return self.crossover(symbol) > 0

    def should_place_sell_order(self, symbol: str) -> bool:
        """
        This method checks if a sell order should be placed for a symbol.
        """
        position = self.get_position(asset=symbol)
        if position is None:
            return False
        return (
                self.get_last_price(asset=symbol) <= self.stop_loss or
                self.crossover(symbol) < 0
        )

    def place_buy_order(self, symbol: str) -> None:
        """
        This method places a buy order for a symbol.
        """
        quantity = self.calculate_position_size(symbol, side='buy')
        if quantity <= 0:
            print(f"No position to buy for {symbol}")
            return
        order = self.create_order(
            asset=symbol,
            quantity=quantity,
            side='buy',
        )
        self.submit_order(order)

    def place_sell_order(self, symbol: str) -> None:
        """
        This method places a sell order for a symbol.
        """
        quantity = self.calculate_position_size(symbol, side='sell')
        if quantity <= 0:
            print(f"No position to sell for {symbol}")
            return
        order = self.create_order(
            asset=symbol,
            quantity=quantity,
            side='sell',
        )
        self.submit_order(order)

    def calculate_position_size(self, symbol: str, side: str) -> float:
        """
        This method calculates the position size for a symbol.
        """
        equity = self.get_portfolio_value()
        risk_per_trade = equity * self.risk_per_trade
        atr = self.get_atr(symbol)
        if atr == 0:
            return 0.001  # Avoid division by zero

        position_size = risk_per_trade / atr

        if side == 'buy':
            position_size = min(
                position_size,
                (equity / len(self.symbols)) / self.get_last_price(asset=symbol)
            )
        elif side == 'sell':
            position_size = min(position_size, self.get_position(asset=symbol).quantity)

        return position_size

    def calculate_stop_loss(self, symbol: str) -> None:
        """
        This method calculates the stop loss for a symbol.
        """
        atr = self.get_atr(symbol)
        if atr == 0:
            self.stop_loss = self.get_last_price(asset=symbol) * self.stop_loss
        else:
            self.stop_loss = self.get_last_price(asset=symbol) - (atr * self.atr_multiplier)

    def get_atr(self, symbol: str) -> float:
        """
        This method calculates the average true range (ATR) for a symbol.
        """
        prices = self.get_historical_prices(symbol, self.atr_period + 1, 'day').df
        high, low, close = prices[['high', 'low', 'close']].values.T
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.column_stack((tr1, tr2, tr3))
        atr = np.convolve(tr.mean(axis=1), np.ones(self.atr_period) / self.atr_period, mode='valid')[-1]
        return atr

    def crossover(self, symbol: str) -> float:
        """
        This method calculates the crossover for a symbol.
        """
        prices = self.get_historical_prices(symbol, self.sma2 + 175, 'day').df['close']
        sma1 = prices.rolling(window=self.sma1).mean().iloc[-1]
        sma2 = prices.rolling(window=self.sma2).mean().iloc[-1]
        return sma1 - sma2


if __name__ == "__main__":
    trade = False  # Set to True to run the strategy live
    if trade:
        # Connect to Alpaca
        alpaca = Alpaca(AlpacaConfig)
        # Create the strategy and trader
        strategy = TrendFollowingStrategy(broker=alpaca)
        trader = Trader()
        trader.add_strategy(strategy)

        # Run the trader
        trader.run_all()
    else:
        # Specify the start and ending times
        backtesting_start = datetime(2020, 9, 30)
        backtesting_end = datetime(2021, 9, 30)

        # Run the backtest
        backtest = TrendFollowingStrategy.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=100000,
            benchmark_asset="XU100.IS",
            # The benchmark asset to use for the backtest to compare to.
        )
