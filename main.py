from datetime import datetime
from typing import List, Tuple
import numpy as np
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy
from lumibot.brokers.alpaca import Alpaca
from lumibot.traders import Trader
from config import AlpacaConfig


class TrendFollowingStrategy(Strategy):
    def __init__(self, *args: Tuple, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.symbols: List[str] = [
            'AMD', 'INTC', 'NVDA', 'AAPL', 'TSLA', 'PLTR', 'AMZN', 'F', 'GM', 'MSFT', 'NFLX', 'DIS',
            'GOOGL', 'IBM', 'META', 'BAC', 'BABA', 'GOOG', 'SONY', 'ASML', 'SAP', 'TSM']
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
                if self.should_place_buy_order(symbol):
                    self.place_buy_order(symbol)
                else:
                    if not self.has_position(symbol):
                        continue
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
            # print(f"No position to buy for {symbol}")
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
            # print(f"No position to sell for {symbol}")
            return
        order = self.create_order(
            asset=symbol,
            quantity=quantity,
            side='sell',
        )
        self.submit_order(order)
    
    def set_symbols(self, symbols: List[str]) -> None:
        """
        This method sets the symbols for the strategy.
        """
        self.symbols = symbols

    def calculate_position_size(self, symbol: str, side: str) -> float:
        """
        This method calculates the position size for a symbol.
        """

        """
        Higher level of risk per trade if the ATR is higher
        """

        atr = self.get_atr(symbol)
        if atr == 0:
            atr = 1
        risk_per_trade = self.risk_per_trade * atr
        cash = self.get_cash()

        # Calculate the position size using equity and risk per trade
        if cash > 0:
            position_size = (cash * risk_per_trade) / self.get_last_price(asset=symbol)
        else:
            position_size = 0

        if side == 'buy':
            return position_size
        else:
            return self.get_position(asset=symbol).quantity


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
        atr = np.convolve(tr.mean(axis=1), np.ones(self.atr_period)/self.atr_period, mode='valid')[-1]
        atr = float(np.round(atr, 2))
        return atr
    
    def crossover(self, symbol: str) -> float:
        """
        This method calculates the crossover for a symbol.
        """
        prices = self.get_historical_prices(symbol, self.sma2+175, 'day').df['close']
        sma1 = prices.rolling(window=self.sma1).mean().iloc[-1]
        sma2 = prices.rolling(window=self.sma2).mean().iloc[-1]
        return sma1 - sma2



"""
Moving Averages (MA): Shows the average price of an asset over a specific period. Widely used to identify trends and support/resistance levels.
Relative Strength Index (RSI): Measures the magnitude of recent price changes to identify overbought or oversold conditions.
Stochastic Oscillator: Compares the closing price to the price range within a specific period, often used to identify momentum shifts.
MACD (Moving Average Convergence Divergence): Compares two moving averages to identify trend strength and potential reversals.
Bollinger Bands: Indicate price volatility based on standard deviations above and below a moving average.
Williams %R: Measures the closing price relative to the price range within a specific period, often used to identify overbought or oversold conditions."""
class AdvancedStrategy(TrendFollowingStrategy):
    def __init__(self, *args: Tuple, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.symbols: List[str] = [
            'AMD', 'INTC', 'NVDA', 'AAPL', 'TSLA', 'PLTR', 'AMZN', 'F', 'GM', 'MSFT', 'NFLX', 'DIS',
            'GOOGL', 'IBM', 'META', 'BAC', 'BABA', 'GOOG', 'SONY', 'ASML', 'SAP', 'TSM']
        # Replace with your desired symbols
        self.sma1: int = 50
        self.sma2: int = 250
        self.risk_per_trade: float = 0.03
        self.atr_period: int = 14
        self.atr_multiplier: int = 2
        self.stop_loss: float = 0
    def should_place_buy_order(self, symbol: str) -> bool:
        """
        This method checks if a buy order should be placed for a symbol.
        """
        if self.weighted_combination_indicators(symbol) > 4:
            return True
        else:
            return False

    def should_place_sell_order(self, symbol: str) -> bool:
        """
        This method checks if a sell order should be placed for a symbol.
        """
        position = self.get_position(asset=symbol)
        if position is None:
            return False
        return (
            self.get_last_price(asset=symbol) <= self.stop_loss or
            self.weighted_combination_indicators(symbol) < 2
        )

    def calculate_indicators(self, symbol):
        # Fetch historical prices once for all indicator calculations
        historical_prices = self.get_historical_prices(symbol, self.sma2 + 175, 'day').df
        close_prices = historical_prices['close']
        low_prices = historical_prices['low']
        high_prices = historical_prices['high']

        # Calculate MACD
        ema1 = close_prices.ewm(span=self.sma1, adjust=False).mean().iloc[-1]
        ema2 = close_prices.ewm(span=self.sma2, adjust=False).mean().iloc[-1]
        macd = ema1 - ema2

        # Calculate RSI
        delta = close_prices.diff()
        delta = delta[1:]
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.ewm(span=self.sma1).mean()
        roll_down = down.abs().ewm(span=self.sma1).mean()
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs)).iloc[-1]

        # Calculate Stochastic Oscillator
        k = 100 * ((close_prices - low_prices.rolling(window=self.sma1).min()) /
                   (high_prices.rolling(window=self.sma1).max() - low_prices.rolling(window=self.sma1).min())).iloc[-1]

        # Calculate Bollinger Bands
        sma = close_prices.rolling(window=self.sma1).mean().iloc[-1]
        std = close_prices.rolling(window=self.sma1).std().iloc[-1]
        upper = sma + (std * 2)
        lower = sma - (std * 2)

        # Calculate Williams %R
        r = -100 * ((high_prices.rolling(window=self.sma1).max() - close_prices) /
                    (high_prices.rolling(window=self.sma1).max() - low_prices.rolling(window=self.sma1).min())).iloc[-1]

        return macd, rsi, k, (upper, lower), r

    def weighted_combination_indicators(self, symbol):
        macd, rsi, k, bb, r = self.calculate_indicators(symbol)
        result = 0

        if macd > 0:
            result += 0.3
        if rsi > 70:
            result += 1.5
        if k > 80:
            result += 1.5
        if bb[0] < self.get_last_price(asset=symbol):
            result += 1
        if bb[1] > self.get_last_price(asset=symbol):
            result += 1
        if r > -20:
            result += 1.5

        return result


if __name__ == "__main__":
    trade = True  # Set to True to run the strategy live
    if trade:
        # Connect to Alpaca
        alpaca = Alpaca(AlpacaConfig)
        # Create the strategy and trader
        strategy = AdvancedStrategy(broker=alpaca)
        trader = Trader()
        trader.add_strategy(strategy)

        # Run the trader
        trader.run_all()
    else:
        # Specify the start and ending times
        backtesting_start = datetime(2023, 3, 30)
        backtesting_end = datetime(2023, 12, 1)

        # Run the backtest
        backtest = AdvancedStrategy.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=100000,
            benchmark_asset="SPY",
            # The benchmark asset to use for the backtest to compare to.
        )
