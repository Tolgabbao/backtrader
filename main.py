from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy
from lumibot.brokers.alpaca import Alpaca
from lumibot.traders import Trader
from config import AlpacaConfig


class TrendFollowingStrategy(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbols = ['AMD', 'INTC', 'NVDA', 'AAPL', 'TSLA', 'PLTR', 'AMZN', 'F', 'AI',
                        'GOOGL', 'IBM', 'META', 'BAC', 'BABA', 'GOOG', 'SONY', 'ASML', 'SAP', 'TSM',
                        'MU', 'VMW', 'DELL', 'HPQ', 'NET', 'STX', 'JNPR', 'LOGI', 'SONO', 'CRSR', 'NTGR'
                        ]  # Replace with your desired symbols
        self.sma1 = 50
        self.sma2 = 250
        self.stop_loss = 0.08
        self.risk_per_trade = 0.03

    def buy_order(self, symbol):
        order = self.create_order(
            asset=symbol,
            quantity=self.calculate_position_size(symbol, side='buy'),
            side='buy',
        )
        self.submit_order(order)

    def sell_order(self, symbol):
        order = self.create_order(
            asset=symbol,
            quantity=self.calculate_position_size(symbol, side='sell'),
            side='sell',
        )
        self.submit_order(order)

    def on_trading_iteration(self):
        if self.broker.is_market_open():
            for symbol in self.symbols:
                position = self.get_position(asset=symbol)
                if position is None:
                    if self.crossover(symbol) > 0:
                        self.buy_order(symbol)
                elif position.quantity > 0:
                    if self.get_historical_prices(symbol, 5, 'day').df['close'][-2] < (1 - self.stop_loss)\
                            * self.get_last_price(asset=symbol):
                        self.sell_order(symbol)
                    elif self.crossover(symbol) < 0:
                        self.sell_order(symbol)

    def crossover(self, symbol):
        prices = self.get_historical_prices(symbol, self.sma2+180, 'day').df['close']
        sma1 = prices.rolling(window=self.sma1).mean().iloc[-1]
        sma2 = prices.rolling(window=self.sma2).mean().iloc[-1]
        return sma1 - sma2

    def calculate_position_size(self, symbol, side) -> float:
        equity = self.get_portfolio_value()
        stop_loss_amount = self.get_last_price(asset=symbol) * self.stop_loss
        position_size = (equity * self.risk_per_trade) / stop_loss_amount
        if side == 'sell' and position_size > self.get_position(asset=symbol).quantity:
            position_size = self.get_position(asset=symbol).quantity
        elif side == 'buy' and position_size * self.get_last_price(asset=symbol) > self.get_cash():
            position_size = (self.get_cash() + 100000) * self.risk_per_trade // stop_loss_amount
        if position_size < 0:
            return 0.001
        return position_size


if __name__ == "__main__":
    trade = True
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
        backtesting_start = datetime(2023, 3, 30)
        backtesting_end = datetime(2023, 6, 30)

        # Run the backtest
        backtest = TrendFollowingStrategy.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=100000,
            benchmark_asset="SPY",
            # The benchmark asset to use for the backtest to compare to.
        )
