from lumibot.strategies import Strategy
from lumibot.brokers.alpaca import Alpaca
from lumibot.traders import Trader


class TrendFollowingStrategy(Strategy):
    def __init__(self, broker):
        super().__init__(broker)
        self.symbol = 'AAPL'  # Replace with your desired symbol
        self.sma1 = 50
        self.sma2 = 200
        self.stop_loss = 0.05
        self.risk_per_trade = 0.02

    def buy_order(self):
        self.create_order(
            asset=self.symbol,
            quantity=self.calculate_position_size(),
            side='buy',
        )

    def sell_order(self):
        self.create_order(
            asset=self.symbol,
            quantity=self.calculate_position_size(),
            side='sell',
        )

    def on_trading_iteration(self):
        if self.broker.is_market_open():
            if self.get_position(asset=self.symbol) is None:
                if self.crossover() > 0:
                    self.buy_order()
            elif self.get_position(asset=self.symbol).quantity > 0:
                if self.get_last_price(asset=self.symbol) < (1 - self.stop_loss) * self.get_last_price():
                    selling_order = self.get_selling_order(self.get_position)
                    if selling_order:
                        self.cancel_order(selling_order)
                    self.sell_order()
                elif self.crossover() < 0:
                    self.sell_order()

    def crossover(self):
        prices = self.broker.get_historical_prices(self.symbol, self.sma2, 'day').df['close']
        sma1 = prices.rolling(window=self.sma1).mean().iloc[-1]
        sma2 = prices.rolling(window=self.sma2).mean().iloc[-1]
        return sma1 - sma2

    def calculate_position_size(self):
        risk = self.broker.get_account().cash * self.risk_per_trade
        stop_loss_amount = self.get_last_price(asset=self.symbol) * self.stop_loss
        position_size = risk / stop_loss_amount
        return position_size

# Connect to Alpaca
alpaca = Alpaca(AlpacaConfig)


# Create the strategy and trader
strategy = TrendFollowingStrategy(broker=alpaca)
trader = Trader()
trader.add_strategy(strategy)

# Run the trader
trader.run_all()
