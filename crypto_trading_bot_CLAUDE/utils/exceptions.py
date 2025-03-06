"""
Custom exceptions for the crypto trading bot
"""

class ExchangeAPIException(Exception):
    """
    Exception raised when an error occurs with exchange API communication
    """
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)
        
    def __str__(self):
        if self.status_code:
            return f"ExchangeAPIException (Status {self.status_code}): {self.message}"
        return f"ExchangeAPIException: {self.message}"


class InsufficientFundsException(Exception):
    """
    Exception raised when there are not enough funds to execute a trade
    """
    def __init__(self, message: str = "Insufficient funds to execute trade", asset: str = None, balance: float = None, required: float = None):
        self.message = message
        self.asset = asset
        self.balance = balance
        self.required = required
        super().__init__(self.message)
        
    def __str__(self):
        if self.asset and self.balance is not None and self.required is not None:
            return f"InsufficientFundsException: {self.message} (Have: {self.balance} {self.asset}, Need: {self.required} {self.asset})"
        return f"InsufficientFundsException: {self.message}"


class InvalidConfigException(Exception):
    """
    Exception raised when there's an issue with the configuration
    """
    def __init__(self, message: str, param: str = None):
        self.message = message
        self.param = param
        super().__init__(self.message)
        
    def __str__(self):
        if self.param:
            return f"InvalidConfigException: {self.message} (Parameter: {self.param})"
        return f"InvalidConfigException: {self.message}"


class StrategyException(Exception):
    """
    Exception raised when there's an issue with a trading strategy
    """
    def __init__(self, message: str, strategy_name: str = None):
        self.message = message
        self.strategy_name = strategy_name
        super().__init__(self.message)
        
    def __str__(self):
        if self.strategy_name:
            return f"StrategyException ({self.strategy_name}): {self.message}"
        return f"StrategyException: {self.message}"


class MarketDataError(Exception):
    """
    Exception raised when there's an issue with market data retrieval or processing
    """
    def __init__(self, message: str, symbol: str = None):
        self.message = message
        self.symbol = symbol
        super().__init__(self.message)
        
    def __str__(self):
        if self.symbol:
            return f"MarketDataError ({self.symbol}): {self.message}"
        return f"MarketDataError: {self.message}"


class ModelError(Exception):
    """
    Exception raised when there's an issue with AI models
    """
    def __init__(self, message: str, model_name: str = None):
        self.message = message
        self.model_name = model_name
        super().__init__(self.message)
        
    def __str__(self):
        if self.model_name:
            return f"ModelError ({self.model_name}): {self.message}"
        return f"ModelError: {self.message}"


class TradingExecutionError(Exception):
    """
    Exception raised when there's an error executing a trade
    """
    def __init__(self, message: str, order_id: str = None, symbol: str = None):
        self.message = message
        self.order_id = order_id
        self.symbol = symbol
        super().__init__(self.message)
        
    def __str__(self):
        if self.order_id and self.symbol:
            return f"TradingExecutionError ({self.symbol}, Order ID: {self.order_id}): {self.message}"
        elif self.symbol:
            return f"TradingExecutionError ({self.symbol}): {self.message}"
        return f"TradingExecutionError: {self.message}"
