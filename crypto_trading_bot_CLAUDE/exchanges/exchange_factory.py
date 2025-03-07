"""
Factory for creating exchange clients
"""
from typing import Optional
from utils.logger import setup_logger
from exchanges.exchange_client import ExchangeClient
from exchanges.binance_client import BinanceClient
from exchanges.paper_trading import PaperTradingClient

logger = setup_logger("exchange_factory")

def create_exchange(name: str, api_key: str = None, api_secret: str = None, 
                   testnet: bool = True, **kwargs) -> Optional[ExchangeClient]:
    """
    Create an exchange client instance
    
    Args:
        name: Exchange name (binance, paper, etc.)
        api_key: API key
        api_secret: API secret
        testnet: Use testnet (for real exchanges)
        **kwargs: Additional exchange-specific parameters
        
    Returns:
        Exchange client instance
    """
    name = name.lower()
    
    if name == "binance":
        return BinanceClient(api_key=api_key, api_secret=api_secret, testnet=testnet, **kwargs)
    elif name == "paper":
        return PaperTradingClient(**kwargs)
    else:
        logger.error(f"Unsupported exchange: {name}")
        return None
