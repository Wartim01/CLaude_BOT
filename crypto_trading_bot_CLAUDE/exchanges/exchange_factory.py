"""
Factory for creating exchange clients
"""
import os
from typing import Any, Optional
from config.config import load_config, CONFIG_PATH
from utils.logger import setup_logger
from exchanges.exchange_client import ExchangeClient
from exchanges.binance_client import BinanceClient
from exchanges.paper_trading import PaperTradingClient

logger = setup_logger("exchange_factory")

def create_exchange(name: str, **kwargs) -> Any:
    """
    Fabrique un client d'échange selon le nom fourni.
    
    Args:
        name: Nom de l'échange ("binance", "paper", etc.)
        **kwargs: Paramètres spécifiques à passer au client
    
    Returns:
        Instance d'un ExchangeClient adapté
    """
    if name.lower() == "binance":
        return BinanceClient(**kwargs)
    elif name.lower() == "paper":
        return PaperTradingClient(**kwargs)
    else:
        raise ValueError(f"Client d'échange pour '{name}' non supporté.")
