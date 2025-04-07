"""
API connector module for interacting with cryptocurrency exchanges
Currently supports Binance
"""
import os
import time
import json
import hmac
import hashlib
import requests
from typing import Dict, List, Optional, Tuple, Union, Any
from urllib.parse import urlencode
from binance.client import Client
from binance.exceptions import BinanceAPIException

from config.config import (  # Updated: import all needed variables from config/config
    API_KEYS,
    USE_TESTNET,                 # Changed from testnet to USE_TESTNET
    MAX_API_RETRIES,             # Previously imported from config.trading_params
    API_RETRY_DELAY              # Previously imported from config.trading_params
)
from utils.logger import setup_logger
from utils.exceptions import ExchangeAPIException

logger = setup_logger("api_connector")

class BinanceConnector:
    """
    Connector for Binance API
    Handles all communications with Binance
    """
    def __init__(self, use_testnet: bool = None):
        """
        Initializes the Binance connector
        
        Args:
            use_testnet: Override config to use testnet or not
        """
        self.use_testnet = USE_TESTNET if use_testnet is None else use_testnet
        network = "testnet" if self.use_testnet else "production"
        # Accéder aux clés via le dictionnaire API_KEYS
        self.api_key = API_KEYS["binance"][network]["API_KEY"]
        self.api_secret = API_KEYS["binance"][network]["API_SECRET"]
        
        self.base_url = "https://testnet.binance.vision/api" if self.use_testnet else "https://api.binance.com/api"
        self.wss_url = "wss://testnet.binance.vision/ws" if self.use_testnet else "wss://stream.binance.com:9443/ws"
        
        self.client = self._initialize_client()
        
    def _initialize_client(self) -> Client:
        """
        Initializes the Binance client
        
        Returns:
            Binance client object
        """
        try:
            client = Client(self.api_key, self.api_secret, testnet=self.use_testnet)
            logger.info(f"Binance client initialized (testnet: {self.use_testnet})")
            
            # Test connectivity to the API
            client.ping()
            server_time = client.get_server_time()
            time_diff = int(time.time() * 1000) - server_time["serverTime"]
            
            logger.info(f"Connected to Binance API. Time difference: {time_diff} ms")
            
            return client
        except BinanceAPIException as e:
            logger.error(f"Failed to initialize Binance client: {str(e)}")
            raise ExchangeAPIException(f"Failed to initialize Binance client: {str(e)}")

    def _get_signature(self, params: Dict) -> str:
        """
        Génère la signature HMAC SHA256 requise pour les requêtes authentifiées
        
        Args:
            params: Paramètres de la requête
            
        Returns:
            Signature encodée en hexadécimal
        """
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                     signed: bool = False) -> Dict:
        """
        Effectue une requête à l'API Binance avec gestion des erreurs et des tentatives
        
        Args:
            method: Méthode HTTP (GET, POST, DELETE)
            endpoint: Point de terminaison de l'API
            params: Paramètres de la requête
            signed: Indique si la requête nécessite une signature
            
        Returns:
            Réponse de l'API sous forme de dictionnaire
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        
        # Paramètres par défaut
        if params is None:
            params = {}
        
        # Ajout du timestamp et de la signature pour les requêtes authentifiées
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._get_signature(params)
        
        # Tentatives avec backoff exponentiel
        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                if method == "GET":
                    response = requests.get(url, params=params, headers=headers)
                elif method == "POST":
                    response = requests.post(url, params=params, headers=headers)
                elif method == "DELETE":
                    response = requests.delete(url, params=params, headers=headers)
                else:
                    raise ValueError(f"Méthode HTTP non supportée: {method}")
                
                # Vérification du code de statut
                response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Tentative {attempt}/{MAX_API_RETRIES} échouée: {str(e)}")
                
                if attempt < MAX_API_RETRIES:
                    # Backoff exponentiel
                    wait_time = API_RETRY_DELAY * (2 ** (attempt - 1))
                    logger.info(f"Nouvelle tentative dans {wait_time} secondes...")
                    time.sleep(wait_time)
                else:
                    logger.error("Nombre maximum de tentatives atteint")
                    raise
    
    def test_connection(self) -> bool:
        """
        Teste la connexion à l'API
        
        Returns:
            True si la connexion est établie, False sinon
        """
        try:
            response = self._make_request("GET", "/v3/ping")
            return True
        except Exception as e:
            logger.error(f"Échec du test de connexion: {str(e)}")
            return False
    
    def get_exchange_info(self) -> Dict:
        """
        Récupère les informations sur l'échange
        
        Returns:
            Informations sur l'échange
        """
        return self._make_request("GET", "/v3/exchangeInfo")
    
    def get_account_info(self) -> Dict:
        """
        Récupère les informations du compte (nécessite une authentification)
        
        Returns:
            Informations du compte
        """
        return self._make_request("GET", "/v3/account", signed=True)
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Récupère le carnet d'ordres pour un symbole donné
        
        Args:
            symbol: Paire de trading (ex: BTCUSDT)
            limit: Nombre d'ordres à récupérer (max 5000)
            
        Returns:
            Carnet d'ordres
        """
        params = {
            "symbol": symbol,
            "limit": limit
        }
        return self._make_request("GET", "/v3/depth", params=params)
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Récupère les trades récents pour un symbole donné
        
        Args:
            symbol: Paire de trading
            limit: Nombre de trades à récupérer (max 1000)
            
        Returns:
            Liste des trades récents
        """
        params = {
            "symbol": symbol,
            "limit": limit
        }
        return self._make_request("GET", "/v3/trades", params=params)
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500, 
                  start_time: Optional[int] = None, end_time: Optional[int] = None) -> List:
        """
        Récupère les données de chandelier (klines/OHLCV)
        
        Args:
            symbol: Paire de trading
            interval: Intervalle de temps (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Nombre de chandeliers à récupérer (max 1000)
            start_time: Timestamp de début (millisecondes)
            end_time: Timestamp de fin (millisecondes)
            
        Returns:
            Liste des données OHLCV
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        return self._make_request("GET", "/v3/klines", params=params)
    
    def create_order(self, symbol: str, side: str, order_type: str, 
                    quantity: Optional[float] = None, price: Optional[float] = None,
                    time_in_force: str = "GTC", **kwargs) -> Dict:
        """
        Crée un nouvel ordre
        
        Args:
            symbol: Paire de trading
            side: Côté (BUY ou SELL)
            order_type: Type d'ordre (LIMIT, MARKET, STOP_LOSS, STOP_LOSS_LIMIT, etc.)
            quantity: Quantité à acheter ou vendre
            price: Prix pour les ordres à cours limité
            time_in_force: Durée de validité de l'ordre (GTC, IOC, FOK)
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Détails de l'ordre créé
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
        }
        
        if quantity:
            params["quantity"] = quantity
            
        if price and order_type != "MARKET":
            params["price"] = price
            
        if order_type == "LIMIT":
            params["timeInForce"] = time_in_force
            
        # Ajout des paramètres supplémentaires
        for key, value in kwargs.items():
            params[key] = value
            
        return self._make_request("POST", "/v3/order", params=params, signed=True)
    
    def get_order(self, symbol: str, order_id: Optional[int] = None, 
                orig_client_order_id: Optional[str] = None) -> Dict:
        """
        Récupère les détails d'un ordre
        
        Args:
            symbol: Paire de trading
            order_id: ID de l'ordre
            orig_client_order_id: ID client de l'ordre
            
        Returns:
            Détails de l'ordre
        """
        params = {"symbol": symbol}
        
        if order_id:
            params["orderId"] = order_id
        elif orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        else:
            raise ValueError("Vous devez spécifier order_id ou orig_client_order_id")
            
        return self._make_request("GET", "/v3/order", params=params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: Optional[int] = None, 
                    orig_client_order_id: Optional[str] = None) -> Dict:
        """
        Annule un ordre
        
        Args:
            symbol: Paire de trading
            order_id: ID de l'ordre
            orig_client_order_id: ID client de l'ordre
            
        Returns:
            Détails de l'ordre annulé
        """
        params = {"symbol": symbol}
        
        if order_id:
            params["orderId"] = order_id
        elif orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        else:
            raise ValueError("Vous devez spécifier order_id ou orig_client_order_id")
            
        return self._make_request("DELETE", "/v3/order", params=params, signed=True)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère tous les ordres ouverts
        
        Args:
            symbol: Paire de trading (optionnel)
            
        Returns:
            Liste des ordres ouverts
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
            
        return self._make_request("GET", "/v3/openOrders", params=params, signed=True)
    
    def get_leverage_brackets(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère les paliers d'effet de levier disponibles
        
        Args:
            symbol: Paire de trading (optionnel)
            
        Returns:
            Informations sur les paliers d'effet de levier
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
            
        return self._make_request("GET", "/fapi/v1/leverageBracket", params=params, signed=True)
    
    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        Définit l'effet de levier pour un symbole donné
        
        Args:
            symbol: Paire de trading
            leverage: Effet de levier (1-125)
            
        Returns:
            Résultat de l'opération
        """
        params = {
            "symbol": symbol,
            "leverage": leverage
        }
        
        # Pour l'API spot avec margin, utiliser l'endpoint correct
        return self._make_request("POST", "/sapi/v1/margin/leverage", params=params, signed=True)
