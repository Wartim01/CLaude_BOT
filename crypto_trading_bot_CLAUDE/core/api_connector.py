# core/api_connector.py
"""
Connecteur pour l'API Binance
Gère les connexions et les appels à l'API de l'échange
"""
import time
import logging
import hmac
import hashlib
import requests
import urllib.parse
from typing import Dict, List, Any, Optional, Union

from config.config import (
    BINANCE_API_KEY, 
    BINANCE_API_SECRET, 
    USE_TESTNET,
    MAX_API_RETRIES, 
    API_RETRY_DELAY
)
from utils.logger import setup_logger

logger = setup_logger("api_connector")

class BinanceConnector:
    """
    Gère les connexions et les appels à l'API Binance
    """
    def __init__(self):
        self.api_key = BINANCE_API_KEY
        self.api_secret = BINANCE_API_SECRET
        
        # Configuration des URLs en fonction du mode (testnet ou production)
        if USE_TESTNET:
            self.base_url = "https://testnet.binance.vision/api"
            logger.info("Mode TestNet activé")
        else:
            self.base_url = "https://api.binance.com/api"
            logger.info("Mode Production activé")
    
    def _get_signature(self, params: Dict) -> str:
        """
        Génère la signature HMAC SHA256 requise pour les requêtes authentifiées
        
        Args:
            params: Paramètres de la requête
            
        Returns:
            Signature encodée en hexadécimal
        """
        query_string = urllib.parse.urlencode(params)
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
        