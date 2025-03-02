# main.py
"""
Point d'entrée principal du bot de trading crypto
"""
import logging
import time
from datetime import datetime
import signal
import sys
import requests

from config.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE
from core.api_connector import BinanceConnector
from core.data_fetcher import MarketDataFetcher
from core.order_manager import OrderManager
from core.position_tracker import PositionTracker
from core.risk_manager import RiskManager
from strategies.technical_bounce import TechnicalBounceStrategy
from strategies.market_state import MarketStateAnalyzer
from ai.scoring_engine import ScoringEngine
from utils.logger import setup_logger

# Configuration du logger
logger = setup_logger("main", LOG_LEVEL, LOG_FORMAT, LOG_FILE)

class TradingBot:
    """
    Classe principale du bot de trading crypto
    """
    def __init__(self):
        logger.info("Initialisation du bot de trading...")
        
        # Initialisation des composants
        self.api = BinanceConnector()
        self.data_fetcher = MarketDataFetcher(self.api)
        self.risk_manager = RiskManager()
        self.position_tracker = PositionTracker()
        self.order_manager = OrderManager(self.api, self.position_tracker)
        
        # Initialisation des stratégies
        self.market_analyzer = MarketStateAnalyzer(self.data_fetcher)
        self.scoring_engine = ScoringEngine()
        self.strategy = TechnicalBounceStrategy(
            self.data_fetcher, 
            self.market_analyzer,
            self.scoring_engine
        )
        
        # Variables d'état
        self.is_running = False
        self.last_trade_time = {}  # Pour suivre le temps entre les trades
        
        # Configuration des gestionnaires de signaux
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        logger.info("Bot initialisé avec succès")

    def start(self):
        """
        Démarre le bot de trading
        """
        self.is_running = True
        logger.info("Démarrage du bot de trading...")
        
        try:
            # Validation de la connexion à l'API
            if not self.api.test_connection():
                logger.error("Échec de la connexion à l'API Binance. Arrêt du bot.")
                return
            
            logger.info("Connexion à l'API Binance réussie.")
            account_info = self.api.get_account_info()
            self.risk_manager.update_account_balance(account_info)
            
            # Boucle principale
            while self.is_running:
                self.trading_cycle()
                time.sleep(15)  # Attente de 15 secondes entre chaque cycle
                
        except Exception as e:
            logger.error(f"Erreur critique lors de l'exécution: {str(e)}")
            self.shutdown()
    
    def trading_cycle(self):
        """
        Cycle principal de trading avec gestion d'erreurs améliorée
        """
        from config.config import TRADING_PAIRS
        
        try:
            current_time = datetime.now()
            
            for pair in TRADING_PAIRS:
                try:
                    # Vérification du cooldown entre trades
                    if pair in self.last_trade_time:
                        from config.trading_params import MIN_TIME_BETWEEN_TRADES
                        time_since_last_trade = (current_time - self.last_trade_time[pair]).total_seconds() / 60
                        if time_since_last_trade < MIN_TIME_BETWEEN_TRADES:
                            logger.debug(f"Cooldown actif pour {pair}: {time_since_last_trade:.1f}/{MIN_TIME_BETWEEN_TRADES} minutes écoulées")
                            continue
                    
                    # Analyse de l'état du marché
                    market_state = self.market_analyzer.analyze_market_state(pair)
                    if not market_state["favorable"]:
                        logger.info(f"Marché défavorable pour {pair}: {market_state['reason']}")
                        continue
                    
                    # Vérification des conditions de risque
                    if not self.risk_manager.can_open_new_position(self.position_tracker):
                        logger.info(f"Conditions de risque non remplies pour {pair}")
                        continue
                    
                    # Recherche d'opportunités de trading
                    opportunity = self.strategy.find_trading_opportunity(pair)
                    if opportunity and opportunity["score"] >= self.strategy.min_score:
                        logger.info(f"Opportunité trouvée pour {pair} (score: {opportunity['score']})")
                        
                        # Calculer le montant à trader
                        trade_amount = self.risk_manager.calculate_position_size(pair, opportunity)
                        
                        if trade_amount > 0:
                            # Exécution de l'ordre
                            order_result = self.order_manager.place_entry_order(
                                pair, 
                                opportunity["side"], 
                                trade_amount, 
                                opportunity["entry_price"],
                                opportunity["stop_loss"],
                                opportunity["take_profit"]
                            )
                            
                            if order_result["success"]:
                                self.last_trade_time[pair] = current_time
                                logger.info(f"Trade exécuté sur {pair}: {order_result}")
                                
                                # Enregistrement des données du trade pour analyse
                                self.strategy.log_trade(opportunity, order_result)
                                
                                # Notification de l'ordre placé
                                self._send_notification(
                                    f"Ordre placé: {pair} {opportunity['side']} à {order_result['entry_price']} " +
                                    f"(SL: {order_result['stop_loss_price']}, TP: {order_result['take_profit_price']})"
                                )
                            else:
                                logger.error(f"Échec de l'ordre pour {pair}: {order_result['message']}")
                    
                    # Gestion des positions ouvertes
                    self.manage_open_positions(pair)
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"Erreur réseau pour {pair}: {str(e)}")
                    time.sleep(5)  # Attendre 5 secondes avant de continuer
                    continue
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de {pair}: {str(e)}")
                    continue
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur réseau lors de la communication avec Binance: {str(e)}")
            # Attendre et réessayer au prochain cycle
            time.sleep(30)
            return
        except Exception as e:
            logger.critical(f"Erreur critique dans le cycle de trading: {str(e)}")
            if self._is_critical_error(str(e)):
                self._send_emergency_notification(f"Erreur critique: {str(e)}")
                self.shutdown()

    def manage_open_positions(self, pair):
        """
        Gestion des positions ouvertes (trailing stops, etc.)
        """
        open_positions = self.position_tracker.get_open_positions(pair)
        
        for position in open_positions:
            # Mise à jour des données de marché
            current_price = self.data_fetcher.get_current_price(pair)
            
            # Mise à jour des trailing stops si nécessaire
            self.order_manager.update_trailing_stop(pair, position, current_price)
    
    def handle_shutdown(self, signum, frame):
        """
        Gestionnaire de signal pour arrêt propre
        """
        logger.info("Signal d'arrêt reçu. Arrêt en cours...")
        self.shutdown()
    
    def shutdown(self):
        """
        Arrêt propre du bot
        """
        self.is_running = False
        
        # Fermeture propre des positions si nécessaire
        # (Peut être commenté pour conserver les positions ouvertes)
        #self.close_all_positions()
        
        logger.info("Bot arrêté avec succès")
        sys.exit(0)
    
    def close_all_positions(self):
        """
        Ferme toutes les positions ouvertes
        """
        all_positions = self.position_tracker.get_all_open_positions()
        
        for pair, positions in all_positions.items():
            for position in positions:
                self.order_manager.close_position(pair, position["id"])
        
        logger.info("Toutes les positions ont été fermées")
    def _is_critical_error(self, error_message: str) -> bool:
        """
        Détermine si une erreur est critique et nécessite un arrêt
        
        Args:
            error_message: Message d'erreur
            
        Returns:
            True si l'erreur est critique, False sinon
        """
        critical_keywords = [
            "Authentication failed", "API key expired", "IP has been banned",
            "Account has been frozen", "Insufficient balance", "System error",
            "Fatal error", "Database corruption"
        ]
        
        return any(keyword in error_message for keyword in critical_keywords)

    def _send_notification(self, message: str, level: str = "info") -> None:
        """
        Envoie une notification
        
        Args:
            message: Message à envoyer
            level: Niveau de la notification (info, warning, critical)
        """
        logger.info(f"Notification ({level}): {message}")
        
        # Si les notifications sont activées dans la configuration
        if hasattr(self, 'notification_service'):
            self.notification_service.send(message, level)

    def _send_emergency_notification(self, message: str) -> None:
        """
        Envoie une notification d'urgence
        
        Args:
            message: Message d'urgence
        """
        self._send_notification(message, "critical")
        
        # Tentative d'envoi par tous les canaux disponibles
        from config.config import NOTIFICATION_EMAIL, ENABLE_NOTIFICATIONS
        
        if ENABLE_NOTIFICATIONS and NOTIFICATION_EMAIL:
            try:
                import smtplib
                from email.mime.text import MIMEText
                from config.config import SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
                
                msg = MIMEText(f"URGENCE - BOT DE TRADING: {message}")
                msg['Subject'] = "ALERTE CRITIQUE - Bot de Trading"
                msg['From'] = SMTP_USER
                msg['To'] = NOTIFICATION_EMAIL
                
                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                    server.starttls()
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.send_message(msg)
            except Exception as e:
                logger.error(f"Impossible d'envoyer l'email d'urgence: {str(e)}")

if __name__ == "__main__":
    bot = TradingBot()
    bot.start()