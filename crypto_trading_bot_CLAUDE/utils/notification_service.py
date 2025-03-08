# utils/notification_service.py
"""
Service de notification pour le bot de trading
"""
import logging
import smtplib
import requests
from email.mime.text import MIMEText
from typing import Dict, Optional, Union
from datetime import datetime

from config.config import (
    ENABLE_NOTIFICATIONS,
    NOTIFICATION_EMAIL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    SMTP_SERVER,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD
)
from utils.logger import setup_logger

logger = setup_logger("notification_service")

class NotificationService:
    """
    GÃ¨re l'envoi de notifications par diffÃ©rents canaux
    """
    def __init__(self):
        self.enabled = ENABLE_NOTIFICATIONS
        self.last_notification_time = {}  # {channel: timestamp}
        self.notification_cooldown = 60  # secondes entre les notifications
    
    def send(self, message: str, level: str = "info") -> bool:
        """
        Envoie une notification par tous les canaux disponibles
        
        Args:
            message: Message Ã  envoyer
            level: Niveau de la notification (info, warning, critical)
            
        Returns:
            True si au moins une notification a Ã©tÃ© envoyÃ©e, False sinon
        """
        if not self.enabled:
            logger.debug(f"Notifications dÃ©sactivÃ©es, message ignorÃ©: {message}")
            return False
        
        # Formater le message
        formatted_message = self._format_message(message, level)
        
        # Flag pour suivre si au moins une notification a Ã©tÃ© envoyÃ©e
        notification_sent = False
        
        # Email
        if NOTIFICATION_EMAIL and (level == "critical" or self._can_send_notification("email")):
            if self._send_email(NOTIFICATION_EMAIL, formatted_message, level):
                notification_sent = True
                self.last_notification_time["email"] = datetime.now()
        
        # Telegram
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and (level == "critical" or self._can_send_notification("telegram")):
            if self._send_telegram(formatted_message):
                notification_sent = True
                self.last_notification_time["telegram"] = datetime.now()
        
        return notification_sent
    
    def _format_message(self, message: str, level: str) -> str:
        """
        Formate un message de notification
        
        Args:
            message: Message Ã  formater
            level: Niveau de la notification
            
        Returns:
            Message formatÃ©
        """
        prefix = {
            "info": "INFO",
            "warning": "âš ï¸� AVERTISSEMENT",
            "critical": "ğŸš¨ ALERTE CRITIQUE"
        }.get(level, "INFO")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"[{prefix}] [{timestamp}] {message}"
    
    def _can_send_notification(self, channel: str) -> bool:
        """
        VÃ©rifie si une notification peut Ãªtre envoyÃ©e sur un canal
        
        Args:
            channel: Canal de notification
            
        Returns:
            True si une notification peut Ãªtre envoyÃ©e, False sinon
        """
        if channel not in self.last_notification_time:
            return True
        
        time_since_last = (datetime.now() - self.last_notification_time[channel]).total_seconds()
        return time_since_last >= self.notification_cooldown
    
    def _send_email(self, recipient: str, message: str, level: str) -> bool:
        """
        Envoie une notification par email
        
        Args:
            recipient: Adresse email du destinataire
            message: Message Ã  envoyer
            level: Niveau de la notification
            
        Returns:
            True si l'email a Ã©tÃ© envoyÃ©, False sinon
        """
        if not (SMTP_SERVER and SMTP_PORT and SMTP_USER and SMTP_PASSWORD):
            logger.error("Configuration SMTP incomplÃ¨te, impossible d'envoyer l'email")
            return False
        
        try:
            subject = {
                "info": "Bot de Trading - Information",
                "warning": "Bot de Trading - Avertissement",
                "critical": "BOT DE TRADING - ALERTE CRITIQUE"
            }.get(level, "Bot de Trading - Notification")
            
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = SMTP_USER
            msg['To'] = recipient
            
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
            
            logger.info(f"Email envoyÃ© Ã  {recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de l'email: {str(e)}")
            return False
    
    def _send_telegram(self, message: str) -> bool:
        """
        Envoie une notification par Telegram
        
        Args:
            message: Message Ã  envoyer
            
        Returns:
            True si le message a Ã©tÃ© envoyÃ©, False sinon
        """
        if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
            logger.error("Configuration Telegram incomplÃ¨te")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("Message Telegram envoyÃ©")
                return True
            else:
                logger.error(f"Erreur lors de l'envoi du message Telegram: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi du message Telegram: {str(e)}")
            return False