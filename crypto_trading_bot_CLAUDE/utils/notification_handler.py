"""
Notification handling system for sending alerts about bot actions and events
"""
import os
import requests
import json
from typing import Optional, Dict, Any
from datetime import datetime

from utils.logger import setup_logger

logger = setup_logger("notification_handler")

class NotificationHandler:
    """
    Handles sending notifications via various channels (Telegram, email, etc.)
    """
    def __init__(self, telegram_token: Optional[str] = None, 
               telegram_chat_id: Optional[str] = None,
               email_config: Optional[Dict] = None):
        """
        Initialize the notification handler
        
        Args:
            telegram_token: Telegram bot token
            telegram_chat_id: Telegram chat ID to send messages to
            email_config: Email configuration for sending emails
        """
        self.telegram_enabled = bool(telegram_token and telegram_chat_id)
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        
        self.email_enabled = bool(email_config)
        self.email_config = email_config
        
        # Track sent notifications for rate limiting
        self.notification_history = []
        self.max_history_size = 100
        
        # Rate limiting settings
        self.rate_limit = {
            "max_per_minute": 5,
            "max_per_hour": 20
        }
        
        logger.info(f"Notification handler initialized. Telegram enabled: {self.telegram_enabled}")
    
    def send_message(self, title: str, message: str, 
                   level: str = "info", 
                   send_telegram: bool = True,
                   send_email: bool = False) -> Dict:
        """
        Send a notification message through configured channels
        
        Args:
            title: Title/subject of the message
            message: Main content of the message
            level: Message level (info, warning, error, success)
            send_telegram: Whether to send via Telegram
            send_email: Whether to send via email
            
        Returns:
            Result of the notification attempt
        """
        # Check rate limits
        if self._is_rate_limited():
            logger.warning("Rate limit reached, skipping notification")
            return {"success": False, "reason": "rate_limited"}
        
        # Record this notification
        notification = {
            "title": title,
            "level": level,
            "timestamp": datetime.now().isoformat(),
            "channels": []
        }
        self.notification_history.append(notification)
        
        # Trim history if needed
        if len(self.notification_history) > self.max_history_size:
            self.notification_history = self.notification_history[-self.max_history_size:]
        
        result = {"success": True, "channels": []}
        
        # Send via Telegram
        if send_telegram and self.telegram_enabled:
            telegram_result = self._send_telegram(title, message, level)
            
            if telegram_result.get("success"):
                notification["channels"].append("telegram")
                result["channels"].append("telegram")
            else:
                result["telegram_error"] = telegram_result.get("error")
                result["success"] = False
        
        # Send via email
        if send_email and self.email_enabled:
            email_result = self._send_email(title, message, level)
            
            if email_result.get("success"):
                notification["channels"].append("email")
                result["channels"].append("email")
            else:
                result["email_error"] = email_result.get("error")
                result["success"] = False
        
        return result
    
    def _send_telegram(self, title: str, message: str, level: str) -> Dict:
        """
        Send a message via Telegram
        
        Args:
            title: Message title
            message: Message body
            level: Message level
            
        Returns:
            Result of the Telegram API call
        """
        if not self.telegram_enabled:
            return {"success": False, "error": "Telegram not configured"}
        
        # Prepare message with emojis based on level
        level_emojis = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅"
        }
        
        emoji = level_emojis.get(level.lower(), "ℹ️")
        formatted_message = f"{emoji} *{title}*\n\n{message}"
        
        # Send message
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": formatted_message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Telegram notification sent: {title}")
                return {"success": True}
            else:
                logger.error(f"Failed to send Telegram message: {response.text}")
                return {"success": False, "error": response.text}
        
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _send_email(self, title: str, message: str, level: str) -> Dict:
        """
        Send a message via email
        
        Args:
            title: Email subject
            message: Email body
            level: Message level
            
        Returns:
            Result of the email sending attempt
        """
        if not self.email_enabled:
            return {"success": False, "error": "Email not configured"}
        
        # This could be implemented using smtplib or an email service API
        logger.info("Email functionality not implemented yet")
        return {"success": False, "error": "Not implemented"}
    
    def _is_rate_limited(self) -> bool:
        """
        Check if sending would exceed rate limits
        
        Returns:
            Whether rate limit is reached
        """
        now = datetime.now()
        minute_ago = now.replace(second=0, microsecond=0)
        hour_ago = now.replace(minute=0, second=0, microsecond=0)
        
        # Count notifications in the last minute and hour
        per_minute = 0
        per_hour = 0
        
        for notification in self.notification_history:
            try:
                timestamp = datetime.fromisoformat(notification["timestamp"])
                
                if timestamp >= minute_ago:
                    per_minute += 1
                
                if timestamp >= hour_ago:
                    per_hour += 1
            except:
                continue
        
        # Check against limits
        if per_minute >= self.rate_limit["max_per_minute"]:
            return True
        
        if per_hour >= self.rate_limit["max_per_hour"]:
            return True
        
        return False
    
    def get_notification_stats(self) -> Dict:
        """
        Get statistics about sent notifications
        
        Returns:
            Notification statistics
        """
        if not self.notification_history:
            return {"count": 0, "by_level": {}}
        
        # Count by level
        by_level = {}
        for notification in self.notification_history:
            level = notification.get("level", "unknown")
            by_level[level] = by_level.get(level, 0) + 1
        
        # Count by channel
        by_channel = {"telegram": 0, "email": 0}
        for notification in self.notification_history:
            channels = notification.get("channels", [])
            for channel in channels:
                by_channel[channel] = by_channel.get(channel, 0) + 1
        
        return {
            "count": len(self.notification_history),
            "by_level": by_level,
            "by_channel": by_channel,
            "last_notification": self.notification_history[-1]["timestamp"] if self.notification_history else None
        }
