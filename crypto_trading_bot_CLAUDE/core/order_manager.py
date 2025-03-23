"""
Gestionnaire d'ordres pour le bot de trading
Gère la création, la modification et l'annulation des ordres
"""
import time
import uuid
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

from core.api_connector import BinanceConnector
from core.position_tracker import PositionTracker
from config.trading_params import (
    STOP_LOSS_PERCENT, 
    TAKE_PROFIT_PERCENT,
    TRAILING_STOP_ACTIVATION,
    TRAILING_STOP_STEP,
    LEVERAGE
)
from utils.logger import setup_logger
from utils.network_utils import retry_with_backoff

logger = setup_logger("order_manager")

class OrderManager:
    """
    Gère les ordres de trading (entrée, sortie, stop-loss, take-profit)
    """
    def __init__(self, api_connector: BinanceConnector, position_tracker: PositionTracker):
        self.api = api_connector
        self.position_tracker = position_tracker
        self.leverage_set = set()  # Paires pour lesquelles le levier a déjà été défini
        self.pending_orders = {}  # order_id: order details
    
    def set_leverage_if_needed(self, symbol: str) -> bool:
        """
        Définit l'effet de levier pour un symbole si ce n'est pas déjà fait
        
        Args:
            symbol: Paire de trading
            
        Returns:
            True si l'opération a réussi, False sinon
        """
        if symbol in self.leverage_set:
            return True
            
        try:
            # Définir l'effet de levier
            response = self.api.set_leverage(symbol, LEVERAGE)
            
            # Vérifier si l'opération a réussi
            if "leverage" in response:
                self.leverage_set.add(symbol)
                logger.info(f"Levier défini à {LEVERAGE}x pour {symbol}")
                return True
            else:
                logger.error(f"Échec de la définition du levier pour {symbol}: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de la définition du levier pour {symbol}: {str(e)}")
            return False
    
    def place_entry_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None,
                       stop_loss_price: Optional[float] = None, take_profit_price: Optional[float] = None,
                       trailing_activation: float = 0.05, trailing_step: float = 0.02) -> Dict:
        """
        Place un ordre d'entrée avec stop-loss et take-profit
        
        Args:
            symbol: Paire de trading
            side: Direction (BUY/SELL)
            quantity: Quantité à acheter/vendre
            price: Prix d'entrée (None pour un ordre au marché)
            stop_loss_price: Prix du stop-loss (calculé automatiquement si None)
            take_profit_price: Prix du take-profit (calculé automatiquement si None)
            trailing_activation: Seuil d'activation du stop suiveur
            trailing_step: Taille du pas du stop suiveur
            
        Returns:
            Résultat de l'opération
        """
        # Vérifier et définir l'effet de levier si nécessaire
        if not self.set_leverage_if_needed(symbol):
            return {"success": False, "message": "Échec de la définition du levier"}
        
        # Générer un identifiant unique pour l'ordre
        client_order_id = f"bot_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        
        try:
            # Déterminer le type d'ordre (MARKET ou LIMIT)
            order_type = "MARKET" if price is None else "LIMIT"
            
            # Paramètres de l'ordre
            order_params = {
                "newClientOrderId": client_order_id
            }
            
            # Placer l'ordre d'entrée
            if order_type == "MARKET":
                entry_order = self.api.create_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    **order_params
                )
            else:
                entry_order = self.api.create_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    time_in_force="GTC",
                    **order_params
                )
            
            logger.info(f"Ordre d'entrée placé pour {symbol}: {entry_order}")
            
            # Traiter la réponse de l'ordre
            if "orderId" in entry_order:
                # Calculer les prix de stop-loss et take-profit si non fournis
                entry_price = float(entry_order.get("price") or entry_order.get("fills", [{}])[0].get("price", 0))
                
                if stop_loss_price is None:
                    if side == "BUY":
                        stop_loss_price = entry_price * (1 - STOP_LOSS_PERCENT/100)
                    else:
                        stop_loss_price = entry_price * (1 + STOP_LOSS_PERCENT/100)
                
                if take_profit_price is None:
                    if side == "BUY":
                        take_profit_price = entry_price * (1 + TAKE_PROFIT_PERCENT/100)
                    else:
                        take_profit_price = entry_price * (1 - TAKE_PROFIT_PERCENT/100)
                
                # Placer les ordres de stop-loss et take-profit
                stop_loss_side = "SELL" if side == "BUY" else "BUY"
                
                # Ordre stop-loss
                stop_loss_order = self.api.create_order(
                    symbol=symbol,
                    side=stop_loss_side,
                    order_type="STOP_LOSS_LIMIT" if price else "STOP_MARKET",
                    quantity=quantity,
                    price=stop_loss_price if price else None,
                    stop_price=stop_loss_price,
                    time_in_force="GTC",
                    newClientOrderId=f"sl_{client_order_id}"
                )
                
                logger.info(f"Ordre stop-loss placé pour {symbol}: {stop_loss_order}")
                
                # Ordre take-profit
                take_profit_order = self.api.create_order(
                    symbol=symbol,
                    side=stop_loss_side,
                    order_type="LIMIT",
                    quantity=quantity,
                    price=take_profit_price,
                    time_in_force="GTC",
                    newClientOrderId=f"tp_{client_order_id}"
                )
                
                logger.info(f"Ordre take-profit placé pour {symbol}: {take_profit_order}")
                
                # Enregistrer la position dans le tracker
                position = {
                    "id": client_order_id,
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price,
                    "entry_order_id": entry_order["orderId"],
                    "stop_loss_order_id": stop_loss_order.get("orderId"),
                    "take_profit_order_id": take_profit_order.get("orderId"),
                    "entry_time": datetime.now(),
                    "trailing_stop_activated": False,
                    "trailing_activation_threshold": trailing_activation,
                    "trailing_step_size": trailing_step,
                    "highest_price": entry_price if side == "BUY" else float('inf'),
                    "lowest_price": entry_price if side == "SELL" else float('-inf')
                }
                
                self.position_tracker.add_position(position)
                
                return {
                    "success": True,
                    "position_id": client_order_id,
                    "entry_price": entry_price,
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price
                }
            else:
                logger.error(f"Échec de l'ordre d'entrée pour {symbol}: {entry_order}")
                return {"success": False, "message": "Échec de l'ordre d'entrée", "response": entry_order}
                
        except Exception as e:
            logger.error(f"Erreur lors du placement de l'ordre pour {symbol}: {str(e)}")
            return {"success": False, "message": str(e)}
    
    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def update_trailing_stop(self, position_id, current_price):
        """
        Update trailing stop for a position with retry logic
        
        Args:
            position_id: ID of the position
            current_price: Current market price
        
        Returns:
            Boolean indicating success
        """
        try:
            # Get position details
            position = self.position_tracker.get_position(position_id)
            if not position:
                logger.warning(f"Position {position_id} not found for trailing stop update")
                return False
            
            # Get current stop parameters
            current_stop_price = position.get("stop_loss_price")
            if not current_stop_price:
                logger.warning(f"Position {position_id} has no stop loss price set")
                return False
            
            # Get position-specific trailing settings
            activation_threshold = position.get("trailing_activation_threshold", 0.05)  # Default 5%
            step_size = position.get("trailing_step_size", 0.02)  # Default 2%
            entry_price = position.get("entry_price", current_price)
            side = position.get("side", "BUY")
            
            # Check if trailing stop is already activated
            is_activated = position.get("trailing_stop_activated", False)
            
            # For long positions
            if side == "BUY":
                # First check if trailing stop should be activated
                if not is_activated:
                    # Activate if price has moved up by the activation threshold
                    profit_pct = (current_price - entry_price) / entry_price
                    if profit_pct >= activation_threshold:
                        # Calculate initial trailing stop level - lock in portion of profit
                        new_stop_price = entry_price + (current_price - entry_price) * 0.5  # Lock in 50% of current profit
                        
                        # Make sure the new stop is higher than current stop
                        if new_stop_price > current_stop_price:
                            logger.info(f"Activating trailing stop for position {position_id} at {new_stop_price}")
                            success = self._update_stop_loss_order(position_id, new_stop_price)
                            if success:
                                # Update position data
                                self.position_tracker.update_position(position_id, {
                                    "stop_loss_price": new_stop_price,
                                    "trailing_stop_activated": True,
                                    "highest_price": current_price
                                })
                                return True
                        return False
                else:
                    # Trailing stop already activated - update highest price seen
                    highest_price = position.get("highest_price", current_price)
                    
                    # Only update if current price is higher than highest price seen
                    if current_price > highest_price:
                        # Update highest price
                        highest_price = current_price
                        
                        # Calculate new stop price based on step_size
                        price_diff = current_price * (1 - step_size)
                        
                        # Only move stop up if new stop is higher than current stop
                        if price_diff > current_stop_price:
                            new_stop_price = price_diff
                            logger.info(f"Updating trailing stop for position {position_id} from {current_stop_price} to {new_stop_price}")
                            
                            # Update stop loss order
                            success = self._update_stop_loss_order(position_id, new_stop_price)
                            if success:
                                # Update position data
                                self.position_tracker.update_position(position_id, {
                                    "stop_loss_price": new_stop_price,
                                    "highest_price": highest_price
                                })
                            return success
                
            # For short positions
            else:  # side == "SELL"
                # First check if trailing stop should be activated
                if not is_activated:
                    # Activate if price has moved down by the activation threshold
                    profit_pct = (entry_price - current_price) / entry_price
                    if profit_pct >= activation_threshold:
                        # Calculate initial trailing stop level - lock in portion of profit
                        new_stop_price = entry_price - (entry_price - current_price) * 0.5  # Lock in 50% of current profit
                        
                        # Make sure the new stop is lower than current stop
                        if new_stop_price < current_stop_price:
                            logger.info(f"Activating trailing stop for position {position_id} at {new_stop_price}")
                            success = self._update_stop_loss_order(position_id, new_stop_price)
                            if success:
                                # Update position data
                                self.position_tracker.update_position(position_id, {
                                    "stop_loss_price": new_stop_price,
                                    "trailing_stop_activated": True,
                                    "lowest_price": current_price
                                })
                                return True
                        return False
                else:
                    # Trailing stop already activated - update lowest price seen
                    lowest_price = position.get("lowest_price", current_price)
                    
                    # Only update if current price is lower than lowest price seen
                    if current_price < lowest_price:
                        # Update lowest price
                        lowest_price = current_price
                        
                        # Calculate new stop price based on step_size
                        price_diff = current_price * (1 + step_size)
                        
                        # Only move stop down if new stop is lower than current stop
                        if price_diff < current_stop_price:
                            new_stop_price = price_diff
                            logger.info(f"Updating trailing stop for position {position_id} from {current_stop_price} to {new_stop_price}")
                            
                            # Update stop loss order
                            success = self._update_stop_loss_order(position_id, new_stop_price)
                            if success:
                                # Update position data
                                self.position_tracker.update_position(position_id, {
                                    "stop_loss_price": new_stop_price,
                                    "lowest_price": lowest_price
                                })
                            return success
            
            # No update needed
            return True
            
        except Exception as e:
            logger.error(f"Error updating trailing stop: {str(e)}")
            # Re-raise to trigger retry logic
            raise

    def _update_stop_loss_order(self, position_id, new_price):
        """
        Updates a stop loss order with the exchange
        
        Args:
            position_id: Position ID
            new_price: New stop price
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get position details
            position = self.position_tracker.get_position(position_id)
            if not position:
                logger.warning(f"Position {position_id} not found")
                return False
                
            # Get stop loss order ID
            stop_order_id = position.get("stop_loss_order_id")
            if not stop_order_id:
                logger.warning(f"No stop loss order ID for position {position_id}")
                return False
                
            symbol = position.get("symbol")
                
            # Cancel existing stop loss
            cancel_result = self.api.cancel_order(
                symbol=symbol,
                order_id=stop_order_id
            )
            
            if not cancel_result:
                logger.error(f"Failed to cancel existing stop loss order {stop_order_id}")
                return False
                
            # Create new stop loss order
            quantity = position.get("quantity")
            side = "SELL" if position.get("side") == "BUY" else "BUY"
            
            new_order = self.api.create_order(
                symbol=symbol,
                side=side,
                order_type="STOP_LOSS",
                time_in_force="GTC",
                quantity=quantity,
                stop_price=new_price,
                price=new_price  # For stop limit orders
            )
            
            if new_order and "orderId" in new_order:
                # Update position with new stop loss order ID
                self.position_tracker.update_position(position_id, {
                    "stop_loss_order_id": new_order["orderId"],
                    "stop_loss_price": new_price
                })
                return True
                
            logger.error(f"Failed to create new stop loss order for position {position_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error updating stop loss order: {str(e)}")
            return False

    def close_position(self, symbol: str, position_id: str) -> Dict:
        """
        Ferme une position manuellement
        
        Args:
            symbol: Paire de trading
            position_id: ID de la position à fermer
            
        Returns:
            Résultat de l'opération
        """
        position = self.position_tracker.get_position(position_id)
        
        if not position:
            return {"success": False, "message": f"Position {position_id} non trouvée"}
        
        try:
            # Annuler les ordres de stop-loss et take-profit
            orders_to_cancel = [
                position.get("stop_loss_order_id"),
                position.get("take_profit_order_id")
            ]
            
            for order_id in orders_to_cancel:
                if order_id:
                    try:
                        self.api.cancel_order(symbol=symbol, order_id=order_id)
                    except Exception as e:
                        logger.warning(f"Erreur lors de l'annulation de l'ordre {order_id}: {str(e)}")
            
            # Créer un ordre de fermeture au marché
            close_side = "SELL" if position["side"] == "BUY" else "BUY"
            
            close_order = self.api.create_order(
                symbol=symbol,
                side=close_side,
                order_type="MARKET",
                quantity=position["quantity"],
                newClientOrderId=f"close_{position_id}"
            )
            
            logger.info(f"Position {position_id} fermée: {close_order}")
            
            # Marquer la position comme fermée dans le tracker
            self.position_tracker.close_position(position_id, close_order)
            
            return {"success": True, "close_order": close_order}
            
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de la position {position_id}: {str(e)}")
            return {"success": False, "message": str(e)}

    def create_order(self, symbol: str, order_type: str, side: str, quantity: float, price: float = None, params: dict = None) -> dict:
        """
        Place an order via the exchange client with retries.
        
        Args:
            symbol: Trading pair
            order_type: "market" or "limit"
            side: "BUY" or "SELL"
            quantity: Amount to trade
            price: Price for limit orders (ignored in market orders)
            params: Additional parameters
            
        Returns:
            Order response dictionary.
        """
        params = params or {}
        attempts = 3
        
        for attempt in range(1, attempts + 1):
            try:
                if order_type.lower() == "market":
                    response = self.api.create_order(symbol, order_type, side, quantity, params=params)
                else:
                    response = self.api.create_order(symbol, order_type, side, quantity, price, params=params)
                    
                logger.info(f"Order placed: {response}")
                
                order_id = response.get("id")
                if order_id:
                    self.pending_orders[order_id] = response
                return response
            except Exception as e:
                logger.error(f"Error placing order (attempt {attempt}): {e}")
                time.sleep(1)
                
        return {"error": "Order placement failed after retries"}

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        """
        Cancel an existing order.
        """
        try:
            response = self.api.cancel_order(symbol, order_id)
            logger.info(f"Order cancelled: {order_id}")
            
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
                
            return response
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {"error": str(e)}

    def check_order_status(self, order_id: str) -> dict:
        """
        Check the status of an order.
        """
        try:
            response = self.api.get_order_status(order_id)
            logger.info(f"Order status for {order_id}: {response}")
            return response
        except Exception as e:
            logger.error(f"Error checking status for order {order_id}: {e}")
            return {"error": str(e)}

    def retry_pending_orders(self) -> None:
        """
        Retry orders that have not been filled yet.
        """
        for order_id, order in list(self.pending_orders.items()):
            status = self.check_order_status(order_id)
            if status.get("status") in ["canceled", "rejected"]:
                logger.info(f"Pending order {order_id} failed, removing from queue.")
                del self.pending_orders[order_id]