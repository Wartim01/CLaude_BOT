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

logger = setup_logger("order_manager")

class OrderManager:
    """
    Gère les ordres de trading (entrée, sortie, stop-loss, take-profit)
    """
    def __init__(self, api_connector: BinanceConnector, position_tracker: PositionTracker):
        self.api = api_connector
        self.position_tracker = position_tracker
        self.leverage_set = set()  # Paires pour lesquelles le levier a déjà été défini
    
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
                       stop_loss_price: Optional[float] = None, take_profit_price: Optional[float] = None) -> Dict:
        """
        Place un ordre d'entrée avec stop-loss et take-profit
        
        Args:
            symbol: Paire de trading
            side: Direction (BUY/SELL)
            quantity: Quantité à acheter/vendre
            price: Prix d'entrée (None pour un ordre au marché)
            stop_loss_price: Prix du stop-loss (calculé automatiquement si None)
            take_profit_price: Prix du take-profit (calculé automatiquement si None)
            
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
    
    def update_trailing_stop(self, symbol: str, position: Dict, current_price: float) -> Dict:
        """
        Met à jour le trailing stop d'une position si nécessaire
        
        Args:
            symbol: Paire de trading
            position: Position à mettre à jour
            current_price: Prix actuel
            
        Returns:
            Résultat de l'opération
        """
        side = position["side"]
        entry_price = position["entry_price"]
        stop_loss_price = position["stop_loss_price"]
        position_id = position["id"]
        
        # Vérifier si le trailing stop doit être activé ou mis à jour
        if side == "BUY":
            # Pour les positions longues
            profit_percent = (current_price - entry_price) / entry_price * 100
            
            # Mettre à jour le prix le plus haut observé
            if current_price > position["highest_price"]:
                position["highest_price"] = current_price
                self.position_tracker.update_position(position_id, position)
            
            # Activation du trailing stop
            if (not position["trailing_stop_activated"] and 
                profit_percent >= TRAILING_STOP_ACTIVATION):
                
                position["trailing_stop_activated"] = True
                self.position_tracker.update_position(position_id, position)
                logger.info(f"Trailing stop activé pour {symbol} (ID: {position_id})")
            
            # Mise à jour du trailing stop
            if position["trailing_stop_activated"]:
                new_stop_loss = position["highest_price"] * (1 - TRAILING_STOP_STEP/100)
                
                if new_stop_loss > stop_loss_price:
                    try:
                        # Annuler l'ancien ordre stop-loss
                        self.api.cancel_order(
                            symbol=symbol,
                            order_id=position["stop_loss_order_id"]
                        )
                        
                        # Créer un nouvel ordre stop-loss
                        new_stop_order = self.api.create_order(
                            symbol=symbol,
                            side="SELL",
                            order_type="STOP_MARKET",
                            quantity=position["quantity"],
                            stop_price=new_stop_loss,
                            newClientOrderId=f"sl_trail_{position_id}"
                        )
                        
                        # Mettre à jour la position
                        position["stop_loss_price"] = new_stop_loss
                        position["stop_loss_order_id"] = new_stop_order["orderId"]
                        self.position_tracker.update_position(position_id, position)
                        
                        logger.info(f"Trailing stop mis à jour pour {symbol} (ID: {position_id}) à {new_stop_loss}")
                        return {"success": True, "new_stop_loss": new_stop_loss}
                    
                    except Exception as e:
                        logger.error(f"Erreur lors de la mise à jour du trailing stop pour {symbol} (ID: {position_id}): {str(e)}")
                        return {"success": False, "message": str(e)}
        
        elif side == "SELL":
            # Pour les positions courtes
            profit_percent = (entry_price - current_price) / entry_price * 100
            
            # Mettre à jour le prix le plus bas observé
            if current_price < position["lowest_price"]:
                position["lowest_price"] = current_price
                self.position_tracker.update_position(position_id, position)
            
            # Activation du trailing stop
            if (not position["trailing_stop_activated"] and 
                profit_percent >= TRAILING_STOP_ACTIVATION):
                
                position["trailing_stop_activated"] = True
                self.position_tracker.update_position(position_id, position)
                logger.info(f"Trailing stop activé pour {symbol} (ID: {position_id})")
            
            # Mise à jour du trailing stop
            if position["trailing_stop_activated"]:
                new_stop_loss = position["lowest_price"] * (1 + TRAILING_STOP_STEP/100)
                
                if new_stop_loss < stop_loss_price:
                    try:
                        # Annuler l'ancien ordre stop-loss
                        self.api.cancel_order(
                            symbol=symbol,
                            order_id=position["stop_loss_order_id"]
                        )
                        
                        # Créer un nouvel ordre stop-loss
                        new_stop_order = self.api.create_order(
                            symbol=symbol,
                            side="BUY",
                            order_type="STOP_MARKET",
                            quantity=position["quantity"],
                            stop_price=new_stop_loss,
                            newClientOrderId=f"sl_trail_{position_id}"
                        )
                        
                        # Mettre à jour la position
                        position["stop_loss_price"] = new_stop_loss
                        position["stop_loss_order_id"] = new_stop_order["orderId"]
                        self.position_tracker.update_position(position_id, position)
                        
                        logger.info(f"Trailing stop mis à jour pour {symbol} (ID: {position_id}) à {new_stop_loss}")
                        return {"success": True, "new_stop_loss": new_stop_loss}
                    
                    except Exception as e:
                        logger.error(f"Erreur lors de la mise à jour du trailing stop pour {symbol} (ID: {position_id}): {str(e)}")
                        return {"success": False, "message": str(e)}
        
        return {"success": True, "message": "Aucune mise à jour nécessaire"}
    
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