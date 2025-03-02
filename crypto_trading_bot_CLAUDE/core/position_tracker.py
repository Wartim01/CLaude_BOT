# core/position_tracker.py
"""
Suivi des positions ouvertes et fermées
"""
import json
import os
from typing import Dict, List, Optional, Union
from datetime import datetime

from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("position_tracker")

class PositionTracker:
    """
    Gère le suivi des positions ouvertes et fermées
    """
    def __init__(self):
        self.open_positions = {}  # {position_id: position_data}
        self.closed_positions = []  # Liste des positions fermées
        
        self.positions_file = os.path.join(DATA_DIR, "positions.json")
        self.load_positions()
    
    def load_positions(self) -> None:
        """
        Charge les positions depuis le fichier de sauvegarde
        """
        if os.path.exists(self.positions_file):
            try:
                with open(self.positions_file, 'r') as f:
                    data = json.load(f)
                    self.open_positions = data.get("open_positions", {})
                    self.closed_positions = data.get("closed_positions", [])
                logger.info(f"Positions chargées: {len(self.open_positions)} ouvertes, {len(self.closed_positions)} fermées")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des positions: {str(e)}")
    
    def save_positions(self) -> None:
        """
        Sauvegarde les positions dans un fichier
        """
        try:
            data = {
                "open_positions": self.open_positions,
                "closed_positions": self.closed_positions
            }
            
            with open(self.positions_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.debug("Positions sauvegardées")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des positions: {str(e)}")
    
    def add_position(self, position: Dict) -> None:
        """
        Ajoute une nouvelle position
        
        Args:
            position: Données de la position
        """
        position_id = position["id"]
        self.open_positions[position_id] = position
        logger.info(f"Position ajoutée: {position_id} ({position['symbol']})")
        self.save_positions()
    
    def update_position(self, position_id: str, position_data: Dict) -> bool:
        """
        Met à jour une position existante
        
        Args:
            position_id: ID de la position
            position_data: Nouvelles données de la position
            
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        if position_id in self.open_positions:
            self.open_positions[position_id] = position_data
            logger.debug(f"Position mise à jour: {position_id}")
            self.save_positions()
            return True
        else:
            logger.warning(f"Tentative de mise à jour d'une position inexistante: {position_id}")
            return False
    
    def close_position(self, position_id: str, close_data: Dict) -> bool:
        """
        Marque une position comme fermée
        
        Args:
            position_id: ID de la position
            close_data: Données de fermeture
            
        Returns:
            True si la fermeture a réussi, False sinon
        """
        if position_id in self.open_positions:
            position = self.open_positions.pop(position_id)
            
            # Ajouter les informations de fermeture
            position["close_time"] = datetime.now()
            position["close_data"] = close_data
            
            # Calculer le P&L
            if "fills" in close_data:
                close_price = float(close_data["fills"][0]["price"])
                entry_price = float(position["entry_price"])
                quantity = float(position["quantity"])
                
                if position["side"] == "BUY":
                    pnl_percent = (close_price - entry_price) / entry_price * 100
                    pnl_absolute = (close_price - entry_price) * quantity
                else:
                    pnl_percent = (entry_price - close_price) / entry_price * 100
                    pnl_absolute = (entry_price - close_price) * quantity
                
                position["pnl_percent"] = pnl_percent
                position["pnl_absolute"] = pnl_absolute
                
                logger.info(f"Position {position_id} fermée: {pnl_percent:.2f}% ({pnl_absolute:.2f} USDT)")
            else:
                logger.info(f"Position {position_id} fermée (P&L non calculable)")
            
            # Ajouter à la liste des positions fermées
            self.closed_positions.append(position)
            self.save_positions()
            return True
        else:
            logger.warning(f"Tentative de fermeture d'une position inexistante: {position_id}")
            return False
    
    def get_position(self, position_id: str) -> Optional[Dict]:
        """
        Récupère les données d'une position
        
        Args:
            position_id: ID de la position
            
        Returns:
            Données de la position, ou None si non trouvée
        """
        return self.open_positions.get(position_id)
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère les positions ouvertes pour un symbole donné
        
        Args:
            symbol: Paire de trading (optionnel)
            
        Returns:
            Liste des positions ouvertes
        """
        if symbol:
            return [p for p in self.open_positions.values() if p["symbol"] == symbol]
        else:
            return list(self.open_positions.values())
    
    def get_all_open_positions(self) -> Dict[str, List[Dict]]:
        """
        Récupère toutes les positions ouvertes, groupées par symbole
        
        Returns:
            Dictionnaire {symbole: [positions]}
        """
        positions_by_symbol = {}
        
        for position in self.open_positions.values():
            symbol = position["symbol"]
            if symbol not in positions_by_symbol:
                positions_by_symbol[symbol] = []
            positions_by_symbol[symbol].append(position)
        
        return positions_by_symbol
    
    def get_closed_positions(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Récupère les positions fermées
        
        Args:
            symbol: Paire de trading (optionnel)
            limit: Nombre maximum de positions à récupérer
            
        Returns:
            Liste des positions fermées
        """
        if symbol:
            filtered = [p for p in self.closed_positions if p["symbol"] == symbol]
        else:
            filtered = self.closed_positions
        
        # Tri par date de fermeture (du plus récent au plus ancien)
        sorted_positions = sorted(
            filtered,
            key=lambda p: p.get("close_time", datetime.min),
            reverse=True
        )
        
        return sorted_positions[:limit]
    
    def get_position_count(self, symbol: Optional[str] = None) -> int:
        """
        Compte le nombre de positions ouvertes
        
        Args:
            symbol: Paire de trading (optionnel)
            
        Returns:
            Nombre de positions ouvertes
        """
        if symbol:
            return len([p for p in self.open_positions.values() if p["symbol"] == symbol])
        else:
            return len(self.open_positions)
    
    def get_daily_trades_count(self, symbol: Optional[str] = None) -> int:
        """
        Compte le nombre de trades effectués aujourd'hui
        
        Args:
            symbol: Paire de trading (optionnel)
            
        Returns:
            Nombre de trades aujourd'hui
        """
        today = datetime.now().date()
        
        # Compter les positions fermées aujourd'hui
        closed_today = [
            p for p in self.closed_positions
            if p.get("close_time") and p.get("close_time").date() == today
            and (symbol is None or p["symbol"] == symbol)
        ]
        
        # Compter les positions ouvertes aujourd'hui
        opened_today = [
            p for p in self.open_positions.values()
            if p.get("entry_time") and p.get("entry_time").date() == today
            and (symbol is None or p["symbol"] == symbol)
        ]
        
        return len(closed_today) + len(opened_today)