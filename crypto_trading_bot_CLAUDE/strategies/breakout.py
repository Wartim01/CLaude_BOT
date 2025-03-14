from strategies.strategy_base import StrategyBase
import pandas as pd

class BreakoutStrategy(StrategyBase):
    """
    Stratégie de cassure détectant la sortie de range ou le franchissement de niveaux clés.
    """
    def __init__(self, breakout_period: int = 20, volume_threshold: float = 1.5):
        # ...existing initialization code...
        self.breakout_period = breakout_period
        self.volume_threshold = volume_threshold  # Facteur pour contraste de volume
        
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> dict:
        """
        Génère un signal basé sur la cassure des bandes de Bollinger ou le franchissement d'un nouveau plus haut.
        
        Args:
            symbol: Le symbole du marché.
            data: DataFrame contenant 'close', 'volume', 'Bollinger_Upper', ...
            
        Returns:
            Dictionnaire contenant 'signal' (BUY/SELL/NEUTRAL) et 'confidence'.
        """
        # Vérifier la présence des colonnes essentielles
        if not all(col in data.columns for col in ["close", "Bollinger_Upper", "volume"]):
            return {"signal": "NEUTRAL", "confidence": 0.5}
        
        last_close = data["close"].iloc[-1]
        upper_band = data["Bollinger_Upper"].iloc[-1]
        current_volume = data["volume"].iloc[-1]
        
        # Calculer le plus haut et volume moyen sur breakout_period
        recent_high = data["close"].iloc[-self.breakout_period:].max()
        average_volume = data["volume"].iloc[-self.breakout_period:].mean()
        
        signal = "NEUTRAL"
        confidence = 0.5
        
        # Critère 1 : Cassure de la bande supérieure avec volume élevé
        if last_close > upper_band and current_volume > self.volume_threshold * average_volume:
            signal = "BUY"
            confidence = 0.8
        # Critère 2 : Nouvel plus haut sur la période avec confirmation de volume
        elif last_close > recent_high and current_volume > average_volume:
            signal = "BUY"
            confidence = 0.7
        
        return {"signal": signal, "confidence": confidence}
