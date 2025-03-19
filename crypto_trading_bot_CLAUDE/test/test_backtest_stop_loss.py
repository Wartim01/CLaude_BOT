import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from utils.logger import setup_logger  # ...existing code...
from ai.models.backtest_engine import BacktestEngine  # ...existing code...

logger = setup_logger("test_backtest_stop_loss")

class TestBacktestStopLoss(unittest.TestCase):
    def setUp(self):
        # Créer un DataFrame synthétique avec 100 périodes (15 minutes chacune)
        dates = pd.date_range(start="2021-01-01", periods=100, freq="15T")
        # Pour les 60 premières périodes, le prix est stable à 100;
        # après, le prix chute brutalement à 90, ce qui doit déclencher le stop loss
        open_prices = np.concatenate([np.full(60, 100), np.full(40, 90)])
        high_prices = open_prices * 1.01
        # Pour la partie de chute, le prix minimal descend en dessous du seuil stop loss (95)
        low_prices = np.concatenate([np.full(60, 100), np.full(40, 85)])
        close_prices = np.concatenate([np.full(60, 100), np.full(40, 90)])
        volume = np.random.uniform(100, 500, 100)
        
        self.df = pd.DataFrame({
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume
        }, index=dates)
        
        self.initial_capital = 1000.0
        self.stop_loss_pct = 5.0  # 5% de perte déclenche stop loss
        self.take_profit_pct = 10.0  # Valeur arbitraire
        
        # Instancier le moteur de backtest avec les paramètres requis
        self.engine = BacktestEngine(
            initial_capital=self.initial_capital,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            fee_rate=0.1,      # 0.1%
            slippage_pct=0.05  # 5%
        )
        # Simuler l'ouverture d'une position à la période 60 (index 59) au prix de 100
        self.entry_index = 59
        self.entry_price = self.df.iloc[self.entry_index]['close']
        self.engine.open_position(
            entry_date=self.df.index[self.entry_index],
            entry_price=self.entry_price,
            size=100  # Exemple : achat de 100 unités
        )
    
    def test_stop_loss_trigger(self):
        # Exécuter la simulation sur toutes les périodes
        self.engine.simulate_trading(self.df)
        # Récupérer la liste des trades enregistrés
        trades = self.engine.get_trades()
        # Vérifier qu'au moins un trade possède exit_reason "Stop-Loss"
        stop_loss_trades = [trade for trade in trades if trade.get('exit_reason') == "Stop-Loss"]
        self.assertTrue(len(stop_loss_trades) > 0, "Aucun trade avec exit_reason 'Stop-Loss' n'a été enregistré")
        
        # Vérifier que le capital final est mis à jour correctement
        final_capital = self.engine.get_final_capital()
        # Calcul théorique : si la position est clôturée à un stop loss fixé à 95,
        # alors PnL = (95 - 100) * 100 = -500 et capital final = initial_capital - 500
        expected_exit_price = 95.0
        expected_pnl = (expected_exit_price - self.entry_price) * 100
        expected_final_capital = self.initial_capital + expected_pnl
        # Tolérance fixée en raison des frais et du slippage
        self.assertAlmostEqual(final_capital, expected_final_capital, delta=5.0,
                               msg=f"Le capital final {final_capital} n'est pas conforme à la valeur attendue {expected_final_capital}")

if __name__ == "__main__":
    unittest.main()