# ğŸ¤– Bot de Trading Crypto avec IA Auto-Adaptative

Un bot de trading crypto sophistiquÃ© basÃ© sur une IA auto-adaptative, conÃ§u pour identifier et trader les rebonds techniques aprÃ¨s des baisses de prix.

## ğŸ“‹ FonctionnalitÃ©s Principales

- **StratÃ©gie de rebond technique** optimisÃ©e pour capturer les corrections haussiÃ¨res
- **IA auto-amÃ©liorante** qui analyse ses propres performances et ajuste ses paramÃ¨tres
- **SystÃ¨me de gestion des risques** avec stop-loss et take-profit adaptatifs
- **Trailing stop dynamique** qui se resserre avec l'augmentation du profit
- **Module de backtest** complet pour tester la stratÃ©gie sur des donnÃ©es historiques
- **Visualisations** des performances et des trades individuels
- **CompatibilitÃ© avec Binance** (testnet et production)

## ğŸ› ï¸� Installation

### PrÃ©requis
- Python 3.8 ou supÃ©rieur
- Compte Binance (standard ou testnet)
- ClÃ©s API Binance avec permissions de trading

### Installation Automatique

Utilisez le script d'installation automatique qui vous guidera Ã  travers le processus :

```bash
python install.py
```

Le script effectuera les actions suivantes :
1. VÃ©rification de la version de Python
2. Installation des dÃ©pendances requises
3. CrÃ©ation des rÃ©pertoires nÃ©cessaires
4. Configuration des paramÃ¨tres Binance
5. Personnalisation des paramÃ¨tres de trading (optionnel)
6. ExÃ©cution de tests de connexion

### Installation Manuelle

Si vous prÃ©fÃ©rez une installation manuelle, suivez ces Ã©tapes :

1. Clonez le dÃ©pÃ´t :
```bash
git clone https://github.com/votre-username/crypto-trading-bot.git
cd crypto-trading-bot
```

2. Installez les dÃ©pendances :
```bash
pip install -r requirements.txt
```

3. CrÃ©ez un fichier `.env` Ã  la racine du projet avec le contenu suivant :
```
BINANCE_API_KEY=votre_clÃ©_api
BINANCE_API_SECRET=votre_clÃ©_secrÃ¨te
USE_TESTNET=True
```

4. CrÃ©ez les rÃ©pertoires nÃ©cessaires :
```bash
mkdir -p data/market_data data/trade_logs data/performance logs
```

## ğŸš€ Utilisation

### Mode Test (Dry Run)

Pour exÃ©cuter le bot sans passer d'ordres rÃ©els (recommandÃ© pour les tests) :

```bash
python main.py --dry-run
```

### Mode Production

Pour exÃ©cuter le bot avec trading rÃ©el :

```bash
python main.py
```

### Options de Configuration

Vous pouvez personnaliser diffÃ©rents aspects du bot en modifiant les fichiers de configuration :

- `config/config.py` : Configuration globale du bot
- `config/trading_params.py` : ParamÃ¨tres spÃ©cifiques Ã  la stratÃ©gie de trading

### RÃ©glage de la StratÃ©gie

Les paramÃ¨tres que vous pouvez ajuster dans `trading_params.py` incluent :

```python
# ParamÃ¨tres de gestion des risques
RISK_PER_TRADE_PERCENT = 7.5  # Pourcentage du capital risquÃ© par trade (5-10%)
STOP_LOSS_PERCENT = 4.0       # Pourcentage de stop-loss (3-5%)
TAKE_PROFIT_PERCENT = 6.0     # Pourcentage de take-profit (5-7%)
LEVERAGE = 3                  # Effet de levier (jusqu'Ã  5x)

# Seuils techniques
RSI_OVERSOLD = 30             # Seuil de survente du RSI
MINIMUM_SCORE_TO_TRADE = 70   # Score minimum pour entrer en position (0-100)
```

## ğŸ§ª Backtesting

Le bot inclut un systÃ¨me de backtest complet pour Ã©valuer la stratÃ©gie sur des donnÃ©es historiques.

### ExÃ©cution d'un Backtest

```bash
python backtest.py --symbol BTCUSDT --timeframe 15m --start 2023-01-01 --end 2023-06-30 --capital 200
python download_data.py --symbol BTCUSDT --interval 15m --start 2023-01-01 --end 2024-12-30
```

### Options de Backtest

- `--symbol` : Paire de trading (ex: BTCUSDT)
- `--timeframe` : Intervalle de temps (ex: 15m, 1h, 4h)
- `--start` : Date de dÃ©but (YYYY-MM-DD)
- `--end` : Date de fin (YYYY-MM-DD)
- `--capital` : Capital initial en USDT
- `--strategy` : StratÃ©gie Ã  tester (par dÃ©faut: technical_bounce)

### Visualisation des RÃ©sultats

Les rÃ©sultats du backtest sont sauvegardÃ©s dans le rÃ©pertoire `data/backtest_results` et incluent :
- Fichier JSON avec les statistiques complÃ¨tes
- Graphique de la courbe d'Ã©quitÃ©
- Graphique de la distribution des profits/pertes

## ğŸ“Š Performances et Analyse

Le bot inclut des outils d'analyse pour Ã©valuer ses performances :

### Analyse des Trades

Pour gÃ©nÃ©rer une analyse des trades rÃ©cents :

```python
from ai.trade_analyzer import TradeAnalyzer
analyzer = TradeAnalyzer(scoring_engine, position_tracker)
report = analyzer.analyze_recent_trades(days=30)
```

### Visualisation des Performances

Pour gÃ©nÃ©rer des visualisations de performance :

```python
from utils.visualizer import TradeVisualizer
visualizer = TradeVisualizer(position_tracker)
equity_curve = visualizer.plot_equity_curve(days=30)
trade_analysis = visualizer.plot_trade_analysis(days=30)
```

## ğŸ§  SystÃ¨me d'IA Auto-Adaptative

L'IA du bot s'amÃ©liore automatiquement en analysant ses performances passÃ©es.

### Composants de l'IA

- **Moteur de Scoring** : Ã‰value les opportunitÃ©s de trading selon multiples critÃ¨res
- **Analyseur de Trades** : Identifie les patterns de succÃ¨s et d'Ã©chec
- **Optimiseur de ParamÃ¨tres** : Ajuste les paramÃ¨tres en fonction des rÃ©sultats
- **Moteur de Raisonnement** : GÃ©nÃ¨re des explications textuelles pour les dÃ©cisions

### Cycle d'Apprentissage

1. L'IA analyse les opportunitÃ©s de trading et attribue un score
2. Le bot exÃ©cute les trades avec un score suffisant
3. L'IA analyse les rÃ©sultats des trades fermÃ©s
4. Les poids des diffÃ©rents facteurs sont ajustÃ©s en consÃ©quence
5. Les paramÃ¨tres de la stratÃ©gie sont optimisÃ©s pÃ©riodiquement

## ğŸ“� Journal des Trades

Chaque trade est enregistrÃ© avec des dÃ©tails complets, incluant :

- OpportunitÃ© initiale avec score et raisonnement
- Conditions de marchÃ© lors de l'entrÃ©e
- Performance rÃ©elle (PnL)
- Auto-critique de l'IA

Les journaux sont stockÃ©s au format JSON dans `data/trade_logs/` et peuvent Ãªtre analysÃ©s pour comprendre les dÃ©cisions du bot.

## âš ï¸� Avertissements et Risques

- **Risque de Perte** : Le trading de crypto-monnaies comporte des risques significatifs. N'investissez que ce que vous pouvez vous permettre de perdre.
- **Effet de Levier** : L'utilisation de l'effet de levier amplifie les profits mais aussi les pertes.
- **Tests** : Commencez toujours en mode test ou avec de petits montants avant d'engager des sommes importantes.
- **Maintenance** : Surveillez rÃ©guliÃ¨rement le bot et son fonctionnement.

## ğŸ“œ License

Ce projet est distribuÃ© sous licence MIT. Voir le fichier `LICENSE` pour plus de dÃ©tails.

## ğŸ¤� Contribution

Les contributions sont les bienvenues! N'hÃ©sitez pas Ã  soumettre des pull requests ou Ã  signaler des problÃ¨mes.

## ğŸ“§ Contact

Pour toute question ou suggestion, veuillez me contacter Ã  [votre-email@exemple.com].