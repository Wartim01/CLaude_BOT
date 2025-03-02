# 🤖 Bot de Trading Crypto avec IA Auto-Adaptative

Un bot de trading crypto sophistiqué basé sur une IA auto-adaptative, conçu pour identifier et trader les rebonds techniques après des baisses de prix.

## 📋 Fonctionnalités Principales

- **Stratégie de rebond technique** optimisée pour capturer les corrections haussières
- **IA auto-améliorante** qui analyse ses propres performances et ajuste ses paramètres
- **Système de gestion des risques** avec stop-loss et take-profit adaptatifs
- **Trailing stop dynamique** qui se resserre avec l'augmentation du profit
- **Module de backtest** complet pour tester la stratégie sur des données historiques
- **Visualisations** des performances et des trades individuels
- **Compatibilité avec Binance** (testnet et production)

## 🛠️ Installation

### Prérequis
- Python 3.8 ou supérieur
- Compte Binance (standard ou testnet)
- Clés API Binance avec permissions de trading

### Installation Automatique

Utilisez le script d'installation automatique qui vous guidera à travers le processus :

```bash
python install.py
```

Le script effectuera les actions suivantes :
1. Vérification de la version de Python
2. Installation des dépendances requises
3. Création des répertoires nécessaires
4. Configuration des paramètres Binance
5. Personnalisation des paramètres de trading (optionnel)
6. Exécution de tests de connexion

### Installation Manuelle

Si vous préférez une installation manuelle, suivez ces étapes :

1. Clonez le dépôt :
```bash
git clone https://github.com/votre-username/crypto-trading-bot.git
cd crypto-trading-bot
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Créez un fichier `.env` à la racine du projet avec le contenu suivant :
```
BINANCE_API_KEY=votre_clé_api
BINANCE_API_SECRET=votre_clé_secrète
USE_TESTNET=True
```

4. Créez les répertoires nécessaires :
```bash
mkdir -p data/market_data data/trade_logs data/performance logs
```

## 🚀 Utilisation

### Mode Test (Dry Run)

Pour exécuter le bot sans passer d'ordres réels (recommandé pour les tests) :

```bash
python main.py --dry-run
```

### Mode Production

Pour exécuter le bot avec trading réel :

```bash
python main.py
```

### Options de Configuration

Vous pouvez personnaliser différents aspects du bot en modifiant les fichiers de configuration :

- `config/config.py` : Configuration globale du bot
- `config/trading_params.py` : Paramètres spécifiques à la stratégie de trading

### Réglage de la Stratégie

Les paramètres que vous pouvez ajuster dans `trading_params.py` incluent :

```python
# Paramètres de gestion des risques
RISK_PER_TRADE_PERCENT = 7.5  # Pourcentage du capital risqué par trade (5-10%)
STOP_LOSS_PERCENT = 4.0       # Pourcentage de stop-loss (3-5%)
TAKE_PROFIT_PERCENT = 6.0     # Pourcentage de take-profit (5-7%)
LEVERAGE = 3                  # Effet de levier (jusqu'à 5x)

# Seuils techniques
RSI_OVERSOLD = 30             # Seuil de survente du RSI
MINIMUM_SCORE_TO_TRADE = 70   # Score minimum pour entrer en position (0-100)
```

## 🧪 Backtesting

Le bot inclut un système de backtest complet pour évaluer la stratégie sur des données historiques.

### Exécution d'un Backtest

```bash
python backtest.py --symbol BTCUSDT --timeframe 15m --start 2023-01-01 --end 2023-06-30 --capital 200
python download_data.py --symbol BTCUSDT --interval 15m --start 2023-01-01 --end 2024-12-30
```

### Options de Backtest

- `--symbol` : Paire de trading (ex: BTCUSDT)
- `--timeframe` : Intervalle de temps (ex: 15m, 1h, 4h)
- `--start` : Date de début (YYYY-MM-DD)
- `--end` : Date de fin (YYYY-MM-DD)
- `--capital` : Capital initial en USDT
- `--strategy` : Stratégie à tester (par défaut: technical_bounce)

### Visualisation des Résultats

Les résultats du backtest sont sauvegardés dans le répertoire `data/backtest_results` et incluent :
- Fichier JSON avec les statistiques complètes
- Graphique de la courbe d'équité
- Graphique de la distribution des profits/pertes

## 📊 Performances et Analyse

Le bot inclut des outils d'analyse pour évaluer ses performances :

### Analyse des Trades

Pour générer une analyse des trades récents :

```python
from ai.trade_analyzer import TradeAnalyzer
analyzer = TradeAnalyzer(scoring_engine, position_tracker)
report = analyzer.analyze_recent_trades(days=30)
```

### Visualisation des Performances

Pour générer des visualisations de performance :

```python
from utils.visualizer import TradeVisualizer
visualizer = TradeVisualizer(position_tracker)
equity_curve = visualizer.plot_equity_curve(days=30)
trade_analysis = visualizer.plot_trade_analysis(days=30)
```

## 🧠 Système d'IA Auto-Adaptative

L'IA du bot s'améliore automatiquement en analysant ses performances passées.

### Composants de l'IA

- **Moteur de Scoring** : Évalue les opportunités de trading selon multiples critères
- **Analyseur de Trades** : Identifie les patterns de succès et d'échec
- **Optimiseur de Paramètres** : Ajuste les paramètres en fonction des résultats
- **Moteur de Raisonnement** : Génère des explications textuelles pour les décisions

### Cycle d'Apprentissage

1. L'IA analyse les opportunités de trading et attribue un score
2. Le bot exécute les trades avec un score suffisant
3. L'IA analyse les résultats des trades fermés
4. Les poids des différents facteurs sont ajustés en conséquence
5. Les paramètres de la stratégie sont optimisés périodiquement

## 📝 Journal des Trades

Chaque trade est enregistré avec des détails complets, incluant :

- Opportunité initiale avec score et raisonnement
- Conditions de marché lors de l'entrée
- Performance réelle (PnL)
- Auto-critique de l'IA

Les journaux sont stockés au format JSON dans `data/trade_logs/` et peuvent être analysés pour comprendre les décisions du bot.

## ⚠️ Avertissements et Risques

- **Risque de Perte** : Le trading de crypto-monnaies comporte des risques significatifs. N'investissez que ce que vous pouvez vous permettre de perdre.
- **Effet de Levier** : L'utilisation de l'effet de levier amplifie les profits mais aussi les pertes.
- **Tests** : Commencez toujours en mode test ou avec de petits montants avant d'engager des sommes importantes.
- **Maintenance** : Surveillez régulièrement le bot et son fonctionnement.

## 📜 License

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à soumettre des pull requests ou à signaler des problèmes.

## 📧 Contact

Pour toute question ou suggestion, veuillez me contacter à [votre-email@exemple.com].