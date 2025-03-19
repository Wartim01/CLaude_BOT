# Auto-Configuration du Nombre de Caractéristiques

Cette documentation explique comment utiliser la fonctionnalité d'auto-optimisation du nombre de caractéristiques dans le bot de trading.

## Introduction

Le nombre de caractéristiques (features) utilisées par un modèle d'apprentissage automatique peut avoir un impact significatif sur ses performances. Un nombre trop faible de caractéristiques peut conduire à un sous-apprentissage, tandis qu'un nombre trop élevé peut entraîner un sur-apprentissage ou une "malédiction de la dimensionnalité".

La fonctionnalité d'auto-configuration permet de déterminer automatiquement le nombre optimal de caractéristiques basé sur les données d'entraînement, améliorant ainsi les performances du modèle.

## Fonctionnalités

- **Analyse automatique** des performances en fonction du nombre de caractéristiques
- **Sélection intelligente** des caractéristiques les plus importantes
- **Génération de caractéristiques additionnelles** via PCA si nécessaire
- **Visualisation** des résultats d'optimisation
- **Persistance** de la configuration optimisée pour une utilisation future

## Utilisation

### Option 1: Via la classe FeatureEngineering

```python
from ai.models.feature_engineering import FeatureEngineering

# Créer une instance avec auto-optimisation activée
fe = FeatureEngineering(auto_optimize=True)

# Déterminer le nombre optimal de caractéristiques
optimal_count = fe.optimize_feature_count(
    data,
    min_features=30,
    max_features=100,
    step_size=10
)

# Configurer le pipeline avec le nombre optimal
fe.optimize_and_configure(data, save_config=True)

# Toutes les futures générations de caractéristiques utiliseront
# automatiquement le nombre optimal
features = fe.create_features(new_data)
```

### Option 2: Via l'utilitaire en ligne de commande

```bash
python utils/feature_optimizer.py --symbol BTCUSDT --timeframe 1h --min 30 --max 100 --step 10
```

## Paramètres d'optimisation

- `min_features`: Nombre minimum de caractéristiques à évaluer
- `max_features`: Nombre maximum de caractéristiques à évaluer
- `step_size`: Pas d'incrément pour tester différents nombres de caractéristiques
- `cv_folds`: Nombre de plis pour la validation croisée temporelle

## Métriques d'évaluation

L'optimisation utilise plusieurs métriques pour évaluer les performances:

- **F1-score**: Moyenne harmonique entre précision et rappel
- **Accuracy**: Pourcentage de prédictions correctes
- **Precision**: Ratio de vrais positifs parmi les prédictions positives
- **Recall**: Ratio de vrais positifs détectés

La sélection du nombre optimal de caractéristiques est principalement basée sur le **F1-score**.

## Algorithme de sélection des caractéristiques

1. Génération de toutes les caractéristiques possibles
2. Pour chaque nombre de caractéristiques à tester:
   - Sélection des caractéristiques par variance ou importance
   - Validation croisée temporelle avec un modèle Random Forest
   - Calcul des métriques de performance
3. Détermination du nombre optimal de caractéristiques
4. Configuration du pipeline avec ce nombre optimal

## Visualisation des résultats

L'outil génère automatiquement des graphiques montrant la relation entre le nombre de caractéristiques et les performances du modèle.

## Conseils d'utilisation

- Exécutez l'optimisation sur un échantillon représentatif de vos données
- Relancez périodiquement l'optimisation lors de changements significatifs dans les conditions du marché
- Assurez-vous d'avoir suffisamment de données pour une évaluation fiable (au moins 1000 points de données recommandés)
- Tenez compte des compromis entre performances et complexité du modèle

## Limitations

- L'optimisation peut prendre du temps sur de grands ensembles de données
- La sélection des caractéristiques est basée sur un modèle Random Forest, qui peut ne pas être optimal pour le modèle LSTM final
