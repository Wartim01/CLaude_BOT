# Dans train_model.py

import argparse
import pandas as pd
import logging
import os
import json
import numpy as np
import psutil
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback, CSVLogger # type: ignore
from tensorflow.keras.optimizers import Adam, RMSprop # type: ignore
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import sys

# Assurez-vous que les imports locaux fonctionnent
from ai.models.feature_engineering import FeatureEngineering
from ai.models.lstm_model import create_lstm_model, create_attention_lstm_model
from utils.data_splitting import split_data # Assurez-vous que ce chemin est correct
from config.model_params import LSTM_DEFAULT_PARAMS, LSTM_OPTIMIZED_PARAMS
from config.feature_config import FIXED_FEATURES

# Configuration du logger
if not logging.getLogger(__name__).hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levellevel)s - %(message)s')
logger = logging.getLogger(__name__)

# Add file handler so logs are stored in the dedicated logs folder
log_dir = r"c:\Users\timot\OneDrive\Bureau\BOT TRADING BIG 2025\logs"
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "train_model.log"), encoding="utf-8")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levellevel)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# --- Fonction parse_args (inchangée) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement et évaluation du modèle de prédiction")
    parser.add_argument("--symbol", type=str, required=True, help="Paire de trading (ex: BTCUSDT)")
    parser.add_argument("--timeframe", type=str, required=True, help="Intervalle de temps (ex: 15m, 1h)")
    parser.add_argument("--data_path", type=str, required=True, help="Répertoire des données de marché")
    parser.add_argument("--model_path", type=str, default=None, help="Chemin pour sauvegarder le modèle")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "transformer"], help="Type de modèle ('lstm' ou 'transformer')")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Fraction pour validation")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, default=64, help="Taille du batch")
    parser.add_argument("--lstm_units", type=str, default="128,64,32", help="Unités LSTM (ex: '128,64')")
    parser.add_argument("--dropout", type=float, default=0.2, help="Taux de dropout")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Taux d'apprentissage")
    parser.add_argument("--use_attention", action="store_true", help="Utiliser l'attention")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimiseur (adam, rmsprop)")
    parser.add_argument("--no_early_stopping", action="store_false", dest="use_early_stopping", help="Désactiver EarlyStopping")
    parser.add_argument("--patience", type=int, default=10, help="Patience pour EarlyStopping")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Verbosité Keras")
    parser.add_argument("--sequence_length", type=int, default=60, help="Longueur de séquence")
    parser.add_argument("--test_size", type=float, default=0.2, help="Split de test")
    parser.add_argument("--random_state", type=int, default=42, help="Graine aléatoire")
    parser.add_argument("--feature_scaler", type=str, default="standard", help="Scaler features (standard, minmax)")
    parser.add_argument("--train_val_split_method", type=str, default="time", help="Méthode split (time, random)")
    parser.add_argument("--l1_reg", type=float, default=None, help="Régularisation L1 (optionnel, priorité sur config)")
    parser.add_argument("--l2_reg", type=float, default=None, help="Régularisation L2 (optionnel, priorité sur config)")
    return parser.parse_args()

# --- Fonction load_data (inchangée) ---
def load_data(data_dir_path, symbol, timeframe):
    """
    Load market data from CSV file within a directory.
    """
    file_name = f"{symbol}_{timeframe}.csv"
    file_path = os.path.join(data_dir_path, file_name)
    logger.info(f"Construction du chemin de fichier : {file_path}")
    try:
        logger.info(f"Tentative de chargement des données depuis : {file_path}")
        if not os.path.isfile(file_path):
             found_file = None
             if os.path.isdir(data_dir_path):
                 for f in os.listdir(data_dir_path):
                     if f.lower() == file_name.lower():
                         found_file = os.path.join(data_dir_path, f)
                         logger.info(f"Fichier trouvé avec recherche insensible à la casse: {found_file}")
                         break
             if not found_file:
                 error_msg = f"Fichier de données non trouvé : {file_path}"
                 logger.error(error_msg)
                 raise FileNotFoundError(error_msg)
             file_path = found_file
        df = pd.read_csv(file_path)
        if df.empty:
            error_msg = f"Le fichier {file_path} est vide"
            logger.error(error_msg)
            raise ValueError(error_msg)
        df.columns = [col.lower() for col in df.columns]
        time_col = None
        for possible_col in ['timestamp', 'date', 'time', 'datetime']:
            if possible_col in df.columns:
                time_col = possible_col
                break
        if time_col is None:
            error_msg = f"Aucune colonne timestamp/date trouvée dans {file_path}. Colonnes disponibles : {df.columns.tolist()}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        try:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df.dropna(subset=[time_col], inplace=True)
            df.set_index(time_col, inplace=True)
            df.sort_index(inplace=True)
            logger.info(f"Données chargées avec succès avec {len(df)} lignes depuis {file_path}")
            return df
        except Exception as e:
            error_msg = f"Erreur lors de la conversion de {time_col} en datetime dans {file_path}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier de données non trouvé : {file_path}")
    except pd.errors.ParserError as e:
        error_msg = f"Erreur lors de l'analyse du fichier CSV {file_path}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Erreur inattendue lors du chargement de {file_path}: {str(e)}"
        logger.error(error_msg)
        raise type(e)(error_msg)


# --- Callbacks (inchangés) ---
class EarlyStoppingOnMemoryLeak(Callback):
    # ... (code inchangé) ...
    """Arrête l'entraînement si une fuite mémoire est détectée"""
    def __init__(self, patience=3, threshold_mb=300, monitor_interval=1):
        super(EarlyStoppingOnMemoryLeak, self).__init__()
        self.patience = patience
        self.threshold_mb = threshold_mb
        self.monitor_interval = monitor_interval
        self.last_usage = []
        self.epochs_increasing = 0
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.monitor_interval == 0:
            process = psutil.Process(os.getpid())
            memory_usage_mb = process.memory_info().rss / (1024 * 1024)
            logger.info(f"Utilisation mémoire au début époque {epoch+1}: {memory_usage_mb:.2f} MB")
            self.last_usage.append(memory_usage_mb)
            if len(self.last_usage) > self.patience + 2:
                self.last_usage.pop(0)
            if len(self.last_usage) >= 3:
                increasing = True
                for i in range(1, len(self.last_usage)):
                    if self.last_usage[i] - self.last_usage[i-1] < self.threshold_mb:
                        increasing = False
                        break
                if increasing:
                    self.epochs_increasing += 1
                    logger.warning(f"Augmentation continue de mémoire (> {self.threshold_mb}MB) détectée: {self.epochs_increasing}/{self.patience}")
                    if self.epochs_increasing >= self.patience:
                        logger.error(f"Fuite mémoire suspectée - arrêt de l'entraînement")
                        self.model.stop_training = True
                else:
                    self.epochs_increasing = 0

class MetricsCallback(Callback):
    # ... (code inchangé) ...
    def __init__(self, validation_data, threshold=0.5):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data
        self.threshold = threshold
        self.val_f1_scores = []
        self.best_val_f1 = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        x_val, y_val = self.validation_data
        # Vérifier si les données de validation sont vides
        if x_val.shape[0] == 0:
            logger.warning(f"Époque {epoch+1}: Données de validation vides, impossible de calculer les métriques.")
            # Assigner des valeurs par défaut ou NaN aux logs pour éviter les erreurs
            logs['val_f1_score'] = np.nan
            logs['val_precision'] = np.nan
            logs['val_recall'] = np.nan
            logs['val_accuracy'] = np.nan
            self.val_f1_scores.append(np.nan)
            return # Sortir si pas de données de validation

        y_pred = self.model.predict(x_val)
        y_pred_direction = y_pred[0] if isinstance(y_pred, list) else y_pred
        y_val_direction = y_val[0] if isinstance(y_val, list) or isinstance(y_val, tuple) else y_val

        # Assurer que y_val_direction n'est pas vide
        if y_val_direction.shape[0] == 0:
            logger.warning(f"Époque {epoch+1}: y_val_direction est vide, impossible de calculer les métriques.")
            logs['val_f1_score'] = np.nan
            logs['val_precision'] = np.nan
            logs['val_recall'] = np.nan
            logs['val_accuracy'] = np.nan
            self.val_f1_scores.append(np.nan)
            return

        y_pred_classes = (y_pred_direction > self.threshold).astype(int)
        val_f1 = f1_score(y_val_direction, y_pred_classes, average='weighted', zero_division=0)
        val_precision = precision_score(y_val_direction, y_pred_classes, average='weighted', zero_division=0)
        val_recall = recall_score(y_val_direction, y_pred_classes, average='weighted', zero_division=0)
        val_accuracy = accuracy_score(y_val_direction, y_pred_classes)
        logs['val_f1_score'] = val_f1
        logs['val_precision'] = val_precision
        logs['val_recall'] = val_recall
        logs['val_accuracy'] = val_accuracy
        self.val_f1_scores.append(val_f1)
        logger.info(f"Époque {epoch+1}: val_loss = {logs.get('val_loss', 'N/A'):.4f}, val_accuracy = {val_accuracy:.4f}, val_f1_score = {val_f1:.4f}")
        if val_f1 > self.best_val_f1:
            improvement = val_f1 - self.best_val_f1
            self.best_val_f1 = val_f1
            logger.info(f"Amélioration du F1 score: +{improvement:.4f} (nouveau meilleur: {val_f1:.4f})")
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        current_val_loss = logs.get('val_loss', float('inf'))
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Progression après {epoch+1} époques:")
            logger.info(f"  Meilleur val_f1_score: {self.best_val_f1:.4f}")
            logger.info(f"  Meilleure val_loss: {self.best_val_loss:.4f}")
            logger.info(f"  Époques sans amélioration du F1: {self.epochs_without_improvement}")


# --- Fonction train_model (MODIFIÉE pour l'appel à split_data et la génération des séquences) ---
def train_model(args):
    """
    Entraîne un modèle LSTM pour la prédiction des prix (version corrigée utilisant args)
    """
    # ... (Extraction des paramètres depuis args comme avant) ...
    symbol = args.symbol
    timeframe = args.timeframe
    data_dir_path = args.data_path
    model_path = args.model_path
    epochs = args.epochs
    batch_size_arg = args.batch_size
    lstm_units_str = args.lstm_units
    dropout_arg = args.dropout
    learning_rate_arg = args.learning_rate
    use_attention_arg = args.use_attention
    optimizer_arg = args.optimizer
    use_early_stopping = args.use_early_stopping
    validation_split_arg = args.validation_split
    sequence_length_arg = args.sequence_length
    test_size = args.test_size
    random_state = args.random_state
    feature_scaler = args.feature_scaler
    train_val_split_method = args.train_val_split_method
    l1_reg_arg = args.l1_reg
    l2_reg_arg = args.l2_reg
    patience_arg = args.patience

    logger.info(f"Début de l'entraînement du modèle pour {symbol} {timeframe}")

    # --- 1. Chargement et Validation des Paramètres ---
    # ... (Logique de priorité inchangée) ...
    default_params = LSTM_DEFAULT_PARAMS
    optimized_params = {}
    try:
        optimized_params_path = os.path.join("config", "optimized_params.json")
        if os.path.exists(optimized_params_path):
            with open(optimized_params_path, 'r') as f:
                all_optimized_params = json.load(f)
            if timeframe in all_optimized_params:
                 optimized_params = all_optimized_params[timeframe]
                 logger.info(f"Paramètres optimisés chargés pour {timeframe}")
            else:
                 logger.info(f"Aucun paramètre optimisé trouvé pour timeframe {timeframe}")
    except Exception as e:
        logger.warning(f"Erreur lors du chargement des paramètres optimisés: {str(e)}")

    def get_final_param(cmd_value, opt_key, default_key, default_value, is_list=False, is_bool=False):
        # ... (Logique inchangée) ...
        if cmd_value is not None and cmd_value != default_params.get(default_key, default_value):
            logger.info(f"Utilisation de '{opt_key}' depuis la ligne de commande: {cmd_value}")
            if is_list and isinstance(cmd_value, str):
                try:
                    return [int(unit) for unit in cmd_value.split(',')]
                except ValueError:
                    logger.warning(f"Format lstm_units invalide ('{cmd_value}'), utilisation du défaut.")
                    return default_params.get(default_key, default_value)
            return cmd_value
        elif opt_key in optimized_params:
            opt_val = optimized_params[opt_key]
            if opt_val is not None:
                 logger.info(f"Utilisation de '{opt_key}' depuis les paramètres optimisés: {opt_val}")
                 if is_list and not isinstance(opt_val, list):
                      logger.warning(f"Paramètre optimisé '{opt_key}' n'est pas une liste, tentative de conversion.")
                      try:
                           return [int(u) for u in str(opt_val).split(',')]
                      except:
                           return default_params.get(default_key, default_value)
                 return opt_val
            else:
                 logger.warning(f"Valeur optimisée pour '{opt_key}' est None, utilisation du défaut.")
                 return default_params.get(default_key, default_value)
        else:
            final_value = default_params.get(default_key, default_value)
            logger.info(f"Utilisation de '{opt_key}' par défaut: {final_value}")
            return final_value

    final_lstm_units = get_final_param(lstm_units_str, 'lstm_units', 'lstm_units', default_params.get('lstm_units', [128, 64, 32]), is_list=True)
    final_dropout = get_final_param(dropout_arg, 'dropout', 'dropout_rate', default_params.get('dropout_rate', 0.2))
    final_learning_rate = get_final_param(learning_rate_arg, 'learning_rate', 'learning_rate', default_params.get('learning_rate', 0.001))
    final_optimizer = get_final_param(optimizer_arg, 'optimizer', 'optimizer', default_params.get('optimizer', "adam"))
    final_use_attention = get_final_param(use_attention_arg, 'use_attention', 'use_attention', default_params.get('use_attention', False), is_bool=True)
    final_sequence_length = get_final_param(sequence_length_arg, 'sequence_length', 'sequence_length', default_params.get('sequence_length', 60))
    final_batch_size = get_final_param(batch_size_arg, 'batch_size', 'batch_size', default_params.get('batch_size', 64))
    final_validation_size = get_final_param(validation_split_arg, 'validation_size', 'validation_split', default_params.get('validation_split', 0.2)) # Utiliser le nom correct pour split_data
    final_l1_reg = get_final_param(l1_reg_arg, 'l1_reg', 'l1_regularization', default_params.get('l1_regularization', 0.0001))
    final_l2_reg = get_final_param(l2_reg_arg, 'l2_reg', 'l2_regularization', default_params.get('l2_regularization', 0.0001))
    final_patience = get_final_param(patience_arg, 'patience', 'early_stopping_patience', default_params.get('early_stopping_patience', 10))

    logger.info("Paramètres finaux pour l'entraînement:")
    # ... (logs des paramètres finaux) ...
    logger.info(f"- Unités LSTM: {final_lstm_units}")
    logger.info(f"- Dropout: {final_dropout}")
    logger.info(f"- Taux d'apprentissage: {final_learning_rate}")
    logger.info(f"- Optimiseur: {final_optimizer}")
    logger.info(f"- Utiliser Attention: {final_use_attention}")
    logger.info(f"- Longueur Séquence: {final_sequence_length}")
    logger.info(f"- Taille Batch: {final_batch_size}")
    logger.info(f"- Split Validation: {final_validation_size}") # Log de la valeur utilisée
    logger.info(f"- Régularisation L1: {final_l1_reg}, L2: {final_l2_reg}")
    logger.info(f"- Patience EarlyStopping: {final_patience}")


    # --- 2. Chargement et Préparation des Données ---
    df = load_data(data_dir_path, symbol, timeframe)
    feature_engineering = FeatureEngineering(save_scalers=True)
    data_with_features = feature_engineering.create_features(df)
    normalized_data = feature_engineering.scale_features(
        data_with_features,
        is_training=True,
        method=feature_scaler,
        feature_group=f"{symbol}_{timeframe}"
    )

    # --- 3. Vérification de Cohérence des Caractéristiques ---
    # ... (code de vérification inchangé) ...
    actual_features = list(normalized_data.columns)
    expected_features = FIXED_FEATURES
    missing_features = [f for f in expected_features if f not in actual_features]
    extra_features = [f for f in actual_features if f not in expected_features]
    order_correct = False
    if not missing_features and not extra_features:
         order_correct = (actual_features == expected_features)

    if missing_features or extra_features or not order_correct:
        error_msg = "Caractéristiques incompatibles détectées après normalisation:\n"
        # ... (logique d'erreur et réharmonisation) ...
        logger.warning("Tentative de réharmonisation des caractéristiques...")
        try:
            normalized_data = normalized_data.reindex(columns=FIXED_FEATURES, fill_value=0)
            if normalized_data.shape[1] == len(FIXED_FEATURES):
                 logger.info(f"Réharmonisation réussie avec {len(normalized_data.columns)} caractéristiques")
            else:
                 raise ValueError("Échec réharmonisation: dimensions incorrectes")
        except Exception as e:
            error_detail = f"Échec de la réharmonisation: {str(e)}"
            logger.error(error_detail)
            raise ValueError(f"Impossible de continuer: {error_detail}")


    # --- 4. Division des Données (Workflow Corrigé) ---
    logger.info("Division des données en ensembles train/validation/test...")
    split_result = split_data(
        normalized_data,
        sequence_length=final_sequence_length,
        validation_size=final_validation_size,
        test_size=test_size,
        random_state=random_state
    )
    # Extraction correcte via clés:
    train_df = split_result['train']
    val_df = split_result['validation']
    test_df = split_result['test']
    logger.info("Génération des séquences pour l'entraînement...")
    X_train, y_train = feature_engineering.create_multi_horizon_data(
        train_df,
        sequence_length=final_sequence_length,
        is_training=True
    )
    X_val, y_val = feature_engineering.create_multi_horizon_data(
        val_df,
        sequence_length=final_sequence_length,
        is_training=True
    )
    X_test, y_test = feature_engineering.create_multi_horizon_data(
        test_df,
        sequence_length=final_sequence_length,
        is_training=True
    )
    
    # --- 5. Validation des Dimensions ---
    logger.info("=== Dimensions et types des données d'entraînement ===")
    logger.info(f"X_train: shape={X_train.shape}, dtype={X_train.dtype}")
    if isinstance(y_train, list) or isinstance(y_train, tuple):
        for i, y in enumerate(y_train): logger.info(f"y_train[{i}]: shape={y.shape}, dtype={y.dtype}")
    else: logger.info(f"y_train: shape={y_train.shape}, dtype={y_train.dtype}")

    logger.info("=== Dimensions et types des données de validation ===")
    logger.info(f"X_val: shape={X_val.shape}, dtype={X_val.dtype}")
    if isinstance(y_val, list) or isinstance(y_val, tuple):
        for i, y in enumerate(y_val): logger.info(f"y_val[{i}]: shape={y.shape}, dtype={y.dtype}")
    else: logger.info(f"y_val: shape={y_val.shape}, dtype={y_val.dtype}")

    logger.info("=== Dimensions et types des données de test ===")
    logger.info(f"X_test: shape={X_test.shape}, dtype={X_test.dtype}")
    if isinstance(y_test, list) or isinstance(y_test, tuple):
        for i, y in enumerate(y_test): logger.info(f"y_test[{i}]: shape={y.shape}, dtype={y.dtype}")
    else: logger.info(f"y_test: shape={y_test.shape}, dtype={y_test.dtype}")

    try:
        # Vérifier la dimension temporelle
        assert X_train.shape[1] == final_sequence_length, f"Dimension temporelle X_train: {X_train.shape[1]} != {final_sequence_length}"
        assert X_val.shape[1] == final_sequence_length, f"Dimension temporelle X_val: {X_val.shape[1]} != {final_sequence_length}"
        assert X_test.shape[1] == final_sequence_length, f"Dimension temporelle X_test: {X_test.shape[1]} != {final_sequence_length}"

        # Vérifier la dimension des caractéristiques
        # Note: Le nombre de features est déterminé par create_multi_horizon_data,
        # qui devrait utiliser les colonnes de normalized_data (après harmonisation)
        expected_features_count = normalized_data.shape[1] # Utiliser le nombre de colonnes après harmonisation
        assert X_train.shape[2] == expected_features_count, f"Nb caractéristiques X_train: {X_train.shape[2]} != {expected_features_count}"
        assert X_val.shape[2] == expected_features_count, f"Nb caractéristiques X_val: {X_val.shape[2]} != {expected_features_count}"
        assert X_test.shape[2] == expected_features_count, f"Nb caractéristiques X_test: {X_test.shape[2]} != {expected_features_count}"

        # Vérifier la cohérence des échantillons
        if isinstance(y_train, list) or isinstance(y_train, tuple):
            for i, y in enumerate(y_train): assert y.shape[0] == X_train.shape[0], f"Incohérence X_train/y_train[{i}]"
        else: assert y_train.shape[0] == X_train.shape[0], "Incohérence X_train/y_train"

        if isinstance(y_val, list) or isinstance(y_val, tuple):
            for i, y in enumerate(y_val): assert y.shape[0] == X_val.shape[0], f"Incohérence X_val/y_val[{i}]"
        else: assert y_val.shape[0] == X_val.shape[0], "Incohérence X_val/y_val"

        if isinstance(y_test, list) or isinstance(y_test, tuple):
            for i, y in enumerate(y_test): assert y.shape[0] == X_test.shape[0], f"Incohérence X_test/y_test[{i}]"
        else: assert y_test.shape[0] == X_test.shape[0], "Incohérence X_test/y_test"

        logger.info("✅ Validation des dimensions: Toutes les vérifications ont réussi")
    except AssertionError as e:
         logger.critical(f"ERREUR CRITIQUE DE DIMENSION: {str(e)}")
         raise ValueError(f"Arrêt de l'entraînement: {str(e)}")

    # --- 6. Création du Modèle ---
    input_shape = (final_sequence_length, X_train.shape[2]) # Utiliser la dimension réelle de X_train
    model_kwargs = {
        'dropout': final_dropout,
        'l1_reg': final_l1_reg,
        'l2_reg': final_l2_reg
    }
    if final_use_attention:
        model = create_attention_lstm_model(input_shape, final_lstm_units, **model_kwargs)
        logger.info(f"Modèle LSTM avec attention créé: {final_lstm_units}")
    else:
        model = create_lstm_model(input_shape, final_lstm_units, **model_kwargs)
        logger.info(f"Modèle LSTM standard créé: {final_lstm_units}")

    # --- 7. Configuration de l'Optimiseur ---
    if final_optimizer.lower() == 'adam':
        opt = Adam(learning_rate=final_learning_rate)
    elif final_optimizer.lower() == 'rmsprop':
        opt = RMSprop(learning_rate=final_learning_rate)
    else:
        opt = Adam(learning_rate=final_learning_rate)

    # --- 8. Compilation du Modèle ---
    # Adapter pour multi-sorties si y_train est une liste
    if isinstance(y_train, list):
        losses = ['binary_crossentropy'] * len(y_train) # Supposer toutes binaires pour l'instant
        metrics = ['accuracy'] * len(y_train)
    else:
        losses = 'binary_crossentropy'
        metrics = ['accuracy']
    model.compile(optimizer=opt, loss=losses, metrics=metrics)
    logger.info("Modèle compilé.")

    # --- 9. Callbacks ---
    callbacks = []
    # S'assurer que validation_data est un tuple (X_val, y_val)
    validation_data_for_callback = (X_val, y_val)
    metrics_callback = MetricsCallback(validation_data=validation_data_for_callback, threshold=0.5)
    callbacks.append(metrics_callback)
    model_path_keras = model_path
    if model_path and not model_path.endswith('.keras'):
        model_path_keras = os.path.splitext(model_path)[0] + '.keras'
        logger.warning(f"ModelCheckpoint utilisera l'extension .keras: {model_path_keras}")
    if model_path_keras:
        os.makedirs(os.path.dirname(model_path_keras), exist_ok=True)
        checkpoint = ModelCheckpoint(model_path_keras, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        callbacks.append(checkpoint)
    csv_log_path = os.path.splitext(model_path_keras)[0] + '_history.csv' if model_path_keras else f'training_{symbol}_{timeframe}_history.csv'
    os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
    csv_logger = CSVLogger(csv_log_path, append=True)
    callbacks.append(csv_logger)
    logger.info(f"Logging CSV activé vers: {csv_log_path}")
    if use_early_stopping:
        early_stopping_loss = EarlyStopping(monitor='val_loss', patience=final_patience, restore_best_weights=True, mode='min', verbose=1)
        callbacks.append(early_stopping_loss)
        # early_stopping_f1 = EarlyStopping(monitor='val_f1_score', patience=final_patience + 5, restore_best_weights=True, mode='max', verbose=1)
        # callbacks.append(early_stopping_f1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=final_patience // 2, min_lr=1e-6, verbose=1)
    callbacks.append(reduce_lr)
    memory_leak_callback = EarlyStoppingOnMemoryLeak(patience=3, threshold_mb=300)
    callbacks.append(memory_leak_callback)

    # --- 10. Calcul des Poids de Classe ---
    class_weights = None
    try:
        y_target_for_weights = y_train[0] if isinstance(y_train, list) else y_train
        y_train_direction = y_target_for_weights.flatten().astype(int)
        unique_classes = np.unique(y_train_direction)
        if len(unique_classes) == 2:
            class_weights_values = compute_class_weight('balanced', classes=unique_classes, y=y_train_direction)
            class_weights = {int(unique_classes[i]): float(class_weights_values[i]) for i in range(len(unique_classes))}
            logger.info(f"Poids des classes calculés: {class_weights}")
        else:
            logger.warning(f"Impossible de calculer les poids de classe ({len(unique_classes)} classes uniques).")
    except Exception as e:
        logger.warning(f"Erreur lors du calcul des poids de classe: {str(e)}")

    # --- 11. Entraînement ---
    logger.info("Début de l'entraînement du modèle...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=final_batch_size,
        validation_data=(X_val, y_val),
        # class_weight=class_weights,  # Ensure this line is removed or commented out
        callbacks=callbacks,
        verbose=args.verbose
    )

    # --- 12. Évaluation Finale et Résumé ---
    logger.info("Évaluation finale sur les données de test...")
    # ... (code d'évaluation et de logging du résumé) ...
    test_eval_results = model.evaluate(X_test, y_test, verbose=0)
    # Adapter l'extraction des métriques si multi-sorties
    test_loss = test_eval_results[0] if isinstance(test_eval_results, list) else test_eval_results
    test_acc = test_eval_results[1] if isinstance(test_eval_results, list) and len(test_eval_results) > 1 else np.nan

    y_pred_test = model.predict(X_test)
    y_pred_test_classes = (y_pred_test[0] > 0.5).astype(int) if isinstance(y_pred_test, list) else (y_pred_test > 0.5).astype(int)
    y_test_direction = y_test[0] if isinstance(y_test, list) else y_test
    test_f1 = f1_score(y_test_direction, y_pred_test_classes, average='weighted', zero_division=0)

    logger.info("==========================================")
    logger.info("Résumé final de l'entraînement du modèle:")
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    best_val_loss = min(history.history.get('val_loss', [float('inf')]))
    best_val_f1 = max(metrics_callback.val_f1_scores) if metrics_callback.val_f1_scores else 0
    logger.info(f"Meilleure val_loss: {best_val_loss:.4f}")
    logger.info(f"Meilleur val_f1_score: {best_val_f1:.4f}")
    # ... (logs de performance cible et convergence) ...
    if best_val_f1 >= 0.75: logger.info("✅ Performance cible atteinte: F1 score ≥ 0.75")
    elif best_val_f1 >= 0.65: logger.info("⚠️ Performance acceptable: 0.65 ≤ F1 score < 0.75")
    else: logger.warning("❌ Performance insuffisante: F1 score < 0.65")
    initial_val_loss = history.history.get('val_loss', [0])[0]
    final_val_loss = history.history.get('val_loss', [0])[-1]
    if initial_val_loss > 0:
         loss_improvement = (initial_val_loss - final_val_loss) / initial_val_loss * 100
         if loss_improvement >= 30: logger.info(f"✅ Bonne convergence: Réduction de la perte de {loss_improvement:.1f}%")
         elif loss_improvement >= 10: logger.info(f"⚠️ Convergence modérée: Réduction de la perte de {loss_improvement:.1f}%")
         else: logger.warning(f"❌ Faible convergence: Réduction de la perte de seulement {loss_improvement:.1f}%")
    else: logger.warning("Impossible de calculer l'amélioration de la perte.")
    logger.info("==========================================")


    return {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_f1_score": test_f1,
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
        "trained_epochs": len(history.history['loss'])
    }

# --- Point d'entrée principal (inchangé) ---
if __name__ == "__main__":
    args = parse_args()
    try:
        train_model(args)
    except ValueError as ve:
        logger.critical(f"Erreur critique pendant l'entraînement: {ve}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Erreur inattendue pendant l'entraînement: {e}", exc_info=True)
        sys.exit(1)
