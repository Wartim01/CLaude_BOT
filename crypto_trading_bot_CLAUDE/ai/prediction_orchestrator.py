"""
Orchestrates multiple prediction models and combines their outputs
to generate unified trading signals and confidence metrics
"""
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from enum import Enum

from config.config import DATA_DIR
from utils.logger import setup_logger
from ai.models.lstm_model import LSTMModel, EnhancedLSTMModel
from ai.models.feature_engineering import FeatureEngineering
from ai.models.continuous_learning import AdvancedContinuousLearning
from ai.scoring_engine import ScoringEngine

logger = setup_logger("prediction_orchestrator")

class PredictionSource(Enum):
    """Enum for different prediction sources"""
    LSTM = "lstm"
    TECHNICAL = "technical"
    ENSEMBLE = "ensemble"
    REVERSAL_DETECTOR = "reversal_detector"
    SCORING_ENGINE = "scoring_engine"

class PredictionOrchestrator:
    """
    Manages multiple prediction models, combines their outputs,
    and provides unified predictions with confidence metrics
    """
    def __init__(self):
        # Initialize components
        self.models = {}
        self.feature_engineering = FeatureEngineering(save_scalers=True)
        self.scoring_engine = ScoringEngine()
        self.continuous_learning = None
        
        # Weights for ensemble predictions
        self.ensemble_weights = {
            PredictionSource.LSTM.value: 0.6,
            PredictionSource.TECHNICAL.value: 0.3,
            PredictionSource.REVERSAL_DETECTOR.value: 0.1
        }
        
        # Track prediction history
        self.prediction_history = {}
        self.max_history_items = 1000
        
        # Performance metrics for models
        self.performance_metrics = {}
        
        # Background update thread
        self.update_thread = None
        self.should_stop = False
        self.update_interval = 3600  # 1 hour
        
        # Directories
        self.models_dir = os.path.join(DATA_DIR, "models", "production")
        self.predictions_dir = os.path.join(DATA_DIR, "predictions")
        self.history_path = os.path.join(self.predictions_dir, "prediction_history.json")
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        
        # Load models
        self._load_models()
        
        # Load prediction history
        self._load_prediction_history()
    
    def _load_models(self):
        """Load all available prediction models"""
        try:
            # Load enhanced LSTM model if available
            enhanced_lstm_path = os.path.join(self.models_dir, "enhanced_lstm_model.h5")
            if os.path.exists(enhanced_lstm_path):
                model = EnhancedLSTMModel()
                model.load(enhanced_lstm_path)
                self.models[PredictionSource.LSTM.value] = model
                logger.info(f"Loaded enhanced LSTM model: {enhanced_lstm_path}")
                
                # Initialize continuous learning for this model
                self.continuous_learning = AdvancedContinuousLearning(
                    model=model,
                    feature_engineering=self.feature_engineering
                )
            else:
                # Try to load standard LSTM model
                lstm_path = os.path.join(self.models_dir, "lstm_final.h5")
                if os.path.exists(lstm_path):
                    model = LSTMModel()
                    model.load(lstm_path)
                    self.models[PredictionSource.LSTM.value] = model
                    logger.info(f"Loaded standard LSTM model: {lstm_path}")
                else:
                    logger.warning("No LSTM model found")
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
    
    def _load_prediction_history(self):
        """Load prediction history from disk"""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    self.prediction_history = json.load(f)
                logger.info(f"Loaded prediction history with {len(self.prediction_history)} entries")
            except Exception as e:
                logger.error(f"Error loading prediction history: {str(e)}")
                self.prediction_history = {}
    
    def _save_prediction_history(self):
        """Save prediction history to disk"""
        try:
            # Limit the size of history
            for symbol in self.prediction_history:
                if len(self.prediction_history[symbol]) > self.max_history_items:
                    self.prediction_history[symbol] = self.prediction_history[symbol][-self.max_history_items:]
            
            with open(self.history_path, 'w') as f:
                json.dump(self.prediction_history, f, indent=2)
            logger.debug("Prediction history saved")
        except Exception as e:
            logger.error(f"Error saving prediction history: {str(e)}")
    
    def start_auto_updates(self):
        """Start automatic prediction updates in the background"""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.should_stop = False
            self.update_thread = threading.Thread(target=self._update_worker)
            self.update_thread.daemon = True
            self.update_thread.start()
            logger.info(f"Automatic prediction updates started (interval: {self.update_interval}s)")
    
    def stop_auto_updates(self):
        """Stop automatic prediction updates"""
        self.should_stop = True
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
            logger.info("Automatic prediction updates stopped")
    
    def _update_worker(self):
        """Background worker thread for automatic updates"""
        while not self.should_stop:
            try:
                # This would typically trigger updates for all tracked symbols
                # For now, we'll just log that we would do this
                logger.debug("Automatic prediction update triggered")
                
                # In a real implementation, this would fetch the latest data
                # for all tracked symbols and update predictions
                
                # Sleep until next update
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in prediction update worker: {str(e)}")
                time.sleep(60)  # Sleep briefly on error before retry
    
    def get_prediction(self, symbol: str, 
                     data: pd.DataFrame,
                     timeframe: str = '1h',
                     include_details: bool = False) -> Dict:
        """
        Generate predictions for the given symbol and data
        
        Args:
            symbol: Trading symbol (e.g. 'BTCUSDT')
            data: DataFrame with OHLCV data
            timeframe: Data timeframe ('1h', '15m', etc.)
            include_details: Whether to include detailed model outputs
            
        Returns:
            Dictionary with unified prediction and confidence
        """
        if data.empty:
            return {
                "success": False,
                "error": "Empty data provided"
            }
        
        # Store all predictions from different models
        all_predictions = {}
        prediction_errors = {}
        
        # 1. Get LSTM model predictions if available
        if PredictionSource.LSTM.value in self.models:
            try:
                lstm_predictions = self._get_lstm_prediction(symbol, data, timeframe)
                all_predictions[PredictionSource.LSTM.value] = lstm_predictions
            except Exception as e:
                logger.error(f"Error getting LSTM predictions for {symbol}: {str(e)}")
                prediction_errors[PredictionSource.LSTM.value] = str(e)
        
        # 2. Get technical scoring predictions
        try:
            technical_score = self._get_technical_score(symbol, data)
            all_predictions[PredictionSource.TECHNICAL.value] = technical_score
        except Exception as e:
            logger.error(f"Error getting technical score for {symbol}: {str(e)}")
            prediction_errors[PredictionSource.TECHNICAL.value] = str(e)
        
        # 3. Check for pattern reversals from reversal detector
        if PredictionSource.LSTM.value in self.models and hasattr(self.models[PredictionSource.LSTM.value], 'reversal_detector'):
            try:
                reversal_prediction = self._get_reversal_prediction(symbol, data)
                all_predictions[PredictionSource.REVERSAL_DETECTOR.value] = reversal_prediction
            except Exception as e:
                logger.error(f"Error getting reversal predictions for {symbol}: {str(e)}")
                prediction_errors[PredictionSource.REVERSAL_DETECTOR.value] = str(e)
        
        # 4. Combine all predictions into a unified prediction
        unified_prediction = self._combine_predictions(all_predictions, symbol)
        
        # 5. Add prediction to history
        self._add_to_prediction_history(symbol, unified_prediction, all_predictions)
        
        # 6. If continuous learning is enabled, process the new data
        if self.continuous_learning is not None:
            try:
                self.continuous_learning.process_new_data(data)
            except Exception as e:
                logger.error(f"Error processing data for continuous learning: {str(e)}")
        
        # 7. Prepare the response
        response = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "prediction": unified_prediction
        }
        
        # Include individual model outputs if requested
        if include_details:
            response["model_predictions"] = all_predictions
            if prediction_errors:
                response["errors"] = prediction_errors
        
        return response
    
    def _get_lstm_prediction(self, symbol: str, data: pd.DataFrame, timeframe: str) -> Dict:
        """
        Get predictions from LSTM model
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
            timeframe: Data timeframe
            
        Returns:
            LSTM prediction results
        """
        # Get the model
        model = self.models[PredictionSource.LSTM.value]
        
        # Process the data for prediction
        predictions = model.predict(data, self.feature_engineering)
        
        # Format the results
        formatted_results = {
            "direction": {},
            "confidence": {}
        }
        
        # Extract the relevant prediction values from the model output
        for horizon_name, horizon_data in predictions.items():
            direction = horizon_data["direction"]
            probability = horizon_data["direction_probability"]
            confidence = horizon_data["confidence"]
            
            formatted_results["direction"][horizon_name] = direction
            formatted_results["confidence"][horizon_name] = confidence
            formatted_results["probability"] = probability
        
        return formatted_results
    
    def _get_technical_score(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Get technical analysis score
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
            
        Returns:
            Technical analysis score and details
        """
        # Prepare data for technical analysis
        from strategies.technical_bounce import detect_technical_bounce
        from strategies.market_state import analyze_market_state
        
        # Analyze market state
        market_state = analyze_market_state(data)
        
        # Detect bounce opportunities
        bounce_results = detect_technical_bounce(data, market_state)
        
        # Calculate score using scoring engine
        score_input = {
            "bounce_signals": bounce_results,
            "market_state": market_state,
            "ohlcv": data,
            "indicators": {}  # We would add indicators here in a full implementation
        }
        
        score_result = self.scoring_engine.calculate_score(score_input, "technical_bounce")
        
        return {
            "score": score_result["score"],
            "signals": bounce_results.get("signals", []),
            "market_state": market_state.get("state", "unknown"),
            "details": score_result.get("details", {})
        }
    
    def _get_reversal_prediction(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Get pattern reversal predictions
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
            
        Returns:
            Reversal prediction results
        """
        # Get the model with reversal detector
        model = self.models[PredictionSource.LSTM.value]
        
        # Process the data for prediction
        _, normalized_data = self.feature_engineering.prepare_data(data)
        
        X, _ = self.feature_engineering.create_multi_horizon_data(
            normalized_data,
            sequence_length=model.input_length,
            horizons=model.prediction_horizons,
            is_training=False
        )
        
        # Get reversal predictions
        reversal_prob, reversal_mag = model.reversal_detector.predict(X)
        
        # Get the latest prediction
        latest_idx = -1
        probability = float(reversal_prob[latest_idx][0])
        magnitude = float(reversal_mag[latest_idx][0])
        
        return {
            "probability": probability,
            "magnitude": magnitude,
            "is_warning": probability > 0.7,  # Warning if probability > 70%
            "direction": "down" if probability > 0.7 else "none"  # Reversal is typically downward
        }
    
    def _combine_predictions(self, all_predictions: Dict, symbol: str) -> Dict:
        """
        Combine predictions from different models into a unified prediction
        
        Args:
            all_predictions: Dictionary of predictions from different models
            symbol: Trading symbol
            
        Returns:
            Unified prediction
        """
        # Default to neutral prediction if we have no data
        if not all_predictions:
            return {
                "signal": "NEUTRAL",
                "direction": "NEUTRAL",
                "confidence": 0.5,
                "horizon": "unknown",
                "strength": 0
            }
        
        # Extract LSTM direction signal (focus on shortest horizon - "3h")
        lstm_direction = "NEUTRAL"
        lstm_confidence = 0.5
        lstm_probability = 50.0
        
        if PredictionSource.LSTM.value in all_predictions:
            lstm_pred = all_predictions[PredictionSource.LSTM.value]
            
            # Try to get the shortest horizon prediction
            for horizon in ["3h", "1h", "short"]:  # Try different horizon names
                if horizon in lstm_pred.get("direction", {}):
                    lstm_direction = lstm_pred["direction"][horizon]
                    lstm_confidence = lstm_pred["confidence"].get(horizon, 0.5)
                    break
            
            lstm_probability = lstm_pred.get("probability", 50.0)
        
        # Extract technical score and convert to direction and confidence
        technical_score = 0
        technical_signals = []
        
        if PredictionSource.TECHNICAL.value in all_predictions:
            tech_pred = all_predictions[PredictionSource.TECHNICAL.value]
            technical_score = tech_pred.get("score", 0)
            technical_signals = tech_pred.get("signals", [])
        
        # Convert technical score to direction and confidence
        technical_direction = "NEUTRAL"
        technical_confidence = 0.5
        
        if technical_score >= 70:
            technical_direction = "BULLISH"
            technical_confidence = technical_score / 100
        elif technical_score <= 30:
            technical_direction = "BEARISH"
            technical_confidence = (100 - technical_score) / 100
        else:
            technical_confidence = 1.0 - abs((technical_score - 50) / 50)
        
        # Check for reversal warnings
        reversal_warning = False
        reversal_probability = 0.0
        
        if PredictionSource.REVERSAL_DETECTOR.value in all_predictions:
            reversal_pred = all_predictions[PredictionSource.REVERSAL_DETECTOR.value]
            reversal_warning = reversal_pred.get("is_warning", False)
            reversal_probability = reversal_pred.get("probability", 0.0)
        
        # Combine the signals using ensemble weights
        lstm_weight = self.ensemble_weights.get(PredictionSource.LSTM.value, 0.6)
        technical_weight = self.ensemble_weights.get(PredictionSource.TECHNICAL.value, 0.3)
        reversal_weight = self.ensemble_weights.get(PredictionSource.REVERSAL_DETECTOR.value, 0.1)
        
        # Adjust weights based on which models are available
        total_weight = 0
        for source in [PredictionSource.LSTM.value, PredictionSource.TECHNICAL.value]:
            if source in all_predictions:
                total_weight += self.ensemble_weights.get(source, 0)
        
        if total_weight == 0:
            total_weight = 1  # Avoid division by zero
        
        # Calculate bullish and bearish scores
        bullish_score = 0
        bearish_score = 0
        
        if PredictionSource.LSTM.value in all_predictions:
            if lstm_direction == "BULLISH":
                bullish_score += lstm_weight * (lstm_probability / 100)
            elif lstm_direction == "BEARISH":
                bearish_score += lstm_weight * ((100 - lstm_probability) / 100)
        
        if PredictionSource.TECHNICAL.value in all_predictions:
            if technical_direction == "BULLISH":
                bullish_score += technical_weight * technical_confidence
            elif technical_direction == "BEARISH":
                bearish_score += technical_weight * technical_confidence
        
        # Apply reversal warning as a modifier (if significant)
        if reversal_warning and reversal_probability > 0.7:
            bearish_score += reversal_weight * reversal_probability
        
        # Normalize scores
        bullish_score = bullish_score / total_weight
        bearish_score = bearish_score / total_weight
        
        # Determine final signal
        signal_strength = abs(bullish_score - bearish_score)
        
        if bullish_score > bearish_score + 0.2:  # Bullish with margin
            signal = "BULLISH"
            direction = "BUY"
        elif bearish_score > bullish_score + 0.2:  # Bearish with margin
            signal = "BEARISH"
            direction = "SELL"
        else:  # Not enough confidence to determine direction
            signal = "NEUTRAL"
            direction = "NEUTRAL"
        
        # Calculate confidence (0.5-1.0, where 0.5 is neutral)
        confidence = 0.5 + abs(bullish_score - bearish_score) / 2
        
        # Normalize strength to 0-10 scale
        strength = int(signal_strength * 10)
        
        # Create unified prediction
        unified_prediction = {
            "signal": signal,
            "direction": direction,
            "confidence": confidence,
            "strength": strength,
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "horizon": "short-term",  # Default to short-term
            "signals": technical_signals,
            "reversal_warning": reversal_warning
        }
        
        return unified_prediction
    
    def _add_to_prediction_history(self, symbol: str, unified_prediction: Dict, all_predictions: Dict):
        """
        Add prediction to history
        
        Args:
            symbol: Trading symbol
            unified_prediction: Combined prediction
            all_predictions: Individual model predictions
        """
        # Initialize symbol history if needed
        if symbol not in self.prediction_history:
            self.prediction_history[symbol] = []
        
        # Create history entry
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "unified": unified_prediction,
            "models": {
                model_name: {
                    "direction": pred.get("direction", {}),
                    "confidence": pred.get("confidence", {}),
                    "score": pred.get("score", None)
                } for model_name, pred in all_predictions.items()
            }
        }
        
        # Add to history
        self.prediction_history[symbol].append(history_entry)
        
        # Save updated history
        self._save_prediction_history()
    
    def update_model_performance(self, symbol: str, prediction_id: str, 
                              actual_outcome: str, pnl: float) -> None:
        """
        Update model performance based on actual outcomes
        
        Args:
            symbol: Trading symbol
            prediction_id: Timestamp of the prediction
            actual_outcome: Actual market outcome ("BULLISH", "BEARISH", "NEUTRAL")
            pnl: Profit/loss from trade based on prediction
        """
        # Find the prediction in history
        if symbol not in self.prediction_history:
            logger.warning(f"No prediction history for symbol {symbol}")
            return
        
        # Find prediction by timestamp
        prediction_entry = None
        for entry in self.prediction_history[symbol]:
            if entry["timestamp"] == prediction_id:
                prediction_entry = entry
                break
        
        if not prediction_entry:
            logger.warning(f"Prediction {prediction_id} not found for {symbol}")
            return
        
        # Update prediction with outcome
        prediction_entry["actual_outcome"] = actual_outcome
        prediction_entry["pnl"] = pnl
        
        # Calculate success/failure for each model
        for model_name, model_prediction in prediction_entry["models"].items():
            # Get predicted direction from the model
            direction = "NEUTRAL"
            
            if "direction" in model_prediction:
                # LSTM model might have multiple horizons
                if isinstance(model_prediction["direction"], dict):
                    # Use shortest horizon
                    for horizon in ["3h", "1h", "short"]:
                        if horizon in model_prediction["direction"]:
                            direction = model_prediction["direction"][horizon]
                            break
                else:
                    direction = model_prediction["direction"]
            
            # Compare with actual outcome
            correct = direction == actual_outcome
            
            # Update model performance metrics
            if model_name not in self.performance_metrics:
                self.performance_metrics[model_name] = {
                    "correct": 0,
                    "incorrect": 0,
                    "accuracy": 0.0,
                    "total_pnl": 0.0,
                    "positive_trades": 0,
                    "negative_trades": 0
                }
            
            if correct:
                self.performance_metrics[model_name]["correct"] += 1
            else:
                self.performance_metrics[model_name]["incorrect"] += 1
            
            total = (self.performance_metrics[model_name]["correct"] + 
                    self.performance_metrics[model_name]["incorrect"])
            
            if total > 0:
                self.performance_metrics[model_name]["accuracy"] = (
                    self.performance_metrics[model_name]["correct"] / total
                )
            
            # Update PnL metrics
            self.performance_metrics[model_name]["total_pnl"] += pnl
            
            if pnl > 0:
                self.performance_metrics[model_name]["positive_trades"] += 1
            elif pnl < 0:
                self.performance_metrics[model_name]["negative_trades"] += 1
        
        # Adjust ensemble weights based on performance
        self._update_ensemble_weights()
        
        # Save updated history
        self._save_prediction_history()
    
    def _update_ensemble_weights(self):
        """Update ensemble weights based on model performance"""
        # Only update if we have performance metrics for at least 2 models
        if len(self.performance_metrics) < 2:
            return
        
        # Calculate weights based on accuracy
        total_accuracy = sum(
            metrics["accuracy"] for metrics in self.performance_metrics.values()
        )
        
        if total_accuracy == 0:
            return
        
        # Assign weights proportional to accuracy
        new_weights = {}
        for model_name, metrics in self.performance_metrics.items():
            new_weights[model_name] = metrics["accuracy"] / total_accuracy
        
        # Ensure we have weights for all expected prediction sources
        for source in [s.value for s in PredictionSource]:
            if source not in new_weights:
                new_weights[source] = 0.0
        
        # Update weights (with some momentum from previous weights)
        for source, weight in new_weights.items():
            old_weight = self.ensemble_weights.get(source, 0.0)
            self.ensemble_weights[source] = old_weight * 0.7 + weight * 0.3
        
        logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for all models
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "models": self.performance_metrics,
            "ensemble_weights": self.ensemble_weights,
            "prediction_counts": {
                symbol: len(history) for symbol, history in self.prediction_history.items()
            }
        }
