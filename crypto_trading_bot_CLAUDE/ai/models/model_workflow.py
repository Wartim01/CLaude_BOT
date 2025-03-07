"""
Module defining the AI model training and inference workflow
Orchestrates the entire model lifecycle from data preparation to deployment
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

from config.config import DATA_DIR
from utils.logger import setup_logger
from ai.models.feature_engineering import FeatureEngineering
from ai.models.lstm_model import EnhancedLSTMModel, LSTMModel
from ai.models.model_trainer import ModelTrainer
from ai.models.continuous_learning import AdvancedContinuousLearning

logger = setup_logger("model_workflow")

class ModelWorkflow:
    """
    Manages the AI model lifecycle including training, evaluation,
    continuous learning, and inference
    """
    def __init__(self, use_enhanced_model: bool = True):
        """
        Initialize the model workflow
        
        Args:
            use_enhanced_model: Whether to use the enhanced LSTM model
        """
        self.use_enhanced_model = use_enhanced_model
        
        # Create component instances
        self.feature_engineering = FeatureEngineering(save_scalers=True)
        
        # Initialize model
        if use_enhanced_model:
            self.model = EnhancedLSTMModel(
                input_length=60,
                feature_dim=30,
                lstm_units=[128, 96, 64],
                dropout_rate=0.3,
                learning_rate=0.0005,
                use_attention=True,
                attention_heads=8,
                use_residual=True,
                prediction_horizons=[
                    (12, "3h", True),
                    (48, "12h", True),
                    (192, "48h", True)
                ]
            )
        else:
            self.model = LSTMModel(
                input_length=60,
                feature_dim=30,
                lstm_units=[128, 64, 32],
                dropout_rate=0.3,
                use_attention=True,
                use_residual=True
            )
        
        # Initialize model trainer
        self.model_trainer = ModelTrainer()
        
        # Initialize continuous learning system
        self.continuous_learning = AdvancedContinuousLearning(
            model=self.model,
            feature_engineering=self.feature_engineering
        )
        
        # Model performance metrics
        self.performance = {}
        
        # Model paths
        self.models_dir = os.path.join(DATA_DIR, "models", "production")
        self.enhanced_model_path = os.path.join(self.models_dir, "enhanced_lstm_model.h5")
        self.standard_model_path = os.path.join(self.models_dir, "lstm_model.h5")
        
        # Create directory if needed
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model state
        self.is_trained = False
        self.last_training_time = None
        self.last_inference_time = None
        self.inference_count = 0
    
    def load_model(self) -> bool:
        """
        Loads the saved model from disk
        
        Returns:
            Whether the model was successfully loaded
        """
        try:
            # Determine which model path to use
            model_path = self.enhanced_model_path if self.use_enhanced_model else self.standard_model_path
            
            if os.path.exists(model_path):
                self.model.load(model_path)
                logger.info(f"Model loaded from {model_path}")
                self.is_trained = True
                return True
            else:
                logger.warning(f"Model file not found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def train_model(self, training_data: pd.DataFrame, validation_data: pd.DataFrame = None,
                   epochs: int = 100, batch_size: int = 32, symbol: str = None) -> Dict:
        """
        Trains the model on the provided data
        
        Args:
            training_data: DataFrame with training data
            validation_data: Optional DataFrame with validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            symbol: Symbol being trained on
            
        Returns:
            Training results
        """
        try:
            # Train using the model trainer
            if self.use_enhanced_model:
                training_results = self.model.train(
                    train_data=training_data,
                    feature_engineering=self.feature_engineering,
                    validation_data=validation_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    symbol=symbol
                )
            else:
                # Use model trainer for standard LSTM model
                self.model_trainer.model = self.model
                self.model_trainer.feature_engineering = self.feature_engineering
                
                training_results = self.model_trainer.train_final_model(
                    training_data,
                    epochs=epochs,
                    batch_size=batch_size
                )
            
            # Save the model
            model_path = self.enhanced_model_path if self.use_enhanced_model else self.standard_model_path
            self.model.save(model_path)
            
            # Update state
            self.is_trained = True
            self.last_training_time = datetime.now().isoformat()
            
            logger.info(f"Model training completed and saved to {model_path}")
            
            return {
                "success": True,
                "model_type": "enhanced" if self.use_enhanced_model else "standard",
                "training_time": self.last_training_time,
                "training_results": training_results
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Generate predictions using the trained model
        
        Args:
            data: DataFrame with market data for prediction
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            loaded = self.load_model()
            if not loaded:
                return {
                    "success": False,
                    "error": "Model not trained and could not be loaded"
                }
        
        try:
            # Generate predictions
            if self.use_enhanced_model:
                predictions = self.model.predict(data, self.feature_engineering)
            else:
                # Preprocess data for standard model
                X, _ = self.feature_engineering.prepare_lstm_data(
                    data, 
                    sequence_length=60, 
                    is_training=False
                )
                
                # Make predictions
                raw_predictions = self.model.model.predict(X)
                
                # Format predictions
                horizons = ["short", "medium", "long"]
                predictions = {}
                
                for i, horizon in enumerate(horizons):
                    if i < len(raw_predictions):
                        predictions[horizon] = {
                            "direction": "BULLISH" if raw_predictions[i][-1][0] > 0.5 else "BEARISH",
                            "direction_probability": float(raw_predictions[i][-1][0] * 100),
                            "confidence": float(abs(raw_predictions[i][-1][0] - 0.5) * 2),
                            "prediction_timestamp": datetime.now().isoformat()
                        }
            
            # Update state
            self.last_inference_time = datetime.now().isoformat()
            self.inference_count += 1
            
            return {
                "success": True,
                "predictions": predictions,
                "timestamp": self.last_inference_time,
                "model_type": "enhanced" if self.use_enhanced_model else "standard"
            }
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_model_with_new_data(self, new_data: pd.DataFrame, 
                               prediction_errors: List[float] = None) -> Dict:
        """
        Updates the model with new data using continuous learning
        
        Args:
            new_data: New market data for model adaptation
            prediction_errors: Optional list of prediction errors
            
        Returns:
            Update results
        """
        if not self.is_trained:
            return {
                "success": False,
                "error": "Cannot update model that has not been trained"
            }
        
        try:
            # Process new data using continuous learning
            update_result = self.continuous_learning.process_new_data(
                data=new_data,
                prediction_errors=prediction_errors
            )
            
            if update_result["updated"]:
                # Model was updated, update state
                self.last_training_time = datetime.now().isoformat()
            
            return update_result
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def evaluate_model_performance(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluates model performance on test data
        
        Args:
            test_data: Test data for model evaluation
            
        Returns:
            Evaluation results
        """
        if not self.is_trained:
            loaded = self.load_model()
            if not loaded:
                return {
                    "success": False,
                    "error": "Model not trained and could not be loaded"
                }
        
        try:
            # Preprocess data
            X, y = self.feature_engineering.prepare_lstm_data(
                test_data, 
                sequence_length=60,
                is_training=True
            )
            
            # Evaluate model
            if self.use_enhanced_model:
                # For enhanced model, we need to handle multiple outputs
                evaluation = self.model.model.evaluate(X, y, verbose=0)
                
                # Prepare evaluation metrics
                metrics = {}
                if isinstance(evaluation, list):
                    # For multi-output models, we get multiple metrics
                    loss_values = evaluation[:len(evaluation)//2]
                    accuracy_values = evaluation[len(evaluation)//2:]
                    
                    metrics["avg_loss"] = float(np.mean(loss_values))
                    metrics["avg_accuracy"] = float(np.mean(accuracy_values))
                else:
                    metrics["loss"] = float(evaluation)
            else:
                # For standard model
                evaluation = self.model.model.evaluate(X, y, verbose=0)
                metrics = {
                    "loss": float(evaluation[0]) if isinstance(evaluation, list) else float(evaluation),
                    "accuracy": float(evaluation[1]) if isinstance(evaluation, list) and len(evaluation) > 1 else None
                }
            
            # Make predictions for additional metrics
            y_pred = self.model.model.predict(X)
            
            if isinstance(y_pred, list):
                # For multi-output models
                directions_true = y[0].flatten() if isinstance(y, list) else y.flatten()
                directions_pred = y_pred[0].flatten() > 0.5
                
                # Calculate classification metrics
                true_positives = np.sum((directions_true == 1) & (directions_pred == 1))
                false_positives = np.sum((directions_true == 0) & (directions_pred == 1))
                true_negatives = np.sum((directions_true == 0) & (directions_pred == 0))
                false_negatives = np.sum((directions_true == 1) & (directions_pred == 0))
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics.update({
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1_score)
                })
            
            # Store performance metrics
            self.performance = {
                **metrics,
                "evaluation_time": datetime.now().isoformat(),
                "test_samples": len(X)
            }
            
            return {
                "success": True,
                "metrics": metrics,
                "model_type": "enhanced" if self.use_enhanced_model else "standard",
                "evaluation_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_model_status(self) -> Dict:
        """
        Get the current model status and metadata
        
        Returns:
            Model status information
        """
        return {
            "model_type": "enhanced" if self.use_enhanced_model else "standard",
            "is_trained": self.is_trained,
            "last_training_time": self.last_training_time,
            "last_inference_time": self.last_inference_time,
            "inference_count": self.inference_count,
            "performance_metrics": self.performance,
            "continuous_learning": {
                "enabled": self.continuous_learning.learning_enabled,
                "total_updates": self.continuous_learning.total_updates,
                "last_update": self.continuous_learning.last_update_time
            },
            "timestamp": datetime.now().isoformat()
        }