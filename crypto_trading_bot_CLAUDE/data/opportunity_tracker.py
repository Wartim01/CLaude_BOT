"""
Opportunity tracking module for analyzing missed and executed opportunities
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from utils.logger import setup_logger
from config.config import DATA_DIR

logger = setup_logger("opportunity_tracker")

class OpportunityTracker:
    """
    Tracks and analyzes trading opportunities that were detected
    but may or may not have been executed based on threshold settings
    """
    def __init__(self):
        """Initialize the opportunity tracker"""
        self.data_dir = os.path.join(DATA_DIR, "opportunities")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.opportunities_file = os.path.join(self.data_dir, "opportunity_history.json")
        self.detected_opportunities = []
        self.executed_opportunities = []
        self.missed_opportunities = []
        
        # Load existing data if available
        self._load_opportunities()
    
    def _load_opportunities(self) -> None:
        """Load opportunity data from file"""
        if not os.path.exists(self.opportunities_file):
            logger.debug("No opportunity history file found. Starting fresh.")
            return
        
        try:
            with open(self.opportunities_file, 'r') as f:
                data = json.load(f)
            
            self.detected_opportunities = data.get('detected', [])
            self.executed_opportunities = data.get('executed', [])
            self.missed_opportunities = data.get('missed', [])
            
            logger.info(f"Loaded opportunity history: {len(self.detected_opportunities)} detected, "
                      f"{len(self.executed_opportunities)} executed, "
                      f"{len(self.missed_opportunities)} missed")
                      
        except Exception as e:
            logger.error(f"Error loading opportunity data: {str(e)}")
    
    def _save_opportunities(self) -> None:
        """Save opportunity data to file"""
        try:
            # Prune old entries to keep file size manageable
            max_entries = 1000  # Keep last 1000 entries of each type
            
            if len(self.detected_opportunities) > max_entries:
                self.detected_opportunities = self.detected_opportunities[-max_entries:]
                
            if len(self.executed_opportunities) > max_entries:
                self.executed_opportunities = self.executed_opportunities[-max_entries:]
                
            if len(self.missed_opportunities) > max_entries:
                self.missed_opportunities = self.missed_opportunities[-max_entries:]
            
            data = {
                'detected': self.detected_opportunities,
                'executed': self.executed_opportunities,
                'missed': self.missed_opportunities,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.opportunities_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.debug(f"Saved opportunity history: {len(self.detected_opportunities)} detected, "
                       f"{len(self.executed_opportunities)} executed, "
                       f"{len(self.missed_opportunities)} missed")
                      
        except Exception as e:
            logger.error(f"Error saving opportunity data: {str(e)}")
    
    def record_opportunity(self, opportunity: Dict, executed: bool = False, 
                           current_threshold: float = None, rejection_reason: str = None) -> None:
        """
        Record a trading opportunity for future analysis
        
        Args:
            opportunity: Opportunity details
            executed: Whether the opportunity was executed as a trade
            current_threshold: Current decision threshold at time of detection
            rejection_reason: Reason for rejection if not executed (new parameter)
        """
        # Create record with basic info
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": opportunity.get("symbol"),
            "direction": opportunity.get("direction", "unknown"),
            "score": opportunity.get("composite_score", 0),
            "executed": executed,
            "threshold": current_threshold
        }
        
        # Add rejection reason if provided
        if not executed and rejection_reason:
            record["rejection_reason"] = rejection_reason
        
        # Add to history
        self.detected_opportunities.append(record)
        
        if executed:
            self.executed_opportunities.append(record)
        else:
            record["threshold_gap"] = current_threshold - record["score"]
            self.missed_opportunities.append(record)
        
        # Save periodically (every 10 opportunities)
        if len(self.detected_opportunities) % 10 == 0:
            self._save_opportunities()
    
    def get_opportunity_stats(self, lookback_hours: int = 24) -> Dict:
        """
        Get statistics about detected, executed and missed opportunities
        
        Args:
            lookback_hours: Hours to look back
            
        Returns:
            Dictionary with opportunity statistics
        """
        lookback_time = datetime.now() - timedelta(hours=lookback_hours)
        
        # Filter opportunities within lookback period
        recent_detected = self._filter_by_time(self.detected_opportunities, lookback_time)
        recent_executed = self._filter_by_time(self.executed_opportunities, lookback_time)
        recent_missed = self._filter_by_time(self.missed_opportunities, lookback_time)
        
        # Calculate statistics
        total_opportunities = len(recent_detected)
        executed_count = len(recent_executed)
        missed_count = len(recent_missed)
        
        # Execution rate
        execution_rate = executed_count / total_opportunities if total_opportunities > 0 else 0
        
        # Average scores
        avg_detected_score = self._calculate_avg_score(recent_detected)
        avg_executed_score = self._calculate_avg_score(recent_executed)
        avg_missed_score = self._calculate_avg_score(recent_missed)
        
        # Count near-miss opportunities (within 5 points of threshold)
        near_misses = [
            op for op in recent_missed
            if op.get("threshold_gap", float('inf')) <= 5
        ]
        
        # Group by symbol
        symbols = {}
        for op in recent_detected:
            symbol = op.get("symbol", "unknown")
            if symbol not in symbols:
                symbols[symbol] = {"detected": 0, "executed": 0, "missed": 0}
            symbols[symbol]["detected"] += 1
            if op.get("executed", False):
                symbols[symbol]["executed"] += 1
            else:
                symbols[symbol]["missed"] += 1
        
        return {
            "lookback_hours": lookback_hours,
            "total_opportunities": total_opportunities,
            "executed_opportunities": executed_count,
            "missed_opportunities": missed_count,
            "execution_rate": execution_rate,
            "near_misses": len(near_misses),
            "avg_detected_score": avg_detected_score,
            "avg_executed_score": avg_executed_score,
            "avg_missed_score": avg_missed_score,
            "symbols": symbols,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_missed_opportunity_rate(self, lookback_hours: int = 24) -> float:
        """
        Calculate the rate of missed opportunities vs executed
        
        Args:
            lookback_hours: Hours to look back
            
        Returns:
            Percentage of missed opportunities
        """
        stats = self.get_opportunity_stats(lookback_hours)
        total = stats["total_opportunities"]
        
        if total == 0:
            return 0.0
            
        return stats["missed_opportunities"] / total * 100
    
    def should_adjust_threshold(self, lookback_hours: int = 24) -> Dict:
        """
        Analyze if threshold should be adjusted based on opportunity statistics
        
        Args:
            lookback_hours: Hours to look back
            
        Returns:
            Dictionary with adjustment recommendation
        """
        stats = self.get_opportunity_stats(lookback_hours)
        
        # Default response structure
        result = {
            "should_adjust": False,
            "direction": None,
            "reason": None,
            "confidence": 0.0
        }
        
        # If not enough data for decision, no adjustment
        if stats["total_opportunities"] < 5:
            result["reason"] = "Insufficient data for adjustment decision"
            return result
        
        # Check if we should lower threshold
        if stats["execution_rate"] < 0.2 and stats["near_misses"] >= 3:
            result["should_adjust"] = True
            result["direction"] = "lower"
            result["reason"] = f"Low execution rate ({stats['execution_rate']:.2%}) with {stats['near_misses']} near-miss opportunities"
            result["confidence"] = 0.7
            return result
        
        # Check if we should raise threshold (many low quality executions)
        if stats["execution_rate"] > 0.8 and stats["executed_opportunities"] > 10:
            result["should_adjust"] = True
            result["direction"] = "raise"
            result["reason"] = f"Very high execution rate ({stats['execution_rate']:.2%}) with many opportunities ({stats['executed_opportunities']})"
            result["confidence"] = 0.6
            return result
        
        # No adjustment needed
        result["reason"] = f"Current execution rate ({stats['execution_rate']:.2%}) is balanced"
        return result
        
    def _filter_by_time(self, opportunity_list: List[Dict], start_time: datetime) -> List[Dict]:
        """Filter opportunities by time"""
        filtered = []
        for op in opportunity_list:
            try:
                op_time = datetime.fromisoformat(op["timestamp"])
                if op_time >= start_time:
                    filtered.append(op)
            except (ValueError, KeyError):
                continue  # Skip entries with invalid timestamps
        return filtered
    
    def _calculate_avg_score(self, opportunity_list: List[Dict]) -> float:
        """Calculate average score from a list of opportunities"""
        if not opportunity_list:
            return 0.0
            
        scores = [op.get("score", 0) for op in opportunity_list]
        return sum(scores) / len(scores)
