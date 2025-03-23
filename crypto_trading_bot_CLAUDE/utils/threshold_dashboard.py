"""
Dashboard for visualizing adaptive threshold adjustments and opportunity statistics
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from config.config import DATA_DIR
from utils.logger import setup_logger
from data.opportunity_tracker import OpportunityTracker

logger = setup_logger("threshold_dashboard")

class AdaptiveThresholdDashboard:
    """
    Dashboard for visualizing and analyzing adaptive threshold system
    """
    
    def __init__(self, opportunity_tracker: Optional[OpportunityTracker] = None):
        """
        Initialize the dashboard
        
        Args:
            opportunity_tracker: OpportunityTracker instance (creates a new one if None)
        """
        self.output_dir = os.path.join(DATA_DIR, "dashboards")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.opportunity_tracker = opportunity_tracker or OpportunityTracker()
        
        # Load adjustment history
        self.adjustment_history = self._load_adjustment_history()
        
    def _load_adjustment_history(self) -> List[Dict]:
        """Load threshold adjustment history from log file"""
        history_file = os.path.join(DATA_DIR, "logs", "threshold_adjustments.log")
        
        if not os.path.exists(history_file):
            logger.warning(f"Threshold adjustment history file not found: {history_file}")
            return []
            
        try:
            adjustments = []
            with open(history_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(" | ")
                    if len(parts) >= 4:
                        timestamp_str = parts[0].strip()
                        direction = parts[1].strip().lower()
                        threshold_str = parts[2].strip()
                        reason = parts[3].strip()
                        
                        # Parse threshold values
                        old_value, new_value = None, None
                        try:
                            threshold_parts = threshold_str.split("->")
                            if len(threshold_parts) == 2:
                                old_value = float(threshold_parts[0].strip())
                                new_value = float(threshold_parts[1].strip())
                        except Exception:
                            pass
                            
                        # Parse timestamp
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str)
                        except Exception:
                            timestamp = None
                            
                        if timestamp and old_value is not None and new_value is not None:
                            adjustments.append({
                                "timestamp": timestamp,
                                "direction": direction,
                                "old_value": old_value,
                                "new_value": new_value,
                                "reason": reason
                            })
            
            logger.info(f"Loaded {len(adjustments)} threshold adjustments from history")
            return adjustments
            
        except Exception as e:
            logger.error(f"Error loading threshold adjustment history: {str(e)}")
            return []
    
    def generate_threshold_evolution_chart(self, days: int = 30, 
                                         output_file: Optional[str] = None) -> str:
        """
        Generate a chart showing threshold evolution over time
        
        Args:
            days: Number of days to include in the chart
            output_file: Output file path (generated from date if None)
            
        Returns:
            Path to the generated chart
        """
        if not self.adjustment_history:
            logger.warning("No adjustment history available for chart generation")
            return ""
            
        # Filter adjustments within time range
        start_date = datetime.now() - timedelta(days=days)
        adjustments = [adj for adj in self.adjustment_history 
                      if adj["timestamp"] >= start_date]
        
        if not adjustments:
            logger.warning(f"No adjustments found in the last {days} days")
            return ""
            
        # Extract data for plotting
        timestamps = [adj["timestamp"] for adj in adjustments]
        threshold_values = [adj["new_value"] for adj in adjustments]
        directions = [adj["direction"] for adj in adjustments]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot threshold evolution
        for i, (ts, value, direction) in enumerate(zip(timestamps, threshold_values, directions)):
            if i > 0:
                # Connect points with lines
                plt.plot([timestamps[i-1], ts], [threshold_values[i-1], value], 
                        'g-' if direction == 'increase' else 'r-', 
                        linewidth=1.5)
            
            # Plot points with direction-based colors
            marker_color = 'green' if direction == 'increase' else 'red'
            plt.scatter([ts], [value], color=marker_color, s=50)
            
            # Add annotations for significant changes (>= 3 points)
            if i > 0 and abs(value - threshold_values[i-1]) >= 3:
                plt.annotate(
                    f"{direction.title()} by {abs(value - threshold_values[i-1]):.1f}", 
                    xy=(ts, value),
                    xytext=(0, 10 if direction == "increase" else -20),
                    textcoords="offset points",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color=marker_color)
                )
        
        # Format plot
        plt.title("Threshold Evolution Over Time", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Threshold Value", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Add summary statistics
        if len(threshold_values) >= 2:
            start_value = threshold_values[0]
            end_value = threshold_values[-1]
            net_change = end_value - start_value
            
            plt.figtext(
                0.02, 0.02,
                f"Period: {days} days | Initial: {start_value:.1f} | Current: {end_value:.1f} | Net Change: {net_change:+.1f}",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
            )
        
        # Save the chart
        if output_file is None:
            now = datetime.now()
            output_file = os.path.join(
                self.output_dir, 
                f"threshold_evolution_{now.strftime('%Y%m%d_%H%M%S')}.png"
            )
            
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Generated threshold evolution chart: {output_file}")
        return output_file
    
    def generate_opportunity_stats_chart(self, days: int = 30, 
                                      output_file: Optional[str] = None) -> str:
        """
        Generate a chart showing opportunity statistics over time
        
        Args:
            days: Number of days to include in the chart
            output_file: Output file path (generated from date if None)
            
        Returns:
            Path to the generated chart
        """
        # Get opportunity stats data
        stats = self.opportunity_tracker.get_opportunity_stats(lookback_hours=days * 24)
        
        if stats["total_opportunities"] == 0:
            logger.warning(f"No opportunities found in the last {days} days")
            return ""
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Prepare data for pie chart
        labels = ['Executed', 'Missed']
        sizes = [stats["executed_opportunities"], stats["missed_opportunities"]]
        colors = ['#66b3ff', '#ff9999']
        explode = (0.1, 0)  # explode the executed slice
        
        # Plot pie chart of opportunities
        ax1.pie(
            sizes, 
            explode=explode, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            shadow=True
        )
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.set_title("Opportunity Execution Rate", fontsize=14)
        
        # Add central text with execution rate
        execution_rate = stats["executed_opportunities"] / stats["total_opportunities"] if stats["total_opportunities"] > 0 else 0
        ax1.text(
            0, 0, 
            f"Execution Rate\n{execution_rate:.1%}",
            ha='center',
            va='center',
            fontsize=14,
            bbox=dict(boxstyle="circle,pad=0.5", facecolor='white', alpha=0.8)
        )
        
        # Plot bar chart of scores
        score_data = [
            stats["avg_executed_score"] if stats["executed_opportunities"] > 0 else 0,
            stats["avg_missed_score"] if stats["missed_opportunities"] > 0 else 0,
            stats["avg_detected_score"],
            stats["current_threshold"]
        ]
        
        x_labels = ['Executed\nScores', 'Missed\nScores', 'All\nScores', 'Current\nThreshold']
        x_positions = range(len(score_data))
        bar_colors = ['green', 'red', 'blue', 'purple']
        
        bars = ax2.bar(x_positions, score_data, color=bar_colors, width=0.6)
        
        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(
                f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom'
            )
        
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(x_labels)
        ax2.set_ylabel("Score Value", fontsize=12)
        ax2.set_title("Opportunity Scores vs Threshold", fontsize=14)
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add text summary
        text = (
            f"Total Opportunities: {stats['total_opportunities']}\n"
            f"Executed: {stats['executed_opportunities']}\n"
            f"Missed: {stats['missed_opportunities']}\n"
            f"Near Misses: {stats['near_misses']}\n"
            f"Period: Last {days} days"
        )
        
        plt.figtext(
            0.02, 0.02, 
            text,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
        )
        
        # Save the chart
        if output_file is None:
            now = datetime.now()
            output_file = os.path.join(
                self.output_dir, 
                f"opportunity_stats_{now.strftime('%Y%m%d_%H%M%S')}.png"
            )
            
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Generated opportunity stats chart: {output_file}")
        return output_file
    
    def generate_full_dashboard(self, days: int = 30) -> Dict[str, str]:
        """
        Generate a complete dashboard with multiple charts
        
        Args:
            days: Number of days to include in the charts
            
        Returns:
            Dictionary with paths to generated files
        """
        now = datetime.now()
        date_str = now.strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        dashboard_dir = os.path.join(self.output_dir, f"dashboard_{date_str}")
        os.makedirs(dashboard_dir, exist_ok=True)
        
        # Generate individual charts
        charts = {}
        
        threshold_chart = self.generate_threshold_evolution_chart(
            days=days,
            output_file=os.path.join(dashboard_dir, "threshold_evolution.png")
        )
        
        opportunity_chart = self.generate_opportunity_stats_chart(
            days=days,
            output_file=os.path.join(dashboard_dir, "opportunity_stats.png")
        )
        
        # Generate HTML dashboard
        html_path = os.path.join(dashboard_dir, "dashboard.html")
        self._generate_html_dashboard(
            threshold_chart=os.path.basename(threshold_chart) if threshold_chart else None,
            opportunity_chart=os.path.basename(opportunity_chart) if opportunity_chart else None,
            stats=self.opportunity_tracker.get_opportunity_stats(lookback_hours=days * 24),
            output_file=html_path,
            days=days
        )
        
        return {
            "dashboard_html": html_path,
            "threshold_chart": threshold_chart,
            "opportunity_chart": opportunity_chart,
            "dashboard_dir": dashboard_dir
        }
    
    def _generate_html_dashboard(self, threshold_chart: Optional[str],
                              opportunity_chart: Optional[str],
                              stats: Dict,
                              output_file: str,
                              days: int) -> None:
        """Generate HTML dashboard page"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Adaptive Threshold System Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #4a6fa5;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .chart-container {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                    padding: 15px;
                }}
                .stats-panel {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                    margin-bottom: 20px;
                }}
                .stat-card {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 15px;
                    flex: 1 1 200px;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4a6fa5;
                }}
                .stat-label {{
                    color: #666;
                    font-size: 14px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
                .footer {{
                    font-size: 12px;
                    color: #666;
                    text-align: center;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Adaptive Threshold System Dashboard</h1>
                <p>Period: Last {days} days • Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            
            <div class="stats-panel">
                <div class="stat-card">
                    <div class="stat-value">{stats["total_opportunities"]}</div>
                    <div class="stat-label">Total Opportunities</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats["executed_opportunities"]}</div>
                    <div class="stat-label">Executed Opportunities</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats["missed_opportunities"]}</div>
                    <div class="stat-label">Missed Opportunities</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats["near_misses"]}</div>
                    <div class="stat-label">Near Misses</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats["current_threshold"]:.1f}</div>
                    <div class="stat-label">Current Threshold</div>
                </div>
            </div>
        """
        
        # Add threshold evolution chart if available
        if threshold_chart:
            html_content += f"""
            <div class="chart-container">
                <h2>Threshold Evolution</h2>
                <img src="{threshold_chart}" alt="Threshold Evolution">
            </div>
            """
            
        # Add opportunity stats chart if available
        if opportunity_chart:
            html_content += f"""
            <div class="chart-container">
                <h2>Opportunity Statistics</h2>
                <img src="{opportunity_chart}" alt="Opportunity Statistics">
            </div>
            """
            
        # Add adjustment history table if available
        if self.adjustment_history:
            recent_adjustments = sorted(
                self.adjustment_history, 
                key=lambda x: x["timestamp"],
                reverse=True
            )[:10]  # Show 10 most recent
            
            html_content += """
            <div class="chart-container">
                <h2>Recent Threshold Adjustments</h2>
                <table style="width:100%; border-collapse: collapse;">
                    <tr style="background-color:#4a6fa5; color:white;">
                        <th style="padding:8px; text-align:left;">Date</th>
                        <th style="padding:8px; text-align:left;">Direction</th>
                        <th style="padding:8px; text-align:left;">Change</th>
                        <th style="padding:8px; text-align:left;">Reason</th>
                    </tr>
            """
            
            for i, adj in enumerate(recent_adjustments):
                bg_color = "#f9f9f9" if i % 2 == 0 else "white"
                direction_color = "green" if adj["direction"] == "increase" else "red"
                change = adj["new_value"] - adj["old_value"]
                
                html_content += f"""
                <tr style="background-color:{bg_color};">
                    <td style="padding:8px;">{adj["timestamp"].strftime('%Y-%m-%d %H:%M')}</td>
                    <td style="padding:8px; color:{direction_color};">{adj["direction"].upper()}</td>
                    <td style="padding:8px;">{adj["old_value"]:.1f} → {adj["new_value"]:.1f} ({change:+.1f})</td>
                    <td style="padding:8px;">{adj["reason"]}</td>
                </tr>
                """
                
            html_content += """
                </table>
            </div>
            """
            
        # Add footer
        html_content += """
            <div class="footer">
                <p>Generated by Adaptive Threshold Dashboard • Cryptocurrency Trading Bot</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML dashboard: {output_file}")
