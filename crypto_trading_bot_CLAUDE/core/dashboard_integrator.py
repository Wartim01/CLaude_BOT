"""
Module for integrating various dashboards and providing a unified interface
for monitoring and controlling the trading system
"""
import os
import time
import threading
from typing import Dict, List, Optional, Tuple, Union, Any
import webbrowser
from datetime import datetime

from config.config import DATA_DIR
from utils.logger import setup_logger
from ui.correlation_dashboard import CorrelationDashboard
from core.risk_manager import RiskManager
from core.position_tracker import PositionTracker
from utils.market_risk_feed import MarketRiskFeed
from utils.correlation_matrix import CorrelationMatrix

logger = setup_logger("dashboard_integrator")

class DashboardIntegrator:
    """
    Integrates various dashboards and monitoring components
    Provides utilities for starting, stopping, and managing all UI components
    """
    def __init__(self, 
                 risk_manager: Optional[RiskManager] = None,
                 position_tracker: Optional[PositionTracker] = None,
                 correlation_matrix: Optional[CorrelationMatrix] = None,
                 market_risk_feed: Optional[MarketRiskFeed] = None):
        """
        Initialize the dashboard integrator
        
        Args:
            risk_manager: Risk manager instance
            position_tracker: Position tracker instance
            correlation_matrix: Correlation matrix instance
            market_risk_feed: Market risk feed instance
        """
        self.risk_manager = risk_manager
        self.position_tracker = position_tracker
        self.correlation_matrix = correlation_matrix
        self.market_risk_feed = market_risk_feed

        # Create components if not provided
        if self.correlation_matrix is None:
            self.correlation_matrix = CorrelationMatrix()
            logger.info("Created new correlation matrix")
            
        # Dashboard components
        self.correlation_dashboard = None
        self.main_dashboard = None
        self.active_dashboards = {}
        
        # Configuration
        self.dashboard_ports = {
            "correlation": 8050,
            "main": 8051,
            "risk": 8052,
            "performance": 8053
        }
        
        # Status
        self.is_running = False
        self.auto_update_thread = None
        self.auto_update_interval = 900  # 15 minutes
        self.should_stop = False
        
        # Create output directory
        self.output_dir = os.path.join(DATA_DIR, "dashboards")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Dashboard integrator initialized")
    
    def start_all_dashboards(self, open_browser: bool = True) -> Dict:
        """
        Starts all available dashboard components
        
        Args:
            open_browser: Whether to automatically open dashboards in the browser
            
        Returns:
            Status dictionary for all dashboards
        """
        results = {}
        
        # Start correlation dashboard if correlation matrix is available
        if self.correlation_matrix is not None:
            corr_result = self.start_correlation_dashboard(open_browser=open_browser)
            results["correlation"] = corr_result
        
        # Start other dashboards as needed
        # ...
        
        # Start auto-update thread
        self.start_auto_updates()
        
        self.is_running = True
        logger.info("All dashboards started")
        
        return results
    
    def start_correlation_dashboard(self, port: int = None, open_browser: bool = True) -> Dict:
        """
        Starts the correlation dashboard
        
        Args:
            port: Port to run the dashboard on (uses default if None)
            open_browser: Whether to automatically open in browser
            
        Returns:
            Result dictionary
        """
        if self.correlation_dashboard is not None:
            return {"success": False, "message": "Correlation dashboard already running"}
        
        if self.correlation_matrix is None:
            return {"success": False, "message": "Correlation matrix not available"}
        
        try:
            # Use specified port or default
            dashboard_port = port or self.dashboard_ports["correlation"]
            
            # Create and start dashboard
            self.correlation_dashboard = CorrelationDashboard(
                correlation_matrix=self.correlation_matrix,
                port=dashboard_port
            )
            
            # Start in background thread
            self.correlation_dashboard.start(background=True)
            
            # Register in active dashboards
            self.active_dashboards["correlation"] = {
                "instance": self.correlation_dashboard,
                "port": dashboard_port,
                "url": f"http://localhost:{dashboard_port}",
                "start_time": datetime.now().isoformat()
            }
            
            # Open in browser if requested
            if open_browser:
                webbrowser.open(f"http://localhost:{dashboard_port}")
            
            logger.info(f"Correlation dashboard started on port {dashboard_port}")
            
            return {
                "success": True,
                "port": dashboard_port,
                "url": f"http://localhost:{dashboard_port}",
                "message": "Correlation dashboard started successfully"
            }
            
        except Exception as e:
            logger.error(f"Error starting correlation dashboard: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def stop_all_dashboards(self) -> Dict:
        """
        Stops all running dashboards
        
        Returns:
            Status dictionary
        """
        results = {}
        
        # Stop correlation dashboard if running
        if self.correlation_dashboard is not None:
            corr_result = self.stop_correlation_dashboard()
            results["correlation"] = corr_result
        
        # Stop other dashboards as needed
        # ...
        
        # Stop auto-update thread
        self.stop_auto_updates()
        
        self.is_running = False
        logger.info("All dashboards stopped")
        
        return results
    
    def stop_correlation_dashboard(self) -> Dict:
        """
        Stops the correlation dashboard
        
        Returns:
            Result dictionary
        """
        if self.correlation_dashboard is None:
            return {"success": False, "message": "Correlation dashboard not running"}
        
        try:
            # Stop the dashboard
            self.correlation_dashboard.stop()
            
            # Remove from active dashboards
            if "correlation" in self.active_dashboards:
                del self.active_dashboards["correlation"]
            
            # Clear reference
            self.correlation_dashboard = None
            
            logger.info("Correlation dashboard stopped")
            
            return {"success": True, "message": "Correlation dashboard stopped successfully"}
            
        except Exception as e:
            logger.error(f"Error stopping correlation dashboard: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def refresh_data(self) -> Dict:
        """
        Refreshes all data sources for dashboards
        
        Returns:
            Status dictionary
        """
        results = {}
        
        # Refresh market risk data if available
        if self.market_risk_feed is not None:
            try:
                start_time = time.time()
                risk_result = self.market_risk_feed.refresh()
                elapsed = time.time() - start_time
                
                results["market_risk"] = {
                    "success": True,
                    "refresh_time": elapsed,
                    "global_risk": risk_result.get("global_risk", {})
                }
                
                logger.info(f"Market risk data refreshed in {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"Error refreshing market risk data: {str(e)}")
                results["market_risk"] = {"success": False, "error": str(e)}
        
        # Refresh correlation data if available
        if self.correlation_matrix is not None and self.risk_manager is not None:
            try:
                # Get current market data
                if hasattr(self.risk_manager, 'update_correlation_data'):
                    # We'll need market data, which would typically be provided by a data fetcher
                    # For now, we'll use a placeholder. In real implementation, you'd get this from
                    # your data provider
                    market_data_dict = {}  # Placeholder, should be {symbol: dataframe}
                    
                    # Update correlation data through risk manager
                    self.risk_manager.update_correlation_data(market_data_dict)
                    results["correlation"] = {"success": True}
                    logger.info("Correlation data refreshed")
                else:
                    results["correlation"] = {
                        "success": False, 
                        "message": "Risk manager doesn't support correlation updates"
                    }
            except Exception as e:
                logger.error(f"Error refreshing correlation data: {str(e)}")
                results["correlation"] = {"success": False, "error": str(e)}
        
        return results
    
    def start_auto_updates(self) -> None:
        """
        Starts automatic data updates in background thread
        """
        if self.auto_update_thread is None or not self.auto_update_thread.is_alive():
            self.should_stop = False
            self.auto_update_thread = threading.Thread(target=self._auto_update_worker)
            self.auto_update_thread.daemon = True
            self.auto_update_thread.start()
            logger.info(f"Auto-updates started (interval: {self.auto_update_interval}s)")
    
    def stop_auto_updates(self) -> None:
        """
        Stops automatic data updates
        """
        self.should_stop = True
        if self.auto_update_thread and self.auto_update_thread.is_alive():
            self.auto_update_thread.join(timeout=2.0)
            logger.info("Auto-updates stopped")
    
    def _auto_update_worker(self) -> None:
        """Worker thread for automatic updates"""
        while not self.should_stop:
            try:
                # Refresh all data
                self.refresh_data()
                
                # Sleep for the specified interval
                time.sleep(self.auto_update_interval)
            except Exception as e:
                logger.error(f"Error in auto-update worker: {str(e)}")
                # Sleep a bit before retrying
                time.sleep(60)
    
    def get_all_dashboard_urls(self) -> Dict[str, str]:
        """
        Returns URLs for all active dashboards
        
        Returns:
            Dictionary of dashboard URLs
        """
        return {
            name: info["url"] 
            for name, info in self.active_dashboards.items()
        }
    
    def export_correlation_report(self, time_window: str = '7d', 
                               output_file: str = None) -> Dict:
        """
        Exports a correlation report to file
        
        Args:
            time_window: Time window for correlation data
            output_file: Output file path (generates default if None)
            
        Returns:
            Result dictionary
        """
        if self.correlation_matrix is None:
            return {"success": False, "message": "Correlation matrix not available"}
        
        try:
            # Generate default output file if not specified
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(
                    self.output_dir, 
                    f"correlation_report_{time_window}_{timestamp}.json"
                )
            
            # Generate the report
            report = self.correlation_matrix.generate_correlation_report(
                time_window=time_window,
                save_path=output_file
            )
            
            # Generate visualization too
            viz_path = os.path.splitext(output_file)[0] + ".png"
            self.correlation_matrix.visualize_matrix(
                time_window=time_window,
                save_path=viz_path,
                show_plot=False
            )
            
            logger.info(f"Correlation report exported to {output_file}")
            
            return {
                "success": True,
                "report_path": output_file,
                "visualization_path": viz_path,
                "high_correlations_count": len(report.get("high_correlations", [])),
                "negative_correlations_count": len(report.get("negative_correlations", [])),
                "average_correlation": report.get("average_correlation", 0)
            }
            
        except Exception as e:
            logger.error(f"Error exporting correlation report: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def export_risk_report(self, symbols: List[str] = None, 
                        output_file: str = None) -> Dict:
        """
        Exports a market risk report to file
        
        Args:
            symbols: List of symbols to include (all available if None)
            output_file: Output file path (generates default if None)
            
        Returns:
            Result dictionary
        """
        if self.market_risk_feed is None:
            return {"success": False, "message": "Market risk feed not available"}
        
        try:
            # Generate default output file if not specified
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(
                    self.output_dir, 
                    f"market_risk_report_{timestamp}.json"
                )
            
            # Get global risk
            global_risk = self.market_risk_feed.get_global_risk()
            
            # Get risk data for specified symbols
            symbol_risks = {}
            
            # If no symbols specified, use all available
            if symbols is None:
                symbols = list(self.market_risk_feed.risk_scores.keys())
            
            # Get risk data for each symbol
            for symbol in symbols:
                symbol_risks[symbol] = self.market_risk_feed.get_symbol_risk(symbol)
            
            # Create the report
            report = {
                "timestamp": datetime.now().isoformat(),
                "global_risk": global_risk,
                "symbol_risks": symbol_risks
            }
            
            # Save to file
            with open(output_file, 'w') as f:
                import json
                json.dump(report, f, indent=2)
            
            logger.info(f"Market risk report exported to {output_file}")
            
            return {
                "success": True,
                "report_path": output_file,
                "symbols_count": len(symbols),
                "global_risk_level": global_risk.get("risk_level", "unknown"),
                "global_risk_score": global_risk.get("risk_score", 0)
            }
            
        except Exception as e:
            logger.error(f"Error exporting market risk report: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_status(self) -> Dict:
        """
        Returns the status of all dashboard components
        
        Returns:
            Status dictionary
        """
        return {
            "is_running": self.is_running,
            "active_dashboards": list(self.active_dashboards.keys()),
            "auto_updates_enabled": self.auto_update_thread is not None and self.auto_update_thread.is_alive(),
            "auto_update_interval": self.auto_update_interval,
            "components_available": {
                "risk_manager": self.risk_manager is not None,
                "position_tracker": self.position_tracker is not None,
                "correlation_matrix": self.correlation_matrix is not None,
                "market_risk_feed": self.market_risk_feed is not None
            },
            "dashboard_ports": self.dashboard_ports
        }