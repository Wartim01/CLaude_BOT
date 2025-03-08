"""
Main dashboard for the trading bot system
Provides a central interface for monitoring and controlling all aspects of the bot
"""
import os
import sys
import threading
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR
from core.risk_manager import RiskManager
from core.position_tracker import PositionTracker
from utils.market_risk_feed import MarketRiskFeed
from utils.logger import setup_logger

logger = setup_logger("main_dashboard")

class MainDashboard:
    """
    Main dashboard for monitoring and controlling the trading bot
    """
    def __init__(self, 
                 risk_manager=None, 
                 position_tracker=None, 
                 market_risk_feed=None,
                 port: int = 8051):
        """
        Initialize the main dashboard
        
        Args:
            risk_manager: Risk manager instance
            position_tracker: Position tracker instance
            market_risk_feed: Market risk feed instance
            port: Port to run the dashboard on
        """
        self.risk_manager = risk_manager
        self.position_tracker = position_tracker
        self.market_risk_feed = market_risk_feed
        self.port = port
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.app.title = "Crypto Trading Bot Dashboard"
        
        # Dashboard state
        self.last_refresh = datetime.datetime.now()
        self.refresh_interval = 60  # seconds
        self.is_running = False
        self.server_thread = None
        
        # Setup layout
        self._setup_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
        logger.info("Main dashboard initialized")
    
    def _setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Crypto Trading Bot Dashboard", style={'textAlign': 'center'}),
                html.Div([
                    html.Button("Refresh Data", id="refresh-button", n_clicks=0),
                    html.Div(id="last-refresh-time", style={"marginLeft": "20px"})
                ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"}),
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
            
            # Tabs for different sections
            dcc.Tabs([
                # Overview Tab
                dcc.Tab(label="Overview", children=[
                    html.Div([
                        html.H2("System Status"),
                        html.Div(id="system-status"),
                        
                        html.Div([
                            html.Div([
                                html.H3("Account Summary"),
                                html.Div(id="account-summary")
                            ], className="six columns"),
                            
                            html.Div([
                                html.H3("Market Risk"),
                                html.Div(id="market-risk-summary")
                            ], className="six columns"),
                        ], className="row"),
                        
                        html.H3("Open Positions"),
                        html.Div(id="positions-table")
                    ])
                ]),
                
                # Positions Tab
                dcc.Tab(label="Positions", children=[
                    html.Div([
                        html.H2("Position Management"),
                        html.Div([
                            html.Button("Close All Positions", id="close-all-button", n_clicks=0),
                            html.Div(id="close-all-result", style={"marginLeft": "20px"})
                        ], style={"display": "flex", "alignItems": "center"}),
                        
                        html.H3("Position Details"),
                        html.Div(id="detailed-positions-table"),
                        
                        html.H3("Position History"),
                        html.Div(id="position-history-table")
                    ])
                ]),
                
                # Risk Management Tab
                dcc.Tab(label="Risk Management", children=[
                    html.Div([
                        html.H2("Risk Management Settings"),
                        html.Div([
                            html.Label("Maximum Position Size (USD):"),
                            dcc.Input(id="max-position-size", type="number", min=10, step=10),
                            html.Button("Update", id="update-risk-settings", n_clicks=0),
                            html.Div(id="risk-settings-result", style={"marginLeft": "20px"})
                        ]),
                        
                        html.H3("Risk Analysis"),
                        html.Div(id="risk-analysis"),
                        
                        html.H3("Market Risk Heatmap"),
                        dcc.Graph(id="risk-heatmap")
                    ])
                ]),
                
                # Performance Tab
                dcc.Tab(label="Performance", children=[
                    html.Div([
                        html.H2("Performance Metrics"),
                        
                        html.Div([
                            html.Div([
                                html.H3("Trading Performance"),
                                dcc.Graph(id="pnl-chart")
                            ], className="six columns"),
                            
                            html.Div([
                                html.H3("Win/Loss Metrics"),
                                html.Div(id="performance-metrics")
                            ], className="six columns"),
                        ], className="row"),
                        
                        html.H3("Trade History"),
                        html.Div(id="trade-history-table")
                    ])
                ]),
                
                # Settings Tab
                dcc.Tab(label="Settings", children=[
                    html.Div([
                        html.H2("Bot Settings"),
                        html.Div([
                            html.Label("Trading Enabled:"),
                            dcc.RadioItems(
                                id="trading-enabled",
                                options=[
                                    {"label": "Enabled", "value": "true"},
                                    {"label": "Disabled", "value": "false"}
                                ],
                                value="false"
                            ),
                            html.Button("Update", id="update-bot-settings", n_clicks=0),
                            html.Div(id="bot-settings-result", style={"marginLeft": "20px"})
                        ]),
                        
                        html.H3("Dashboard Settings"),
                        html.Div([
                            html.Label("Refresh Interval (seconds):"),
                            dcc.Input(id="refresh-interval", type="number", min=5, step=5, value=60),
                            html.Button("Update", id="update-dashboard-settings", n_clicks=0),
                            html.Div(id="dashboard-settings-result", style={"marginLeft": "20px"})
                        ])
                    ])
                ])
            ]),
            
            # Footer
            html.Div([
                html.Hr(),
                html.P("Crypto Trading Bot Dashboard - Â© 2023", style={"textAlign": "center"})
            ]),
            
            # Interval component for periodic refresh
            dcc.Interval(
                id="auto-refresh-interval",
                interval=self.refresh_interval * 1000,  # in milliseconds
                n_intervals=0
            ),
            
            # Store component to hold data
            dcc.Store(id="dashboard-data"),
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        # Callback to refresh data
        @self.app.callback(
            Output("dashboard-data", "data"),
            Output("last-refresh-time", "children"),
            [
                Input("refresh-button", "n_clicks"),
                Input("auto-refresh-interval", "n_intervals")
            ]
        )
        def refresh_data(n_clicks, n_intervals):
            # Collect data from all sources
            data = self._collect_dashboard_data()
            
            # Update last refresh time
            self.last_refresh = datetime.datetime.now()
            refresh_time = f"Last refreshed: {self.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}"
            
            return data, refresh_time
        
        # ... additional callbacks for updating each section of the dashboard ...
        
        # For now, we'll just implement a couple of example callbacks
        
        # Update system status
        @self.app.callback(
            Output("system-status", "children"),
            [Input("dashboard-data", "data")]
        )
        def update_system_status(data):
            if not data:
                return html.Div("No data available")
            
            system_status = data.get("system_status", {})
            
            return html.Div([
                html.Table([
                    html.Tr([
                        html.Th("Component"),
                        html.Th("Status"),
                        html.Th("Details")
                    ]),
                    html.Tr([
                        html.Td("Trading Bot"),
                        html.Td("Running" if system_status.get("bot_running", False) else "Stopped"),
                        html.Td(system_status.get("bot_uptime", "-"))
                    ]),
                    html.Tr([
                        html.Td("Market Risk"),
                        html.Td(system_status.get("market_risk_level", "Unknown")),
                        html.Td(f"Score: {system_status.get('market_risk_score', '-')}")
                    ]),
                    html.Tr([
                        html.Td("Open Positions"),
                        html.Td(str(system_status.get("open_positions_count", 0))),
                        html.Td(f"Value: ${system_status.get('open_positions_value', 0):.2f}")
                    ])
                ])
            ])
        
        # Update account summary
        @self.app.callback(
            Output("account-summary", "children"),
            [Input("dashboard-data", "data")]
        )
        def update_account_summary(data):
            if not data:
                return html.Div("No data available")
            
            account = data.get("account", {})
            
            return html.Div([
                html.Table([
                    html.Tr([
                        html.Th("Metric"),
                        html.Th("Value")
                    ]),
                    html.Tr([
                        html.Td("Total Balance"),
                        html.Td(f"${account.get('total_balance', 0):.2f}")
                    ]),
                    html.Tr([
                        html.Td("Available Balance"),
                        html.Td(f"${account.get('available_balance', 0):.2f}")
                    ]),
                    html.Tr([
                        html.Td("Daily P&L"),
                        html.Td(f"${account.get('daily_pnl', 0):.2f}")
                    ]),
                    html.Tr([
                        html.Td("Total P&L"),
                        html.Td(f"${account.get('total_pnl', 0):.2f}")
                    ])
                ])
            ])
    
    def _collect_dashboard_data(self) -> dict:
        """
        Collect data from all sources for the dashboard
        
        Returns:
            Dictionary with all dashboard data
        """
        data = {
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # System status data
        system_status = {
            "bot_running": True,  # Placeholder
            "bot_uptime": "3h 24m",  # Placeholder
            "market_risk_level": "Medium",
            "market_risk_score": 55,
            "open_positions_count": 0,
            "open_positions_value": 0
        }
        
        # Account data
        account = {
            "total_balance": 10000,  # Placeholder
            "available_balance": 8500,  # Placeholder
            "daily_pnl": 120,  # Placeholder
            "total_pnl": 450  # Placeholder
        }
        
        # Get data from risk manager if available
        if self.risk_manager:
            try:
                # Placeholder - in real implementation you'd get actual data
                pass
            except Exception as e:
                logger.error(f"Error getting risk manager data: {str(e)}")
        
        # Get data from position tracker if available
        if self.position_tracker:
            try:
                # open_positions = self.position_tracker.get_all_open_positions()
                # system_status["open_positions_count"] = sum(len(positions) for positions in open_positions.values())
                # system_status["open_positions_value"] = sum(
                #     sum(p["entry_price"] * p["quantity"] for p in positions if isinstance(p, dict))
                #     for positions in open_positions.values()
                # )
                pass
            except Exception as e:
                logger.error(f"Error getting position tracker data: {str(e)}")
        
        # Get data from market risk feed if available
        if self.market_risk_feed:
            try:
                global_risk = self.market_risk_feed.get_global_risk()
                system_status["market_risk_level"] = global_risk.get("risk_level", "Unknown").capitalize()
                system_status["market_risk_score"] = global_risk.get("risk_score", 50)
            except Exception as e:
                logger.error(f"Error getting market risk data: {str(e)}")
        
        # Combine all data
        data["system_status"] = system_status
        data["account"] = account
        
        return data
    
    def start(self, background: bool = True, open_browser: bool = True):
        """
        Start the dashboard server
        
        Args:
            background: Run in background thread if True, blocking if False
            open_browser: Automatically open dashboard in browser
        """
        if background:
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.is_running = True
            logger.info(f"Main dashboard started in background at http://localhost:{self.port}")
            
            if open_browser:
                # Open in browser with a slight delay to ensure server is up
                def open_browser_delayed():
                    time.sleep(1.5)
                    import webbrowser
                    webbrowser.open(f"http://localhost:{self.port}")
                
                threading.Thread(target=open_browser_delayed).start()
        else:
            logger.info(f"Starting main dashboard at http://localhost:{self.port}")
            self.is_running = True
            self._run_server()
    
    def _run_server(self):
        """Run the Dash server"""
        try:
            self.app.run_server(debug=False, port=self.port, host="0.0.0.0")
        except Exception as e:
            logger.error(f"Error running dashboard server: {str(e)}")
            self.is_running = False
    
    def stop(self):
        """Stop the dashboard server"""
        if self.is_running:
            # Dash doesn't provide a clean way to stop the server
            # This is a workaround that's not ideal
            try:
                func = request.environ.get('werkzeug.server.shutdown')
                if func is None:
                    logger.warning("Dashboard server could not be stopped cleanly")
                else:
                    func()
            except:
                logger.warning("Error stopping dashboard server")
            
            self.is_running = False
            logger.info("Main dashboard stopped")

# For testing
if __name__ == "__main__":
    dashboard = MainDashboard()
    dashboard.start(background=False)
