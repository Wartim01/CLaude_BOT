"""
Dashboard component for visualizing correlation matrices and risk metrics
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import threading

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR
from utils.correlation_matrix import CorrelationMatrix
from utils.logger import setup_logger

logger = setup_logger("correlation_dashboard")

class CorrelationDashboard:
    """
    Interactive dashboard for visualizing correlation matrices and portfolio risk
    """
    def __init__(self, correlation_matrix: CorrelationMatrix = None, port: int = 8050):
        """
        Initialize the correlation dashboard
        
        Args:
            correlation_matrix: CorrelationMatrix instance
            port: Port to run the dashboard on
        """
        self.correlation_matrix = correlation_matrix or CorrelationMatrix()
        self.port = port
        self.app = dash.Dash(__name__)
        self.data_refresh_interval = 3600  # seconds
        self.last_refresh = datetime.now()
        
        # Setup layout
        self._setup_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Thread for the dashboard server
        self.server_thread = None
        self.is_running = False
    
    def _setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Crypto Correlation Dashboard", style={'textAlign': 'center'}),
            
            html.Div([
                html.Label("Time Window:"),
                dcc.Dropdown(
                    id='time-window-dropdown',
                    options=[
                        {'label': '1 Day', 'value': '1d'},
                        {'label': '1 Week', 'value': '7d'},
                        {'label': '1 Month', 'value': '30d'}
                    ],
                    value='7d'
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginBottom': 20}),
            
            html.Div([
                html.Label("Correlation Threshold:"),
                dcc.Slider(
                    id='correlation-threshold-slider',
                    min=0.5,
                    max=0.95,
                    step=0.05,
                    marks={i/100: f'{i/100}' for i in range(50, 100, 10)},
                    value=0.7
                )
            ], style={'width': '50%', 'marginBottom': 20}),
            
            html.Div([
                html.Button('Refresh Data', id='refresh-button', n_clicks=0),
                html.Div(id='last-refresh-time')
            ], style={'marginBottom': 20}),
            
            # Dashboard tabs
            dcc.Tabs([
                # Matrix Visualization Tab
                dcc.Tab(label="Correlation Matrix", children=[
                    html.Div([
                        dcc.Graph(id='correlation-heatmap')
                    ])
                ]),
                
                # High Correlation Pairs Tab
                dcc.Tab(label="High Correlation Pairs", children=[
                    html.Div([
                        dash_table.DataTable(
                            id='high-correlation-table',
                            columns=[
                                {'name': 'Symbol 1', 'id': 'symbol1'},
                                {'name': 'Symbol 2', 'id': 'symbol2'},
                                {'name': 'Correlation', 'id': 'correlation'}
                            ],
                            style_table={'overflowX': 'auto'},
                            style_data_conditional=[
                                {
                                    'if': {'column_id': 'correlation', 'filter_query': '{correlation} > 0.8'},
                                    'backgroundColor': '#FFA07A',
                                    'color': 'white'
                                }
                            ]
                        )
                    ])
                ]),
                
                # Diversification Opportunities Tab
                dcc.Tab(label="Diversification Opportunities", children=[
                    html.Div([
                        html.Label("Select Asset:"),
                        dcc.Dropdown(id='asset-dropdown'),
                        dcc.Graph(id='diversification-chart')
                    ])
                ]),
                
                # Portfolio Risk Analysis Tab
                dcc.Tab(label="Portfolio Risk Analysis", children=[
                    html.Div([
                        html.Label("Current Portfolio Weights:"),
                        html.Div(id='portfolio-input-container', children=[
                            # Will be populated dynamically based on available symbols
                        ]),
                        html.Button('Calculate Risk', id='calculate-risk-button', n_clicks=0),
                        html.Div(id='portfolio-risk-metrics'),
                        dcc.Graph(id='portfolio-risk-chart')
                    ])
                ])
            ]),
            
            # Store component for sharing data between callbacks
            dcc.Store(id='correlation-data'),
            dcc.Store(id='portfolio-data'),
            
            # Interval component for periodic refresh
            dcc.Interval(
                id='auto-refresh-interval',
                interval=self.data_refresh_interval * 1000,  # in milliseconds
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self):
        """Setup the dashboard callbacks"""
        # Callback to update correlation data
        @self.app.callback(
            Output('correlation-data', 'data'),
            Output('last-refresh-time', 'children'),
            Input('refresh-button', 'n_clicks'),
            Input('auto-refresh-interval', 'n_intervals'),
            Input('time-window-dropdown', 'value')
        )
        def update_correlation_data(n_clicks, n_intervals, time_window):
            # Check if we have data for this time window
            if time_window in self.correlation_matrix.matrices:
                matrix = self.correlation_matrix.matrices[time_window]
                
                if matrix is not None and not matrix.empty:
                    # Convert to a format that can be stored in dcc.Store
                    matrix_dict = matrix.to_dict()
                    
                    # Get high correlation pairs
                    high_corr_pairs = self.correlation_matrix.get_highly_correlated_pairs(
                        threshold=0.7,
                        time_window=time_window
                    )
                    
                    # Format the data for storage
                    data = {
                        'matrix': matrix_dict,
                        'symbols': list(matrix.index),
                        'high_corr_pairs': high_corr_pairs,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.last_refresh = datetime.now()
                    refresh_message = f"Last refreshed: {self.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}"
                    return data, refresh_message
            
            # If we don't have data, return empty
            return {}, f"No data available. Last refresh attempt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Callback to update heatmap
        @self.app.callback(
            Output('correlation-heatmap', 'figure'),
            Input('correlation-data', 'data'),
            Input('correlation-threshold-slider', 'value')
        )
        def update_heatmap(data, threshold):
            if not data or 'matrix' not in data:
                return go.Figure().update_layout(
                    title="No correlation data available",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                )
            
            # Convert back to DataFrame
            matrix = pd.DataFrame.from_dict(data['matrix'])
            
            # Create a mask for correlations below threshold (for visual emphasis)
            mask = (abs(matrix) < threshold) & (abs(matrix) > 0)
            
            # Create the heatmap
            fig = px.imshow(
                matrix,
                x=matrix.columns,
                y=matrix.columns,
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1],
                labels=dict(color="Correlation")
            )
            
            # Add text annotations
            annotations = []
            for i, row in enumerate(matrix.values):
                for j, val in enumerate(row):
                    if i != j:  # Skip diagonal
                        annotations.append(dict(
                            x=j, y=i,
                            text=f"{val:.2f}",
                            showarrow=False,
                            font=dict(
                                color='white' if abs(val) > 0.5 else 'black',
                                size=10
                            )
                        ))
            
            fig.update_layout(
                title=f"Correlation Matrix (Highlighting |r| > {threshold})",
                xaxis_title="Asset",
                yaxis_title="Asset",
                height=700,
                annotations=annotations
            )
            
            return fig
        
        # Callback to update high correlation table
        @self.app.callback(
            Output('high-correlation-table', 'data'),
            Input('correlation-data', 'data'),
            Input('correlation-threshold-slider', 'value')
        )
        def update_high_correlation_table(data, threshold):
            if not data or 'matrix' not in data:
                return []
            
            # Convert back to DataFrame
            matrix = pd.DataFrame.from_dict(data['matrix'])
            
            # Get pairs with correlation above threshold
            high_corr_pairs = []
            for i, symbol1 in enumerate(matrix.index):
                for j, symbol2 in enumerate(matrix.columns):
                    if i < j and abs(matrix.iloc[i, j]) >= threshold:
                        high_corr_pairs.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': float(matrix.iloc[i, j])
                        })
            
            # Sort by absolute correlation value (descending)
            high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # Format correlation values
            for pair in high_corr_pairs:
                pair['correlation'] = f"{pair['correlation']:.4f}"
            
            return high_corr_pairs
        
        # Callback to populate asset dropdown
        @self.app.callback(
            Output('asset-dropdown', 'options'),
            Output('asset-dropdown', 'value'),
            Input('correlation-data', 'data')
        )
        def update_asset_dropdown(data):
            if not data or 'symbols' not in data:
                return [], None
            
            symbols = data['symbols']
            options = [{'label': symbol, 'value': symbol} for symbol in symbols]
            default_value = symbols[0] if symbols else None
            return options, default_value
        
        # Callback to update diversification chart
        @self.app.callback(
            Output('diversification-chart', 'figure'),
            Input('correlation-data', 'data'),
            Input('asset-dropdown', 'value')
        )
        def update_diversification_chart(data, selected_asset):
            if not data or 'matrix' not in data or not selected_asset:
                return go.Figure().update_layout(
                    title="No data available for diversification analysis",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                )
            
            # Convert back to DataFrame
            matrix = pd.DataFrame.from_dict(data['matrix'])
            
            if selected_asset not in matrix.index:
                return go.Figure().update_layout(title=f"Asset {selected_asset} not found in correlation data")
            
            # Get correlations for the selected asset
            correlations = matrix[selected_asset].drop(selected_asset).sort_values()
            
            # Create a bar chart showing correlations
            fig = px.bar(
                x=correlations.index,
                y=correlations.values,
                labels={'x': 'Asset', 'y': 'Correlation'},
                color=correlations.values,
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1]
            )
            
            # Add reference lines
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=0.7,
                y1=0.7,
                line=dict(color="red", width=2, dash="dash"),
                xref="paper",
                yref="y"
            )
            
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=-0.7,
                y1=-0.7,
                line=dict(color="green", width=2, dash="dash"),
                xref="paper",
                yref="y"
            )
            
            fig.update_layout(
                title=f"Correlation of {selected_asset} with Other Assets",
                xaxis_title="Asset",
                yaxis_title="Correlation",
                yaxis=dict(range=[-1, 1]),
                height=500
            )
            
            return fig
        
        # Callback to setup portfolio input fields
        @self.app.callback(
            Output('portfolio-input-container', 'children'),
            Input('correlation-data', 'data')
        )
        def update_portfolio_inputs(data):
            if not data or 'symbols' not in data:
                return html.Div("No assets available for portfolio analysis")
            
            symbols = data['symbols']
            
            inputs = []
            for i, symbol in enumerate(symbols):
                inputs.append(
                    html.Div([
                        html.Label(symbol),
                        dcc.Input(
                            id=f'weight-{symbol}',
                            type='number',
                            placeholder=f'Weight for {symbol}',
                            min=0,
                            max=100,
                            step=0.1,
                            value=0 if i > 3 else (100/min(4, len(symbols)))  # Default equal weight for first 4 assets
                        )
                    ], style={'margin': '5px', 'display': 'inline-block'})
                )
            
            return inputs
        
        # Callback to calculate portfolio risk
        @self.app.callback(
            Output('portfolio-risk-metrics', 'children'),
            Output('portfolio-risk-chart', 'figure'),
            Output('portfolio-data', 'data'),
            Input('calculate-risk-button', 'n_clicks'),
            Input('correlation-data', 'data'),
            [Input(f'weight-{symbol}', 'value') for symbol in self.correlation_matrix.all_symbols]
        )
        def calculate_portfolio_risk(n_clicks, correlation_data, *weights):
            if not correlation_data or 'symbols' not in correlation_data:
                return (
                    "No correlation data available for portfolio analysis",
                    go.Figure(),
                    {}
                )
            
            symbols = correlation_data['symbols']
            
            # Build portfolio dictionary
            portfolio = {}
            for i, symbol in enumerate(symbols):
                if i < len(weights) and weights[i] is not None and weights[i] > 0:
                    portfolio[symbol] = weights[i]
            
            if not portfolio:
                return (
                    "Please enter weights for at least one asset",
                    go.Figure(),
                    {}
                )
            
            # Calculate risk metrics
            try:
                matrix = pd.DataFrame.from_dict(correlation_data['matrix'])
                
                # Normalize weights to sum to 1
                total_weight = sum(portfolio.values())
                normalized_portfolio = {k: v / total_weight for k, v in portfolio.items()}
                
                # Calculate weighted average correlation
                weighted_corr = 0
                total_weight_product = 0
                
                for i, (sym1, w1) in enumerate(normalized_portfolio.items()):
                    for j, (sym2, w2) in enumerate(normalized_portfolio.items()):
                        if sym1 != sym2:
                            corr = matrix.loc[sym1, sym2]
                            weight_product = w1 * w2
                            weighted_corr += corr * weight_product
                            total_weight_product += weight_product
                
                avg_correlation = weighted_corr / total_weight_product if total_weight_product > 0 else 0
                
                # Calculate diversification score
                diversification_score = max(0, 1 - abs(avg_correlation))
                
                # Determine risk level
                if diversification_score > 0.75:
                    risk_level = "Low"
                    risk_color = "green"
                elif diversification_score > 0.5:
                    risk_level = "Moderate"
                    risk_color = "orange"
                else:
                    risk_level = "High"
                    risk_color = "red"
                
                # Create output
                risk_metrics = html.Div([
                    html.H3("Portfolio Risk Analysis"),
                    html.Div([
                        html.P(f"Average Weighted Correlation: {avg_correlation:.4f}"),
                        html.P(f"Diversification Score: {diversification_score:.4f}"),
                        html.P(f"Risk Concentration Level: ", style={'display': 'inline'}),
                        html.Span(risk_level, style={'color': risk_color, 'fontWeight': 'bold'})
                    ])
                ])
                
                # Create chart for portfolio weights
                weight_fig = px.pie(
                    names=list(normalized_portfolio.keys()),
                    values=list(normalized_portfolio.values()),
                    title="Portfolio Allocation"
                )
                
                # Create network graph for correlations
                # (Not implemented in this simplified version)
                
                return (
                    risk_metrics,
                    weight_fig,
                    {
                        'portfolio': portfolio,
                        'avg_correlation': avg_correlation,
                        'diversification_score': diversification_score,
                        'risk_level': risk_level
                    }
                )
            except Exception as e:
                logger.error(f"Error calculating portfolio risk: {str(e)}")
                return (
                    f"Error calculating portfolio risk: {str(e)}",
                    go.Figure(),
                    {}
                )
    
    def start(self, background: bool = True):
        """
        Start the dashboard server
        
        Args:
            background: Run in background thread if True, blocking if False
        """
        if background:
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.is_running = True
            logger.info(f"Correlation dashboard started in background at http://localhost:{self.port}")
        else:
            logger.info(f"Starting correlation dashboard at http://localhost:{self.port}")
            self.is_running = True
            self._run_server()
    
    def _run_server(self):
        """Run the Dash server"""
        try:
            self.app.run_server(debug=False, port=self.port)
        except Exception as e:
            logger.error(f"Error running dashboard server: {str(e)}")
            self.is_running = False
    
    def stop(self):
        """Stop the dashboard server"""
        if self.is_running:
            # Dash doesn't provide a clean way to stop the server
            # This is a workaround that's not ideal
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                logger.warning("Dashboard server could not be stopped cleanly")
            else:
                func()
            
            self.is_running = False
            logger.info("Correlation dashboard stopped")
