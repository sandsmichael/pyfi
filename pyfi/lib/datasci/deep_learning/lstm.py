import sys, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

sp500_sectors ={'^SP500-15':"Materials",
                '^SP500-20':"Industrials", 
                '^SP500-25':"Consumer Discretionary", 
                '^SP500-30':"Consumer Staples", 
                '^SP500-35':"Health Care",
                '^SP500-40':"Financials",
                '^SP500-45':"Info Tech",
                '^SP500-50':"Comm Services",
                '^SP500-55':"Utilities",
                '^SP500-60':"Real Estate",
                '^GSPE':"Energy" }


benchmark_weights = {
    'Materials': 0.025,
    'Industrials': 0.10,
    'Consumer Discretionary': 0.10,
    'Consumer Staples': 0.05,
    'Health Care': 0.10,
    'Financials': 0.15,
    'Info Tech': 0.30,
    'Comm Services': 0.10,
    'Utilities': 0.025,
    'Real Estate': 0.025,
    'Energy': 0.025
}



PORTFOLIO_HOLDINGS = list(sp500_sectors.values())
    # 'AGG', 'TLT',   # 'IVV',
    # 'IWN', 'IWO',   # 'IWM', # Russell Large Value/Growth
    # 'IWF', 'IWD',   # 'IWB', # Russell Mid Value/Growth
    # 'IWP', 'IWS',   # 'IWR', # Russell Small Value/Growth


FEATURE_COLUMNS = [
    
]

NUM_EPISODES = 1
WINDOW_SIZE = 252


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Log                                                                                                              │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
import json
from datetime import datetime
import logging
from pathlib import Path

class ModelLogger:
    """Handles structured logging of model training and evaluation metrics."""
    def __init__(self, log_dir='./output/logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log structure
        self.log = {
            'run_id': datetime.now().strftime('%Y%m%d'),  # '%Y%m%d_%H%M%S'
            'config': {
                'portfolio_holdings': PORTFOLIO_HOLDINGS,
                'window_size': WINDOW_SIZE,
                'num_episodes': NUM_EPISODES
            },
            'training': {
                'folds': []
            },
            'validation': {},
            'predictions': {
                'single_model': {},
                'ensemble': {}
            }
        }
        
        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / f"model_{self.log['run_id']}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_dataset_config(self, dataset):
        """Log dataset configuration."""
        self.log['config']['dataset'] = {
            'window_size': dataset.window_size,
            'num_features': dataset.state_size,
            'num_assets': dataset.num_assets,
            'data_shape': dataset.data.shape,
            'portfolio_cols': dataset.portfolio_columns,
            'feature_cols': dataset.feature_columns,
            'forward_cols': dataset.forward_columns,
        }
    
    def log_fold(self, fold_idx, metrics, losses, returns_df):
        """Log metrics for a single fold."""
        fold_data = {
            'fold_number': fold_idx + 1,
            'metrics': metrics,
            'avg_loss': np.mean(losses),
            'returns_summary': returns_df.describe().to_dict()
        }
        self.log['training']['folds'].append(fold_data)
    
    def log_validation(self, results_df, validation_performance):
        """Log validation period results."""
        self.log['validation'] = {
            'summary_stats': results_df.describe().to_dict(),
            'performance': validation_performance
        }
    
    def log_prediction(self, portfolio, uncertainty=None, mode='single'):
        """Log model predictions."""
        prediction_data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'allocations': portfolio
        }
        if uncertainty:
            prediction_data['uncertainty'] = uncertainty
            
        if mode == 'single':
            self.log['predictions']['single_model'] = prediction_data
        else:
            self.log['predictions']['ensemble'] = prediction_data
    
    def save(self):
        """Save log to JSON file."""
        log_file = self.log_dir / f"model_log_{self.log['run_id']}.json"
        with open(log_file, 'w') as f:
            json.dump(self.log, f, indent=4, default=str)
        self.logger.info(f"Saved log to {log_file}")



logger = ModelLogger()

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Portfolio Dataset                                                                                                │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
class PortfolioDataset:
    """
    Handles data preparation and loading for LSTM-based portfolio optimization.
    Read data from data.csv including portfolio returns, forward returns, macro features, etc.
    Prepare sequences/batches based on window size for training and prediction.
    """
    def __init__(self, data_path="data.csv", window_size=WINDOW_SIZE, portfolio_columns=None):
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.data.sort_index(inplace=True)
        
        self.portfolio_columns = portfolio_columns
        self.forward_columns = [f"{col}_fwd_1y" for col in portfolio_columns]
        self.feature_columns = [col for col in self.data.columns if "_fwd_" not in col and 'IVV' not in col]
        
        self.window_size = window_size
        self.state_size = len(self.feature_columns)
        self.num_assets = len(self.portfolio_columns)
        
        # print("\nDataset Configuration:")
        # print(f"Number of features: {self.state_size}")
        # print(f"Window size: {window_size}")
        # print(f"Number of assets: {self.num_assets}")
        
    def prepare_sequence(self, start_idx):
        """Prepare a single sequence of data for LSTM."""
        # Get features for window [t : t+window_size]
        features = self.data.iloc[start_idx:start_idx + self.window_size][self.feature_columns].values
            
        # Handle the case when we're at the last window
        if start_idx + self.window_size >= len(self.data):
            # Use the last available forward returns
            returns = self.data.iloc[-1][self.forward_columns].values
        else:
            # Get forward returns at end of window (t+window_size)
            returns = self.data.iloc[start_idx + self.window_size][self.forward_columns].values
        
        return torch.FloatTensor(features), torch.FloatTensor(returns)
        
    def get_prediction_data(self):
        """Get the most recent window of data for prediction."""
        # Use the last window_size rows for features
        last_window_start = len(self.data) - self.window_size
        
        # Get features from the last window
        features = self.data.iloc[last_window_start:][self.feature_columns].values
        
        # For returns, use the last row's forward returns
        returns = self.data.iloc[-1][self.forward_columns].values
        
        return torch.FloatTensor(features), torch.FloatTensor(returns)
    

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Model                                                                                                            │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
class HybridPortfolioModel(nn.Module):
    """
    Hybrid LSTM model for portfolio optimization.
    Combines LSTM for market understanding with allocation head for portfolio weights.
    """
    def __init__(self, input_dim, hidden_dim, num_assets, num_layers=2, benchmark_weights=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for temporal pattern recognition
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Portfolio allocation head
        self.allocation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_assets),
            nn.Softmax(dim=-1),
            MinMaxScaler(benchmark_weights, deviation=0.10) 
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last LSTM output for allocation
        last_hidden = lstm_out[:, -1, :]
        # Generate portfolio weights
        weights = self.allocation(last_hidden)
        return weights

    def get_feature_attribution(self, x):
            """Calculate feature importance using integrated gradients."""
            x.requires_grad = True
            
            # Get original prediction
            original_weights = self.forward(x)
            
            # Calculate gradients for each asset allocation
            feature_importance = []
            for asset_idx in range(original_weights.shape[1]):
                self.zero_grad()
                original_weights[0, asset_idx].backward(retain_graph=True)
                
                # Get gradients with respect to input
                input_gradients = x.grad.data.numpy()[0]  # [window_size, features]
                
                # Average importance across time steps
                avg_importance = np.abs(input_gradients).mean(axis=0)
                feature_importance.append(avg_importance)
                
            return np.array(feature_importance)  # [num_assets, num_features]



class MinMaxScaler(nn.Module):
    """Ensures allocations stay within benchmark-relative bounds and uses whole percentage points."""
    def __init__(self, benchmark_weights, deviation=0.10):
        super().__init__()
        self.benchmark_weights = torch.tensor([benchmark_weights[asset] for asset in PORTFOLIO_HOLDINGS])
        self.deviation = deviation
        
    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        
        # Ensure x is 2D
        x = x.view(batch_size, -1)
        
        # Expand benchmark weights to match batch dimension
        benchmark = self.benchmark_weights.expand(batch_size, -1)
        
        # Initial normalization
        x = x / x.sum(dim=1, keepdim=True)
        
        # Round to whole percentage points (0.01)
        x = torch.round(x * 100) / 100
        
        # Calculate bounds relative to benchmark
        min_weights = torch.maximum(benchmark - self.deviation, torch.tensor(0.0))
        max_weights = torch.minimum(benchmark + self.deviation, torch.tensor(1.0))
        
        # Round bounds to whole percentage points
        min_weights = torch.round(min_weights * 100) / 100
        max_weights = torch.round(max_weights * 100) / 100
        
        # Iterative enforcement of constraints
        max_iterations = 100
        for _ in range(max_iterations):
            # Enforce minimum allocations
            below_min = x < min_weights
            x = torch.where(below_min, min_weights, x)
            
            # Round after min enforcement
            x = torch.round(x * 100) / 100
            x = x / x.sum(dim=1, keepdim=True)
            
            # Enforce maximum allocations
            above_max = x > max_weights
            if above_max.any():
                x = torch.where(above_max, max_weights, x)
                remaining = 1 - (x * above_max).sum(dim=1, keepdim=True)
                scale = remaining / (x * ~above_max).sum(dim=1, keepdim=True)
                x = torch.where(above_max, x, x * scale)
                
                # Round after max enforcement
                x = torch.round(x * 100) / 100
            
            # Final normalization and rounding
            x = x / x.sum(dim=1, keepdim=True)
            x = torch.round(x * 100) / 100
            
            # Check if all constraints are satisfied
            if torch.all((x >= min_weights - 1e-6) & (x <= max_weights + 1e-6)):
                break
        
        return x


class PortfolioOptimizer:
    """
    Handles training and optimization of the hybrid portfolio model.
    """
    def __init__(self, dataset, hidden_dim=64, lr=0.001):
        self.dataset = dataset
        self.model = HybridPortfolioModel(
            input_dim=dataset.state_size,
            hidden_dim=hidden_dim,
            num_assets=dataset.num_assets,
            benchmark_weights=benchmark_weights
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
                    
    def calculate_sharpe(self, weights, forward_returns):
        """Calculate Sharpe ratio with volatility constraint relative to benchmark."""
        # Get trailing 12-month window of returns (252 trading days)
        window_end = self.dataset.data.index[-1]
        window_start = window_end - pd.DateOffset(months=12)
        trailing_data = self.dataset.data.loc[window_start:window_end]
        
        # Calculate trailing portfolio volatility
        trailing_asset_returns = trailing_data[self.dataset.portfolio_columns]
        portfolio_returns = (trailing_asset_returns * weights.detach().numpy()).sum(axis=1)
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
        
        # Calculate trailing benchmark volatility
        benchmark_returns = trailing_data['IVV']  # ivv_fw_1y?
        benchmark_vol = np.std(benchmark_returns) * np.sqrt(252)
        
        # Calculate expected portfolio return
        portfolio_return = (weights * forward_returns).sum()
        
        # Base Sharpe ratio
        sharpe = portfolio_return / (portfolio_vol + 1e-6)
        
        # Add volatility constraint penalty
        vol_penalty = torch.tensor(0.0)
        if portfolio_vol > benchmark_vol:
            vol_penalty = torch.tensor((portfolio_vol - benchmark_vol) * 2.0)
        
        # Add diversification penalties
        concentration_penalty = torch.sum(weights * weights) * 0.3
        entropy_penalty = -torch.sum(weights * torch.log(weights + 1e-6)) * 0.4
        min_allocation_penalty = torch.sum(torch.clamp(0.05 - weights, min=0)) * 1.0
        
        # Combine metrics
        adjusted_sharpe = sharpe - vol_penalty #- concentration_penalty + entropy_penalty - min_allocation_penalty
        
        return adjusted_sharpe
    

    def train_step(self, features, returns):
        """Execute one training step."""
        self.optimizer.zero_grad()
        
        # Get model prediction
        weights = self.model(features.unsqueeze(0))
        
        # Calculate Sharpe ratio (negative for minimization)
        loss = -self.calculate_sharpe(weights.squeeze(), returns)
        
        # Backpropagate and update
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), weights.detach().squeeze()


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Evaluate                                                                                                         │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
def evaluate_fold_performance(model, dataset, test_indices, benchmark_col='IVV'):
    """Evaluate model performance using actual realized returns."""
    model.eval()
    portfolio_returns = []
    benchmark_returns = []
    dates = []
    
    with torch.no_grad():
        for idx in test_indices:
            # Check if we have enough data for next day's return
            if idx + dataset.window_size + 1 >= len(dataset.data):
                continue
                
            # Get model prediction
            features, _ = dataset.prepare_sequence(idx)
            weights = model(features.unsqueeze(0))
            allocation = weights.squeeze().numpy()
            
            try:
                # Get actual returns and date
                actual_returns = np.array([
                    dataset.data[col].iloc[idx + dataset.window_size + 1]
                    for col in dataset.portfolio_columns
                ])
                date = dataset.data.index[idx + dataset.window_size + 1]
                
                # Calculate realized portfolio return
                portfolio_return = np.sum(allocation * actual_returns)
                portfolio_returns.append(portfolio_return)
                
                # Get benchmark return
                benchmark_return = dataset.data[benchmark_col].iloc[idx + dataset.window_size + 1]
                benchmark_returns.append(benchmark_return)
                
                dates.append(date)
                
            except IndexError as e:
                print(f"Warning: Skipping evaluation at index {idx} due to insufficient data")
                continue
    
    if not portfolio_returns:
        print("Warning: No valid returns calculated")
        return {
            'portfolio_return': 0,
            'benchmark_return': 0,
            'excess_return': 0,
            'portfolio_sharpe': 0,
            'benchmark_sharpe': 0,
            'portfolio_vol': 0,
            'benchmark_vol': 0,
            'returns_df': pd.DataFrame(),
            'test_start_date': None,
            'test_end_date': None,
            'lookback_start_date': None,
            'lookback_end_date': None            
        }

    # Create DataFrame with daily returns
    returns_df = pd.DataFrame({
        'Portfolio': np.exp(portfolio_returns) - 1,  # Convert from log to simple returns
        'Benchmark': np.exp(benchmark_returns) - 1   # Convert from log to simple returns
    }, index=dates)
    
    # Calculate performance metrics
    portfolio_cum_return = (1 + returns_df['Portfolio']).prod() - 1
    benchmark_cum_return = (1 + returns_df['Benchmark']).prod() - 1
    
    portfolio_sharpe = (np.mean(portfolio_returns)) / (np.std(portfolio_returns) + 1e-6) * np.sqrt(252)
    benchmark_sharpe = (np.mean(benchmark_returns)) / (np.std(benchmark_returns) + 1e-6) * np.sqrt(252)

    test_start_date = dates[0]
    test_end_date = dates[-1]
    lookback_start_date = dataset.data.index[test_indices[0]]
    lookback_end_date = dataset.data.index[test_indices[0] + dataset.window_size]
        
    return {
        'portfolio_return': portfolio_cum_return,
        'benchmark_return': benchmark_cum_return,
        'excess_return': portfolio_cum_return - benchmark_cum_return,
        'portfolio_sharpe': portfolio_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'portfolio_vol': np.std(portfolio_returns) * np.sqrt(252),
        'benchmark_vol': np.std(benchmark_returns) * np.sqrt(252),
        'returns_df': returns_df,
        'test_start_date': test_start_date,
        'test_end_date': test_end_date,
        'lookback_start_date': lookback_start_date,
        'lookback_end_date': lookback_end_date        
    }

def plot_kfold_metrics(results_df, losses_by_fold, returns_dfs, save_path='./output/kfold_metrics.png'):
    """Create comprehensive visualization of k-fold training metrics."""
    # Create two figures
    fig1, fig2 = plt.figure(figsize=(15, 10)), plt.figure(figsize=(15, 8))
    
    # First figure - Original metrics
    gs1 = plt.GridSpec(2, 2, figure=fig1)
    
    # 1. Excess Returns by Fold
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.plot(range(1, len(results_df) + 1), results_df['excess_return'], 'g-o')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Excess Returns by Fold')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Excess Return')
    ax1.grid(True)
    
    # 2. Portfolio Volatility
    ax2 = fig1.add_subplot(gs1[0, 1])
    x = range(1, len(results_df) + 1)
    ax2.plot(x, results_df['portfolio_vol'], 'b-o', label='Portfolio')
    ax2.plot(x, results_df['benchmark_vol'], 'r--o', label='Benchmark')
    ax2.set_title('Annualized Volatility by Fold')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Volatility')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Sharpe Ratios
    ax3 = fig1.add_subplot(gs1[1, 0])
    x = range(1, len(results_df) + 1)
    ax3.plot(x, results_df['portfolio_sharpe'], 'b-o', label='Portfolio')
    ax3.plot(x, results_df['benchmark_sharpe'], 'r--o', label='Benchmark')
    ax3.set_title('Sharpe Ratios by Fold')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Returns Comparison
    ax4 = fig1.add_subplot(gs1[1, 1])
    width = 0.35
    ax4.bar([i - width/2 for i in x], results_df['portfolio_return'], 
            width, label='Portfolio', color='blue', alpha=0.6)
    ax4.bar([i + width/2 for i in x], results_df['benchmark_return'], 
            width, label='Benchmark', color='red', alpha=0.6)
    ax4.set_title('Returns Comparison by Fold')
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Return')
    ax4.legend()
    ax4.grid(True)
    
    # Second figure - Episode losses by fold
    num_folds = len(losses_by_fold)
    gs2 = plt.GridSpec(num_folds, 1, figure=fig2)
    
    for fold, fold_losses in enumerate(losses_by_fold):
        ax = fig2.add_subplot(gs2[fold])
        
        # Simply reshape losses into episodes and take mean
        episode_losses = np.array_split(fold_losses, NUM_EPISODES)
        episode_avgs = [np.mean(episode) for episode in episode_losses]
        
        # Plot average loss per episode
        ax.plot(range(1, NUM_EPISODES + 1), episode_avgs, 'b-o')
        ax.set_title(f'Fold {fold + 1} Average Loss by Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Loss')
        ax.grid(True)
    

    # Adjust layouts
    fig1.suptitle('K-Fold Cross Validation Metrics', fontsize=16, y=1.02)
    fig2.suptitle('Episode Losses by Fold', fontsize=16, y=1.02)
    fig1.tight_layout()
    fig2.tight_layout()
    
    # Save plots
    fig1.savefig(save_path, bbox_inches='tight', dpi=300)
    fig2.savefig(save_path.replace('.png', '_episode_losses.png'), bbox_inches='tight', dpi=300)
    plt.close('all')



     # Create new figure for returns time series
    # Create figure for returns time series subplots
    fig3 = plt.figure(figsize=(15, 10))
    num_folds = len(returns_dfs)
    gs3 = plt.GridSpec(num_folds, 1, figure=fig3)

    for fold, returns_df in enumerate(returns_dfs):
        ax = fig3.add_subplot(gs3[fold])
        cum_returns = (1 + returns_df).cumprod()
        # Plot portfolio returns for this fold
        ax.plot(cum_returns.index, cum_returns['Portfolio'], 
                'b-', label='Portfolio', alpha=0.8)
        # Plot benchmark returns
        ax.plot(cum_returns.index, cum_returns['Benchmark'], 
                'r--', label='Benchmark', alpha=0.8)
        # Add fold period information
        start_date = returns_df.index[0]
        end_date = returns_df.index[-1]
        ax.set_title(f'Fold {fold + 1} Returns: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
        ax.grid(True)
        ax.legend()
        ax.set_ylabel('Cumulative Return')
        # # Format x-axis
        # ax.xaxis.set_major_locator(mdates.YearLocator())
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Only show x-label on bottom subplot
        if fold == num_folds - 1:
            ax.set_xlabel('Date')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    # Adjust layout
    fig3.suptitle('Out-of-Sample Test Performance by Fold', fontsize=16, y=1.02)
    fig3.tight_layout()
    
    # Save plot
    fig3.savefig(save_path.replace('.png', '_returns.png'), bbox_inches='tight', dpi=300)
    plt.close()


def plot_validation_performance(trained_models, dataset, validation_indices, benchmark_col='IVV', save_path='./output/validation_performance.png'):
    """Plot how each fold's model performs on the validation set."""
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Get validation period returns for each model
    validation_returns = []
    for i, model in enumerate(trained_models):
        performance = evaluate_fold_performance(
            model, dataset, validation_indices, benchmark_col
        )
        validation_returns.append(performance['returns_df'])
    
    # Plot cumulative returns
    for i, returns_df in enumerate(validation_returns):
        simple_returns = np.exp(returns_df) - 1
        cum_returns = (1 + simple_returns['Portfolio']).cumprod()
        plt.plot(cum_returns.index, cum_returns, 
                label=f'Model {i+1}', alpha=0.6)
    
    # Plot benchmark
    benchmark_returns = validation_returns[0]['Benchmark']
    simple_benchmark = np.exp(benchmark_returns) - 1
    cum_benchmark = (1 + simple_benchmark).cumprod()
    plt.plot(cum_benchmark.index, cum_benchmark, 
            'r--', label='Benchmark', linewidth=2, alpha=0.8)
    
    # Customize plot
    plt.title('Model Performance on Validation Period\n(Last Year of Data)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.legend()
    
    # Format x-axis
    # ax.xaxis.set_major_locator(mdates.YearLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    
    # Add validation period dates
    start_date = benchmark_returns.index[0]
    end_date = benchmark_returns.index[-1]
    plt.text(0.02, 0.98, f'Validation Period:\n{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Run Training                                                                                                     │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
def k_fold_train_evaluate(k=5, num_episodes=NUM_EPISODES, benchmark_col='IVV'):
    """Perform k-fold cross validation with benchmark comparison and out-of-sample validation.
    
    In the current implementation, each episode within a fold starts with the model's state from the previous episode. 
    This means that the weights learned in one episode are used as the starting point for the next episode within the 
    same fold. This is a standard approach in training neural networks, where the model is iteratively improved over
    multiple episodes (or epochs). the learning from previous episodes within a fold does not feed into the starting point for the new fold.
    Each fold starts with a freshly initialized model.
    """
    
    # logger = ModelLogger()
    dataset = PortfolioDataset(data_path="data.csv", portfolio_columns=PORTFOLIO_HOLDINGS)
    logger.log_dataset_config(dataset)
    
    # Reserve validation period (last year)
    validation_size = 252
    total_samples = len(dataset.data) - dataset.window_size - validation_size
    fold_size = total_samples // k
    
    print(f"\nStarting {k}-fold Cross Validation")
    print("-" * 60)

    fold_results = []
    losses_by_fold = []
    returns_dfs = []
    trained_models = []  # Store all trained models
    
    validation_indices = range(total_samples, len(dataset.data) - dataset.window_size)
    
    for fold in range(k):
        print(f"\nFold {fold + 1}/{k}")
        print("-" * 40)

        
        # Calculate fold indices
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size
        test_indices = range(test_start, test_end)
        train_indices = list(range(0, test_start)) + list(range(test_end, total_samples))
        
        # Initialize model and optimizer
        optimizer = PortfolioOptimizer(dataset, hidden_dim=64, lr=0.0005)
        
        # Store all losses for this fold
        fold_losses = []
        
        # Training loop
        for episode in range(num_episodes):
            random.shuffle(train_indices)
            
            for idx in train_indices:
                features, returns = dataset.prepare_sequence(idx)
                loss, _ = optimizer.train_step(features, returns)
                fold_losses.append(float(loss))
            
            avg_loss = np.mean(fold_losses[-len(train_indices):])
            print(f"Episode {episode + 1} - Avg Loss: {avg_loss:.4f}")
        
        losses_by_fold.append(fold_losses)
        
        # Evaluate on test fold
        test_performance = evaluate_fold_performance(
            optimizer.model,
            dataset,
            test_indices,
            benchmark_col
        )
        
        fold_metrics = {
            'portfolio_return': test_performance['portfolio_return'],
            'benchmark_return': test_performance['benchmark_return'],
            'excess_return': test_performance['excess_return'],
            'portfolio_sharpe': test_performance['portfolio_sharpe'],
            'benchmark_sharpe': test_performance['benchmark_sharpe'],
            'portfolio_vol': test_performance['portfolio_vol'],
            'benchmark_vol': test_performance['benchmark_vol'],
        }
        
        fold_results.append(fold_metrics)
        returns_dfs.append(test_performance['returns_df'])
        trained_models.append(optimizer.model)
        
        print(f"\nFold {fold + 1} Test Results (Ret. Not. Anlzd; Vol is.):")
        print(f"Portfolio Return: {fold_metrics['portfolio_return']:.2%}")
        print(f"Benchmark Return: {fold_metrics['benchmark_return']:.2%}")
        print(f"Excess Return: {fold_metrics['excess_return']:.2%}")
        print(f"Portfolio Sharpe: {fold_metrics['portfolio_sharpe']:.2f}")
        print(f"Test Period: {test_performance['test_start_date']} - {test_performance['test_end_date']}")
        print(f"Lookback Period (window size): {test_performance['lookback_start_date']} - {test_performance['lookback_end_date']}")

        logger.log_fold(fold, fold_metrics, fold_losses, test_performance['returns_df'])

    
    # Save all models
    os.makedirs('./output/ensemble', exist_ok=True)
    for i, model in enumerate(trained_models):
        torch.save({
            'model_state_dict': model.state_dict(),
            'fold': i + 1,
            'portfolio_holdings': PORTFOLIO_HOLDINGS,
            'input_dim': dataset.state_size,
            'num_assets': dataset.num_assets
        }, f'./output/ensemble/model_fold_{i+1}.pth')
    
    results_df = pd.DataFrame(fold_results)
    logger.log_validation(results_df, {
        'trained_models': len(trained_models),
        'validation_period': str(dataset.data.index[validation_indices[0]])
    })
    logger.save()

    plot_kfold_metrics(
        results_df, 
        losses_by_fold, 
        returns_dfs
    )

    plot_validation_performance(
        trained_models, 
        dataset, 
        validation_indices,
        benchmark_col='IVV',
        save_path='./output/validation_period_performance.png'
    )  

    analyze_allocation_changes(
        trained_models,
        dataset,
        validation_indices,
        threshold=0.05
    )

    print("\nValidation Period Summary:")
    print("-" * 60)
    print('\nResults Df:')
    print(results_df.head())
    print("\nMean Performance:")
    print(results_df.mean().round(4))
    print("\nPerformance Standard Deviation:")
    print(results_df.std().round(4))
    
    return results_df, logger


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Predict                                                                                                          │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""


def get_ensemble_allocation(model_dir='./output/ensemble', k=5):
    """Get current portfolio allocation using ensemble of models and save validation period weights."""
    dataset = PortfolioDataset(data_path="data.csv", portfolio_columns=PORTFOLIO_HOLDINGS)
    
    # Get validation period indices (last year)
    validation_size = 252
    start_idx = len(dataset.data) - validation_size - dataset.window_size
    validation_indices = range(start_idx, len(dataset.data) - dataset.window_size)
    validation_dates = dataset.data.index[validation_indices]
    print(f"Making predictions for Validation Period: {validation_dates[0]} to {validation_dates[-1]}")
    print(f"Using data window size of {dataset.window_size} days")

    # Initialize storage for daily predictions
    daily_predictions = {
        date: [] for date in validation_dates
    }
    
    # Load and predict with each model for each day
    for fold in range(k):
        model_path = f'{model_dir}/model_fold_{fold+1}.pth'
        
        try:
            # Initialize model
            model = HybridPortfolioModel(dataset.state_size, 64, dataset.num_assets, benchmark_weights=benchmark_weights)
            checkpoint = torch.load(model_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Get predictions for each day in validation period
            with torch.no_grad():
                for idx, date in zip(validation_indices, validation_dates):
                    features, _ = dataset.prepare_sequence(idx)
                    weights = model(features.unsqueeze(0))
                    daily_predictions[date].append(weights.squeeze().numpy())
                    
        except Exception as e:
            print(f"Error loading model {fold+1}: {e}")
            continue
    
    if not daily_predictions:
        raise ValueError("No valid predictions from ensemble")
    
    # Create DataFrames for weights and uncertainties
    weights_data = []
    uncertainty_data = []
    
    for date in validation_dates:
        predictions = np.array(daily_predictions[date])
        mean_weights = np.mean(predictions, axis=0)
        std_weights = np.std(predictions, axis=0)
        
        weights_data.append({
            'Date': date,
            **{asset: weight for asset, weight in zip(PORTFOLIO_HOLDINGS, mean_weights)}
        })
        
        uncertainty_data.append({
            'Date': date,
            **{f'{asset}_std': std for asset, std in zip(PORTFOLIO_HOLDINGS, std_weights)}
        })
    
    weights_df = pd.DataFrame(weights_data).set_index('Date')
    uncertainty_df = pd.DataFrame(uncertainty_data).set_index('Date')
    
    # Calculate summary statistics
    summary_stats = pd.DataFrame({
        'Mean': weights_df.mean(),
        'Std': weights_df.std(),
        'Min': weights_df.min(),
        'Max': weights_df.max()
    })
    
    # Save to Excel
    excel_path = './output/ensemble_weights.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        weights_df.to_excel(writer, sheet_name='Daily Weights')
        uncertainty_df.to_excel(writer, sheet_name='Daily Uncertainties')
        summary_stats.to_excel(writer, sheet_name='Summary Statistics')
    
    print(f"\nSaved ensemble weights to: {excel_path}")
    
    # Return current allocation (last day)
    portfolio = weights_df.iloc[-1].to_dict()
    uncertainty = uncertainty_df.iloc[-1].to_dict()
    
    logger.log_prediction(portfolio, uncertainty, mode='ensemble')
    logger.save()
    
    return portfolio, uncertainty


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Analyze                                                                                                          │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
def analyze_allocation_changes(trained_models, dataset, period_indices, threshold=0.01):
    """Analyze what features drive allocation changes."""
    
    results = []
    for idx in period_indices:
        features, _ = dataset.prepare_sequence(idx)
        date = dataset.data.index[idx + dataset.window_size]
        
        # Get ensemble predictions and feature importance
        all_importances = []
        all_weights = []
        
        for model in trained_models:
            model.eval()
            with torch.no_grad():
                weights = model(features.unsqueeze(0))
                all_weights.append(weights.squeeze().numpy())
            
            # Calculate feature importance
            importance = model.get_feature_attribution(features.unsqueeze(0))
            all_importances.append(importance)
        
        # Average across models
        mean_weights = np.mean(all_weights, axis=0)
        mean_importance = np.mean(all_importances, axis=0)
        
        # Detect significant changes in allocation
        if idx > period_indices[0]:
            weight_changes = np.abs(mean_weights - prev_weights)
            if np.any(weight_changes > threshold):
                # Get top contributing features
                changed_assets = np.where(weight_changes > threshold)[0]
                for asset_idx in changed_assets:
                    asset_name = dataset.portfolio_columns[asset_idx]
                    change = mean_weights[asset_idx] - prev_weights[asset_idx]
                    
                    # Get top features for this asset
                    asset_importance = mean_importance[asset_idx]
                    top_features = np.argsort(asset_importance)[-3:]  # Top 3 features
                    
                    results.append({
                        'date': date,
                        'asset': asset_name,
                        'weight_change': change,
                        'top_features': [
                            (dataset.feature_columns[i], float(asset_importance[i]))
                            for i in top_features
                        ]
                    })
        
        prev_weights = mean_weights
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(results)
    df.to_excel('./output/feature_attribution.xlsx', index=False)
    
    return df


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Init                                                                                                             │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
if __name__ == "__main__":
    ### Train: Cross Validation
    results, log = k_fold_train_evaluate(k=5)
    print("\nFinal Cross-Validation Results:")
    print(results.describe().round(4))   


    ## Inference
    # allocation = get_current_allocation()
    # print("\nFinal Portfolio Allocation:")
    # print("-" * 40)
    # for asset, weight in sorted(allocation.items()):
    #     print(f"{asset:4s}: {weight:7.2%}")
    # print("-" * 40)


    # Get ensemble prediction
    portfolio, uncertainty = get_ensemble_allocation(k=5)





# def get_current_allocation(model_path='./output/trained_portfolio_lstm.pth'):
#     """Get current portfolio allocation prediction."""
#     # logger = ModelLogger()
#     dataset = PortfolioDataset(data_path="data.csv", portfolio_columns=PORTFOLIO_HOLDINGS)
#     model = HybridPortfolioModel(dataset.state_size, 64, dataset.num_assets)
    
#     # Add numpy scalar to safe globals
#     import numpy._core.multiarray
#     torch.serialization.add_safe_globals([
#         numpy._core.multiarray.scalar
#     ])
    
#     # Load trained model
#     try:
#         checkpoint = torch.load(model_path, weights_only=False)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         print("Ensuring model exists...")
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model file {model_path} not found. Run training first.")
#         return None
    
#     # Load state dict
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     # Get prediction
#     features, _ = dataset.get_prediction_data()
#     with torch.no_grad():
#         weights = model(features.unsqueeze(0))
    
#     # Format results
#     portfolio = {
#         asset: float(weight)
#         for asset, weight in zip(PORTFOLIO_HOLDINGS, weights[0])
#     }

#     logger.log_prediction(portfolio)
#     logger.save()

#     # Validate allocation constraints
#     min_alloc = min(portfolio.values())
#     max_alloc = max(portfolio.values())
#     total_alloc = sum(portfolio.values())
    
#     print("\nAllocation Validation:")
#     print(f"Minimum allocation: {min_alloc:.2%}")
#     print(f"Maximum allocation: {max_alloc:.2%}")
#     print(f"Total allocation: {total_alloc:.2%}")
    
#     print(f"\nAllocation as of: {dataset.data.index[-1]}")
#     return portfolio
