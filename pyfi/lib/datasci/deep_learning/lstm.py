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


DATA_FP = r"C:\Users\micha\OneDrive\Documents\data\lstm_data.csv"
PORTFOLIO_HOLDINGS = list(sp500_sectors.values())
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
    def __init__(self, data_path=DATA_FP, window_size=WINDOW_SIZE, portfolio_columns=None):
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.data.sort_index(inplace=True)
        
        self.portfolio_columns = portfolio_columns
        self.forward_columns = [f"{col}_fwd_1y" for col in portfolio_columns]
        self.feature_columns = [col for col in self.data.columns if "_fwd_" not in col and 'SPX' not in col]
        
        self.window_size = window_size
        self.state_size = len(self.feature_columns)
        self.num_assets = len(self.portfolio_columns)
        
        # print("\nDataset Configuration:")
        # print(f"Number of features: {self.state_size}")
        # print(f"Window size: {window_size}")
        # print(f"Number of assets: {self.num_assets}")
        
    def prepare_sequence(self, start_idx):
        """Prepare a single sequence of data for LSTM.
        Passes a single row (index) of data to the model containing the dataset features and the forward portfolio returns at one point in time.
        """
        # Get features for window [t : t+window_size]
        features = self.data.iloc[start_idx:start_idx + self.window_size][self.feature_columns].values
            
        # Handle the case when we're at the last window
        if start_idx + self.window_size >= len(self.data):
            # Use the last available forward returns
            port_fwd_ret = self.data.iloc[-1][self.forward_columns].values  # array of fwd 252d returns for each asset in portfolio
            port_hist_ret = self.data.iloc[-252:][self.portfolio_columns].values  # array of historical returns for each asset in portfolio
            
            bm_fwd_ret = self.data.iloc[-1]['SPX_fwd_1y']  # forward 252d return for benchmark
            bm_hist_ret = self.data.iloc[-252:]['SPX'].values  # historical returns for benchmark
        else:
            # Get forward returns at end of window (t+window_size)
            port_fwd_ret = self.data.iloc[start_idx + self.window_size][self.forward_columns].values  # array of fwd 252d returns for each asset in portfolio
            port_hist_ret = self.data.iloc[start_idx:start_idx + self.window_size][self.portfolio_columns].values  # array of historical returns for each asset in portfolio
            
            bm_fwd_ret = self.data.iloc[start_idx + self.window_size]['SPX_fwd_1y']  # forward 252d return for benchmark
            bm_hist_ret = self.data.iloc[start_idx:start_idx + self.window_size]['SPX'].values  # historical returns for benchmark

        return (
                    torch.FloatTensor(features), 
                    port_fwd_ret,  
                    port_hist_ret, 
                    bm_fwd_ret,    
                    bm_hist_ret
                )  
        

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
            MinMaxScaler(benchmark_weights, deviation=0.05) 
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
    """Ensures allocations are within benchmark bounds without normalization."""
    def __init__(self, benchmark_weights, deviation=0.10):
        super().__init__()
        self.benchmark_weights = torch.tensor([benchmark_weights[asset] for asset in PORTFOLIO_HOLDINGS])
        self.deviation = deviation
    
    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        x = x.view(batch_size, -1)
        
        # Expand benchmark weights
        benchmark = self.benchmark_weights.expand(batch_size, -1)
        
        # Calculate bounds
        min_weights = benchmark - self.deviation
        max_weights = benchmark + self.deviation
        
        # Ensure bounds are valid
        min_weights = torch.maximum(min_weights, torch.tensor(0.0))
        max_weights = torch.minimum(max_weights, torch.tensor(1.0))
        
        # Just project to bounds without normalization
        x = torch.minimum(torch.maximum(x, min_weights), max_weights)
        
        return x


class PortfolioOptimizer:
    """
    Handles training and optimization of the hybrid portfolio model.
    """
    def __init__(self, dataset:PortfolioDataset, hidden_dim=64, lr=0.001):
        self.dataset = dataset
        self.model = HybridPortfolioModel(
            input_dim=dataset.state_size,
            hidden_dim=hidden_dim,
            num_assets=dataset.num_assets,
            benchmark_weights=benchmark_weights
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)


    def calculate_sharpe(self, weights, port_fwd_ret, port_hist_ret, bm_fwd_ret, bm_hist_ret):
        """Calculate Sharpe ratio with volatility matching and weight constraints.
        The loss function, inverse of adjusted sharpe ratio, is a function of:
            the portfolio weights and forward returns, portfolio forward volatility, and the benchmark forward returns and volatility.
        """
        if torch.is_tensor(weights):
            weights = weights.detach().numpy()
        
        # weights = weights / np.sum(weights)

        benchmark_return = bm_fwd_ret
        benchmark_vol = np.std(bm_hist_ret) * np.sqrt(252)
        
        portfolio_return = np.sum(weights * port_fwd_ret)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(np.cov(port_hist_ret.T), weights))) * np.sqrt(252)
        
        # Calculate Sharpe and penalties
        sharpe = portfolio_return / (portfolio_vol)

        vol_deviation = abs(portfolio_vol - benchmark_vol) / benchmark_vol  # Relative deviation
        vol_penalty = sharpe * vol_deviation  # Scale penalty to Sharpe

        adjusted_sharpe = torch.tensor(sharpe - vol_penalty, requires_grad=True)

        # print(' ###### Printing from calculate_sharpe() ###### ')
        # print(portfolio_vol)
        # print(benchmark_vol)
        # print(sharpe)
        # print(vol_deviation)
        # print(vol_penalty)
        # print( '###### ######')
                 
        return adjusted_sharpe
    

    def train_step(self, features, port_fwd_ret, port_hist_ret, bm_fwd_ret, bm_hist_ret):
        """Execute one training step. 
        Note: Core function accepting a single index of feature space data in order to generate predicted weights. 
        Then checks accuracy based on forward return loss function.
        """
        self.optimizer.zero_grad()
        
        # Get model prediction
        weights = self.model(features.unsqueeze(0)) # based on features, model outputs the weights
          
        # Calculate Sharpe ratio (negative for minimization)
        loss = -self.calculate_sharpe(weights.squeeze(), port_fwd_ret, port_hist_ret, bm_fwd_ret, bm_hist_ret) # based on model output weights, it checks to see if it did a good job!
        
        # Backpropagate and update
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), weights.detach().squeeze()


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Evaluate                                                                                                         │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

class KFoldValidator:
    """Handles k-fold cross validation for LSTM portfolio optimization."""
    
    def __init__(self, dataset, k=5, num_episodes=NUM_EPISODES):
        self.dataset = dataset
        self.k = k
        self.num_episodes = num_episodes
        self.trained_models = []
        self.fold_results = []
        self.losses_by_fold = []
        self.returns_dfs = []
        
        # Calculate validation period
        self.validation_size = 252
        self.total_samples = len(dataset.data) - dataset.window_size - self.validation_size
        self.fold_size = self.total_samples // k
        self.validation_indices = range(
            self.total_samples, 
            len(dataset.data) - dataset.window_size
        )
        
        logger.log_dataset_config(dataset)


    def train_fold(self, fold):
        """Train model on a single fold."""
        test_start = fold * self.fold_size
        test_end = (fold + 1) * self.fold_size
        test_indices = range(test_start, test_end)
        train_indices = list(range(0, test_start)) + list(range(test_end, self.total_samples))
        
        optimizer = PortfolioOptimizer(self.dataset, hidden_dim=64, lr=0.0005)
        fold_losses = []
        
        print(f"\nFold {fold + 1}/{self.k}")
        print("-" * 40)
        
        for episode in range(self.num_episodes):
            random.shuffle(train_indices)
            for idx in train_indices:
                features, port_fwd_ret, port_hist_ret, bm_fwd_ret, bm_hist_ret = self.dataset.prepare_sequence(idx)
                loss, _ = optimizer.train_step(features, port_fwd_ret, port_hist_ret, bm_fwd_ret, bm_hist_ret)
                fold_losses.append(float(loss))
            
            avg_loss = np.mean(fold_losses[-len(train_indices):])
            print(f"Episode {episode + 1} - Avg Loss: {avg_loss:.4f}")
        
        return optimizer.model, fold_losses, test_indices


    def test_step(self, model, test_indices, benchmark_col='SPX'):
        """Evaluate model performance using daily returns over test period.
        NOTE: Testing is making daily predictions for weights and returning the actual returns for the portfolio and benchmark. 
        Testing is not based on forward returns. Training is.
        """
        model.eval()
        data = []
        
        with torch.no_grad():
            for idx in test_indices:
                if idx + self.dataset.window_size >= len(self.dataset.data):
                    continue
                
                # Get features and historical returns for weight prediction
                features, _, port_hist_ret, _, bm_hist_ret = self.dataset.prepare_sequence(idx)
                date = self.dataset.data.index[idx + self.dataset.window_size]
                
                # Get next day's actual returns
                next_day_idx = idx + self.dataset.window_size + 1
                if next_day_idx >= len(self.dataset.data):
                    continue
                    
                try:
                    # Get model weights based on historical data
                    weights = model(features.unsqueeze(0)).squeeze().numpy()
                    
                    # Get next day's actual returns (still in log form)
                    next_day_returns = self.dataset.data.iloc[next_day_idx][self.dataset.portfolio_columns].values
                    next_day_benchmark = self.dataset.data.iloc[next_day_idx][benchmark_col]
                    
                    # Convert log returns to simple returns for performance calculation
                    portfolio_return = np.exp(np.sum(weights * next_day_returns)) - 1
                    benchmark_return = np.exp(next_day_benchmark) - 1
                    
                    # Calculate volatility using historical log returns
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(np.cov(port_hist_ret.T), weights))) * np.sqrt(252)
                    benchmark_vol = np.std(bm_hist_ret) * np.sqrt(252)
                    
                    data.append({
                        'date': date,
                        'weights': weights,
                        'portfolio_return': portfolio_return,
                        'benchmark_return': benchmark_return,
                        'portfolio_vol': portfolio_vol,
                        'benchmark_vol': benchmark_vol
                    })
                    
                except IndexError:
                    continue
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(data).set_index('date')
        
        # Calculate cumulative performance
        cum_port_return = (1 + returns_df['portfolio_return']).prod() - 1
        cum_bench_return = (1 + returns_df['benchmark_return']).prod() - 1
        
        return {
            'cumulative_portfolio_return': cum_port_return,
            'cumulative_benchmark_return': cum_bench_return,
            'cumulative_excess_return': cum_port_return - cum_bench_return,
            'annualized_portfolio_return': (1 + cum_port_return) ** (252/len(returns_df)) - 1,
            'annualized_benchmark_return': (1 + cum_bench_return) ** (252/len(returns_df)) - 1,
            'average_portfolio_vol': returns_df['portfolio_vol'].mean(),
            'average_benchmark_vol': returns_df['benchmark_vol'].mean(),
            'portfolio_sharpe': (returns_df['portfolio_return'].mean() * 252) / (returns_df['portfolio_return'].std() * np.sqrt(252)),
            'benchmark_sharpe': (returns_df['benchmark_return'].mean() * 252) / (returns_df['benchmark_return'].std() * np.sqrt(252)),
            'test_start_date': returns_df.index[0],
            'test_end_date': returns_df.index[-1],
            'lookback_start_date': self.dataset.data.index[test_indices[0]],
            'lookback_end_date': self.dataset.data.index[test_indices[0] + self.dataset.window_size],
            'returns_df': returns_df,
            'weights_df': pd.DataFrame([d['weights'] for d in data], index=returns_df.index, columns=self.dataset.portfolio_columns)
        }
    
    
    def plot_metrics(self, results_df, save_path='./output/kfold_metrics.png'):
        """Plot k-fold cross validation metrics."""
        fig1, fig2 = plt.figure(figsize=(15, 10)), plt.figure(figsize=(15, 8))
        
        # First figure - Performance metrics
        gs1 = plt.GridSpec(2, 2, figure=fig1)
        x = range(1, len(results_df) + 1)
        width = 0.35
        
        metrics = [
            ('excess_return', 'Excess Returns', 'g'),
            ('portfolio_vol', 'Portfolio Volatility', 'blue'),
            ('portfolio_sharpe', 'Sharpe Ratios', 'blue'),
            ('portfolio_return', 'Returns', 'blue')
        ]
        
        for i, (metric, title, color) in enumerate(metrics):
            ax = fig1.add_subplot(gs1[i//2, i%2])
            if metric == 'excess_return':
                # Plot excess returns directly
                ax.bar(x, results_df[metric], color=color, alpha=0.6)
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            else:
                # Plot portfolio and benchmark metrics
                portfolio_values = results_df[metric]
                benchmark_values = results_df[f'benchmark_{metric.split("_")[1]}']
                
                ax.bar([i - width/2 for i in x], portfolio_values, 
                    width, label='Portfolio', color=color, alpha=0.6)
                ax.bar([i + width/2 for i in x], benchmark_values, 
                    width, label='Benchmark', color='red', alpha=0.6)
                ax.legend()
            
            ax.set_title(f'{title} by Fold')
            ax.set_xlabel('Fold')
            ax.set_ylabel(title.split()[0])
            ax.grid(True)
        
        
        # Second figure - Episode losses
        gs2 = plt.GridSpec(self.k, 1, figure=fig2)
        for fold, losses in enumerate(self.losses_by_fold):
            ax = fig2.add_subplot(gs2[fold])
            episode_losses = np.array_split(losses, self.num_episodes)
            episode_avgs = [np.mean(episode) for episode in episode_losses]
            ax.plot(range(1, self.num_episodes + 1), episode_avgs, 'b-o')
            ax.set_title(f'Fold {fold + 1} Average Loss by Episode')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Average Loss')
            ax.grid(True)
        
        # Save plots
        fig1.suptitle('K-Fold Cross Validation Metrics', fontsize=16, y=1.02)
        fig2.suptitle('Training Losses by Fold', fontsize=16, y=1.02)
        fig1.tight_layout()
        fig2.tight_layout()
        
        fig1.savefig(save_path, bbox_inches='tight', dpi=300)
        fig2.savefig(save_path.replace('.png', '_losses.png'), bbox_inches='tight', dpi=300)
        plt.close('all')
        
        # Plot returns time series
        self.plot_returns_series(save_path.replace('.png', '_returns.png'))
    

    def plot_returns_series(self, save_path):
        """Plot returns time series for each fold."""
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(self.k, 1, figure=fig)
        
        for fold, returns_df in enumerate(self.returns_dfs):
            ax = fig.add_subplot(gs[fold])
            cum_returns = (1 + returns_df).cumprod()
            
            ax.plot(cum_returns.index, cum_returns['portfolio_return'], 
                   'b-', label='Portfolio', alpha=0.8)
            ax.plot(cum_returns.index, cum_returns['benchmark_return'], 
                   'r--', label='Benchmark', alpha=0.8)
            
            start_date = returns_df.index[0]
            end_date = returns_df.index[-1]
            ax.set_title(f'Fold {fold + 1} Returns: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
            ax.grid(True)
            ax.legend()
            ax.set_ylabel('Cumulative Return')
            
            if fold == self.k - 1:
                ax.set_xlabel('Date')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.suptitle('Out-of-Sample Test Performance by Fold', fontsize=16, y=1.02)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_validation_performance(self, save_path='./output/validation_performance.png'):
        """Plot model performance on validation period for each model and benchmark."""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        validation_returns = []
        for i, model in enumerate(self.trained_models):
            performance = self.test_step(model, self.validation_indices)
            validation_returns.append(performance['returns_df'])
            
            cum_returns = (1 + performance['returns_df']['portfolio_return']).cumprod()
            plt.plot(cum_returns.index, cum_returns, label=f'Model {i+1}', alpha=0.6)
        
        benchmark_returns = validation_returns[0]['benchmark_return']
        cum_benchmark = (1 + benchmark_returns).cumprod()
        plt.plot(cum_benchmark.index, cum_benchmark, 
                'r--', label='Benchmark', linewidth=2, alpha=0.8)
        
        plt.title('Model Performance on Validation Period\n(Last Year of Data)', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.legend()
        
        start_date = benchmark_returns.index[0]
        end_date = benchmark_returns.index[-1]
        plt.text(0.02, 0.98, 
                f'Validation Period:\n{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    

    def run(self):
        """Execute k-fold validation."""
        print(f"\nStarting {self.k}-fold Cross Validation")
        print("-" * 60)
        
        for fold in range(self.k):
            # Train fold
            model, losses, test_indices = self.train_fold(fold)
            self.trained_models.append(model)
            self.losses_by_fold.append(losses)
            
            # Evaluate fold
            performance = self.test_step(model, test_indices)
            self.fold_results.append({
                'portfolio_return': performance['cumulative_portfolio_return'],  # Updated key
                'benchmark_return': performance['cumulative_benchmark_return'],  # Updated key
                'excess_return': performance['cumulative_excess_return'],  # Updated key
                'portfolio_sharpe': performance['portfolio_sharpe'],
                'benchmark_sharpe': performance['benchmark_sharpe'],
                'portfolio_vol': performance['average_portfolio_vol'],  # Updated key
                'benchmark_vol': performance['average_benchmark_vol']   # Updated key
            })
            self.returns_dfs.append(performance['returns_df'])
            
            # Save model
            os.makedirs('./output/ensemble', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold': fold + 1,
                'portfolio_holdings': PORTFOLIO_HOLDINGS,
                'input_dim': self.dataset.state_size,
                'num_assets': self.dataset.num_assets
            }, f'./output/ensemble/model_fold_{fold+1}.pth')
            
            # Log results
            logger.log_fold(fold, self.fold_results[-1], losses, performance['returns_df'])
        
        # Create results DataFrame and plot
        results_df = pd.DataFrame(self.fold_results)
        logger.log_validation(results_df, {
            'trained_models': len(self.trained_models),
            'validation_period': str(self.dataset.data.index[self.validation_indices[0]])
        })
        logger.save()
        
        self.plot_metrics(results_df)
        self.plot_validation_performance()
        
        print("\nValidation Period Summary:")
        print("-" * 60)
        print(results_df.describe().round(4))
        
        return results_df



""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Predict                                                                                                          │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

def get_ensemble_allocation(model_dir='./output/ensemble', k=5):
    """
    Get current portfolio allocation using ensemble of models and save validation period weights.
    Includes individual model predictions and ensemble statistics.
    write weights to ensemble_weights which is average of each individual model. each model is used to predict each days weights across all days in validation period.
    """
    dataset = PortfolioDataset(data_path=DATA_FP, portfolio_columns=PORTFOLIO_HOLDINGS)
    
    validation_size = 252
    validation_end = len(dataset.data)
    validation_start = validation_end - validation_size
    validation_indices = range(validation_start, validation_end)
    
    validation_dates = dataset.data.index[validation_indices]
    print(f"Making predictions for Validation Period: {validation_dates[0]} to {validation_dates[-1]}")
    print(f"Using data window size of {dataset.window_size} days")

    # Store predictions from each model
    model_predictions = {}
    daily_predictions = {date: [] for date in validation_dates}
    
    # Load and predict with each model
    for fold in range(k):
        model_path = f'{model_dir}/model_fold_{fold+1}.pth'
        model_weights_data = []
        
        try:
            # Initialize and load model
            model = HybridPortfolioModel(dataset.state_size, 64, dataset.num_assets, 
                                       benchmark_weights=benchmark_weights)
            checkpoint = torch.load(model_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Get predictions for each day
            with torch.no_grad():
                for idx, date in zip(validation_indices, validation_dates):
                    features, port_fwd_ret, port_hist_ret, bm_fwd_ret, bm_hist_ret = dataset.prepare_sequence(idx)
                    weights = model(features.unsqueeze(0))
                    weights_np = weights.squeeze().numpy()
                    
                    # Store for ensemble
                    daily_predictions[date].append(weights_np)
                    
                    # Store individual model predictions
                    model_weights_data.append({
                        'Date': date,
                        **{asset: weight for asset, weight in zip(PORTFOLIO_HOLDINGS, weights_np)}
                    })
            
            # Create DataFrame for this model
            model_predictions[f'Model_{fold+1}'] = pd.DataFrame(model_weights_data).set_index('Date')
                    
        except Exception as e:
            print(f"Error loading model {fold+1}: {e}")
            continue
    
    if not daily_predictions:
        raise ValueError("No valid predictions from ensemble")
    
    # Calculate ensemble statistics
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
    
    # Create DataFrames
    weights_df = pd.DataFrame(weights_data).set_index('Date')
    uncertainty_df = pd.DataFrame(uncertainty_data).set_index('Date')
    
    # Calculate summary statistics
    summary_stats = pd.DataFrame({
        'Mean': weights_df.mean(),
        'Std': weights_df.std(),
        'Min': weights_df.min(),
        'Max': weights_df.max()
    })
    
    # Calculate correlations between models
    model_weights = pd.concat(model_predictions.values())
    model_corr = model_weights.corr()
    
    # Save to Excel with multiple sheets
    excel_path = './output/ensemble_weights.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        weights_df.to_excel(writer, sheet_name='Ensemble_Weights')
        uncertainty_df.to_excel(writer, sheet_name='Daily_Uncertainties')
        summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
        model_corr.to_excel(writer, sheet_name='Model_Correlations')
        
        # Add individual model sheets
        for model_name, model_df in model_predictions.items():
            model_df.to_excel(writer, sheet_name=model_name)
    
    print(f"\nSaved ensemble weights to: {excel_path}")
    
    # Return current allocation
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
    validator = KFoldValidator(dataset=PortfolioDataset(data_path=DATA_FP,
                                                        window_size=252,
                                                        portfolio_columns=sp500_sectors.values()
                                                        ), k=5, num_episodes=1)
    results_df = validator.run()

    # Get ensemble prediction
    portfolio, uncertainty = get_ensemble_allocation(k=5)



