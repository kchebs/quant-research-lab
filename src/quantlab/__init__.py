"""Quant Research Lab: DS/ML/AI research pipeline for equity markets.

Subpackages
-----------
data         : price loading (yfinance + local cache) and fundamentals snapshots
stats        : hypothesis testing and statistical research utilities
screening    : rule-based investment screening engine
indicators   : technical indicators (clean-room implementations)
forecasting  : naive next-day price baselines (CI MAE/RMSE smoke)
backtest     : market simulator with transaction costs and performance metrics
learners     : from-scratch trees, bagging, linear regression, InsaneLearner
strategies   : manual, ML, and theoretically optimal trading strategies
rl           : tabular Q-learning (trading env + grid worlds)
portfolio    : Sharpe-maximizing portfolio optimization
sim          : martingale / probability simulations
experiments  : learner assessment and synthetic best-for-* datasets
"""

__version__ = "1.1.0"
