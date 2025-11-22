#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 05:22:44 2025

@author: z3r0
"""

"""
main.py
CLI Entrypoint for Financial Triage Engine MVP
Orchestrates: Stream -> Agents -> Optimizer -> Recommendations
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import StockPredictorANN, prepare_sequences, train_model, normalize_prices, denormalize_price
from optimizer import optimize_portfolio, calculate_expected_returns, calculate_covariance_ledoit_wolf
from stream import MarketStream
from agents import MarketMaker, ForensicAccountant

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def train_or_load_model(ticker: str, data_dir: str = './data', force_retrain: bool = False) -> tuple:
    """
    Train ANN on historical data or load pre-trained model
    
    Returns:
        (model, price_min, price_max) tuple
    """
    model_path = Path(f'models/{ticker}_model.pth')
    model_path.parent.mkdir(exist_ok=True)
    
    model = StockPredictorANN()
    
    if model_path.exists() and not force_retrain:
        logger.info(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        
        # Load normalization params
        norm_path = Path(f'models/{ticker}_norm.npz')
        if norm_path.exists():
            norm_data = np.load(norm_path)
            price_min, price_max = norm_data['min'], norm_data['max']
        else:
            price_min, price_max = None, None
        
        return model, price_min, price_max
    
    # Train new model
    logger.info(f"Training new model for {ticker}...")
    
    stream = MarketStream(data_dir)
    df = stream.load_full_history(ticker)
    
    prices = df['Close'].values
    
    # Normalize prices
    normalized_prices, price_min, price_max = normalize_prices(prices)
    
    # Prepare sequences (5 -> 1 prediction)
    X, y = prepare_sequences(normalized_prices, window_size=5)
    
    # 80/20 train/test split per Paper 1502.06434v1
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train with early stopping (max 130k epochs per paper, but early stop if converged)
    train_model(
        model,
        train_loader,
        val_loader=test_loader,
        epochs=10000,  # Reduced for MVP; paper specifies 130k but early stopping handles it
        learning_rate=0.001,
        early_stopping_patience=50
    )
    
    # Save model and normalization params
    torch.save(model.state_dict(), model_path)
    np.savez(f'models/{ticker}_norm.npz', min=price_min, max=price_max)
    
    logger.info(f"Model saved to {model_path}")
    
    return model, price_min, price_max


def run_triage_engine(args):
    """
    Main orchestration loop
    """
    logger.info("=" * 80)
    logger.info("FINANCIAL TRIAGE ENGINE MVP - STARTING")
    logger.info("=" * 80)
    
    # Step 1: Train or load ANN model
    model, price_min, price_max = train_or_load_model(
        args.ticker,
        data_dir=args.data_dir,
        force_retrain=args.retrain
    )
    
    # Step 2: Initialize agents
    market_maker = MarketMaker(ann_model=model, window_size=30)
    forensic_agent = ForensicAccountant()
    
    # Load financial report if provided
    if args.report:
        forensic_agent.load_report(args.report)
        health = forensic_agent.analyze()
        logger.info(f"Financial Health Score: {health.score:.1f}/100 | {health.notes}")
    
    # Step 3: Initialize market stream
    stream = MarketStream(data_dir=args.data_dir)
    
    # Step 4: Playback simulation
    logger.info(f"\nStarting market playback for {args.ticker}...")
    logger.info("-" * 80)
    
    tick_count = 0
    portfolio_rebalance_interval = args.rebalance_days  # Rebalance every N ticks
    
    price_history = []
    return_history = []
    
    for tick in stream.playback(
        ticker=args.ticker,
        speed=args.speed,
        start_date=args.start_date,
        end_date=args.end_date
    ):
        tick_count += 1
        current_price = tick['Close']
        current_date = tick['Date']
        
        # Agent analysis
        signal = market_maker.receive_tick(current_price)
        
        # Store for portfolio optimization
        price_history.append(current_price)
        if len(price_history) > 1:
            daily_return = (price_history[-1] - price_history[-2]) / price_history[-2]
            return_history.append(daily_return)
        
        # Log decision
        if signal:
            log_msg = (
                f"[{current_date.date()}] [{args.ticker}] "
                f"Price: {current_price:.2f} | "
                f"ANN Pred: {signal.predicted_price:.2f if signal.predicted_price else 'N/A'} | "
                f"RSI: {signal.rsi:.1f if signal.rsi else 'N/A'} | "
                f"MACD: {signal.macd:.3f if signal.macd else 'N/A'} | "
                f"Volatility: {signal.volatility_state} | "
                f"Action: {signal.action} (Strength: {signal.strength:.1f})"
            )
            logger.info(log_msg)
        
        # Portfolio rebalancing (every N days of simulation)
        if tick_count % portfolio_rebalance_interval == 0 and len(return_history) > 30:
            logger.info("\n" + "="*80)
            logger.info(f"PORTFOLIO REBALANCING TRIGGERED (Day {tick_count})")
            logger.info("="*80)
            
            # For MVP, we'll simulate a simple single-asset "portfolio"
            # In a real scenario, you'd have multiple tickers
            
            # Calculate expected return (blend historical + ANN prediction)
            historical_mean_return = np.mean(return_history[-60:])  # Last 60 days
            
            if signal and signal.predicted_price:
                ann_predicted_return = (signal.predicted_price - current_price) / current_price
                mu_val = 0.5 * historical_mean_return + 0.5 * ann_predicted_return
            else:
                mu_val = historical_mean_return

            # --- TRIAGE LOGIC INTEGRATION ---
            # If Forensic Score is low (<40), severely penalize expected return
            if 'health' in locals() and health.score < 40:
                logger.warning(f"Forensic Penalty Applied: Score {health.score}")
                mu_val = mu_val * 0.5  # Cut expected return in half due to fundamental risk
            
            mu = np.array([mu_val])
            
            # Calculate covariance (for single asset, just variance)
            recent_returns = np.array(return_history[-60:]).reshape(-1, 1)
            sigma = calculate_covariance_ledoit_wolf(recent_returns)
            
            # Optimize (for single asset, this will just return w=1.0, but demonstrates the framework)
            result = optimize_portfolio(
                expected_returns=mu,
                covariance_matrix=sigma,
                risk_aversion=args.risk_aversion,
                max_weight=1.0  # Single asset
            )
            
            logger.info(f"Optimal Weight: {result['weights'][0]:.2%}")
            logger.info(f"Expected Return: {result['expected_return']:.4f}" if result['expected_return'] else "N/A")
            logger.info(f"Portfolio Risk (σ): {result['risk']:.4f}" if result['risk'] else "N/A")
            logger.info("="*80 + "\n")
        
        # Limit simulation if max_ticks specified
        if args.max_ticks and tick_count >= args.max_ticks:
            logger.info(f"\nReached max ticks limit ({args.max_ticks}). Stopping simulation.")
            break
    
    logger.info("\n" + "="*80)
    logger.info("SIMULATION COMPLETE")
    logger.info(f"Total ticks processed: {tick_count}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Financial Triage Engine MVP - Stock Market Analysis & Portfolio Optimization"
    )
    
    # Required arguments
    parser.add_argument(
        'ticker',
        type=str,
        help='Stock ticker symbol (must match CSV filename in data directory)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data',
        help='Directory containing CSV files (default: ../data)'
    )
    
    parser.add_argument(
        '--report',
        type=str,
        help='Path to JSON financial report for forensic analysis'
    )
    
    parser.add_argument(
        '--speed',
        type=float,
        default=0.0,
        help='Playback speed in seconds between ticks (default: 0.0 = fastest)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for simulation (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for simulation (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--rebalance-days',
        type=int,
        default=30,
        help='Rebalance portfolio every N ticks/days (default: 30)'
    )
    
    parser.add_argument(
        '--risk-aversion',
        type=float,
        default=1.0,
        help='Risk aversion parameter λ for optimization (default: 1.0)'
    )
    
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Force retrain ANN model even if saved model exists'
    )
    
    parser.add_argument(
        '--max-ticks',
        type=int,
        help='Maximum number of ticks to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.error("Please create the directory and add CSV files named as <TICKER>.csv")
        return
    
    # Run the engine
    try:
        run_triage_engine(args)
    except KeyboardInterrupt:
        logger.info("\n\nSimulation interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == '__main__':
    main()