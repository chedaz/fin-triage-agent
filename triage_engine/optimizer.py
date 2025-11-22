#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 05:20:14 2025

@author: z3r0
"""

"""
optimizer.py
Constrained Mean-Variance Optimization (MVO)
Per "Untitled document (1).pdf" specification
Uses cvxpy for convex optimization
"""

import cvxpy as cp
import numpy as np
from sklearn.covariance import LedoitWolf
import logging

logger = logging.getLogger(__name__)


def optimize_portfolio(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_aversion: float = 1.0,
    max_weight: float = 0.25,
    min_weight: float = 0.0
) -> dict:
    """
    Solve the Mean-Variance Optimization problem
    
    Objective: Maximize w^T μ - (λ/2) w^T Σ w
    
    Args:
        expected_returns (μ): Expected returns for each asset (N,)
        covariance_matrix (Σ): Covariance matrix of returns (N, N)
        risk_aversion (λ): Risk aversion parameter (higher = more conservative)
        max_weight: Maximum allocation to single asset (e.g., 0.25 = 25%)
        min_weight: Minimum allocation (0 for long-only)
    
    Returns:
        dict with 'weights', 'expected_return', 'risk', 'status'
    
    Constraints per document:
    - Sum of weights = 1 (full investment)
    - w >= 0 (long only)
    - w <= max_weight (diversification)
    """
    n_assets = len(expected_returns)
    
    # Decision variable: portfolio weights
    w = cp.Variable(n_assets)
    
    # Objective function: Maximize w^T μ - (λ/2) w^T Σ w
    # Note: cvxpy minimizes, so we negate to maximize
    portfolio_return = w.T @ expected_returns
    portfolio_risk = cp.quad_form(w, covariance_matrix)
    
    objective = cp.Maximize(portfolio_return - (risk_aversion / 2) * portfolio_risk)
    
    # Constraints per Untitled document (1).pdf
    constraints = [
        cp.sum(w) == 1,           # Full investment
        w >= min_weight,          # Long only (or minimum weight)
        w <= max_weight           # Max single asset allocation
    ]
    
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.ECOS)  # Use ECOS solver (good for QP)
        
        if problem.status in ['optimal', 'optimal_inaccurate']:
            weights = w.value
            exp_return = portfolio_return.value
            risk = np.sqrt(portfolio_risk.value)
            
            logger.info(f"Optimization successful. Expected Return: {exp_return:.4f}, Risk (σ): {risk:.4f}")
            
            return {
                'weights': weights,
                'expected_return': exp_return,
                'risk': risk,
                'status': problem.status
            }
        else:
            logger.warning(f"Optimization failed with status: {problem.status}")
            # Return equal weights as fallback
            return {
                'weights': np.ones(n_assets) / n_assets,
                'expected_return': None,
                'risk': None,
                'status': problem.status
            }
    
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        # Fallback to equal weights
        return {
            'weights': np.ones(n_assets) / n_assets,
            'expected_return': None,
            'risk': None,
            'status': 'error'
        }


def calculate_expected_returns(
    historical_returns: np.ndarray,
    ann_predictions: np.ndarray = None,
    blend_weight: float = 0.5
) -> np.ndarray:
    """
    Calculate expected returns blending historical and ANN forecasts
    Per spec: μ = Expected Returns (blended historical + ANN forecast)
    
    Args:
        historical_returns: Historical return series (T, N)
        ann_predictions: ANN predicted next-day returns (N,) [optional]
        blend_weight: Weight for ANN predictions (0.5 = 50/50 blend)
    
    Returns:
        Expected returns vector (N,)
    """
    # Historical mean return
    historical_mean = np.mean(historical_returns, axis=0)
    
    if ann_predictions is not None and blend_weight > 0:
        # Blend historical + ANN forecast
        mu = (1 - blend_weight) * historical_mean + blend_weight * ann_predictions
    else:
        mu = historical_mean
    
    return mu


def calculate_covariance_ledoit_wolf(returns: np.ndarray) -> np.ndarray:
    """
    Calculate covariance matrix with Ledoit-Wolf shrinkage
    Per spec: "use Ledoit-Wolf shrinkage"
    
    This reduces estimation error in the covariance matrix, especially
    important for small sample sizes or high-dimensional portfolios.
    
    Args:
        returns: Return series (T, N) where T=time, N=assets
    
    Returns:
        Shrunk covariance matrix (N, N)
    """
    lw = LedoitWolf()
    cov_matrix, _ = lw.fit(returns).covariance_, lw.shrinkage_
    
    logger.info(f"Ledoit-Wolf shrinkage applied. Shrinkage intensity: {lw.shrinkage_:.4f}")
    
    return cov_matrix