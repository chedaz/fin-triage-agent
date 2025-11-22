#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 05:18:58 2025

@author: z3r0
"""

"""
models.py
ANN Model for Stock Price Prediction
Architecture: 5:21:21:1 per Paper 1502.06434v1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPredictorANN(nn.Module):
    """
    Feedforward Multi-Layer Perceptron for Stock Price Prediction
    Architecture: 5:21:21:1 per Paper 1502.06434v1
    - Input Layer: 5 neurons (last 5 days closing prices)
    - Hidden Layer 1: 21 neurons
    - Hidden Layer 2: 21 neurons
    - Output Layer: 1 neuron (predicted 6th day price)
    """
    
    def __init__(self):
        super(StockPredictorANN, self).__init__()
        
        # 5:21:21:1 Architecture per Paper 1502.06434v1
        self.input_layer = nn.Linear(5, 21)
        self.hidden1 = nn.Linear(21, 21)
        self.output_layer = nn.Linear(21, 1)
        
        # Activation function (Sigmoid commonly used in the paper's era)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through the network"""
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden1(x))
        x = self.output_layer(x)  # Linear output for regression
        return x


def prepare_sequences(prices: np.ndarray, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert price series into supervised learning sequences
    Input: Last 5 prices -> Output: 6th price
    
    Args:
        prices: 1D array of closing prices
        window_size: Number of historical prices to use (default 5)
    
    Returns:
        X: Input sequences (N, 5)
        y: Target values (N, 1)
    """
    X, y = [], []
    
    for i in range(len(prices) - window_size):
        X.append(prices[i:i+window_size])
        y.append(prices[i+window_size])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)


def train_model(
    model: StockPredictorANN,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 130000,  # Paper specifies ~130k cycles
    learning_rate: float = 0.001,
    early_stopping_patience: int = 50,
    device: str = 'cpu'
) -> dict:
    """
    Train the ANN model with Early Stopping (MVP correction per instructions)
    
    Args:
        model: The StockPredictorANN instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        epochs: Maximum training cycles (default 130000 per paper)
        learning_rate: Learning rate for optimizer
        early_stopping_patience: Stop if no improvement after N epochs
        device: 'cpu' or 'cuda'
    
    Returns:
        Training history dict
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"Starting training for max {epochs} epochs with early stopping (patience={early_stopping_patience})")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Log every 1000 epochs (reduced from 130k to be practical)
            if (epoch + 1) % 1000 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        else:
            if (epoch + 1) % 1000 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f}")
    
    # Load best model if validation was used
    if val_loader:
        model.load_state_dict(torch.load('best_model.pth'))
    
    logger.info("Training complete")
    return history


def normalize_prices(prices: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize prices to [0, 1] range for better training
    
    Returns:
        normalized_prices, min_val, max_val (for denormalization)
    """
    min_val = prices.min()
    max_val = prices.max()
    normalized = (prices - min_val) / (max_val - min_val + 1e-8)
    return normalized, min_val, max_val


def denormalize_price(normalized_price: float, min_val: float, max_val: float) -> float:
    """Denormalize a price back to original scale"""
    return normalized_price * (max_val - min_val) + min_val