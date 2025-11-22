#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 05:22:16 2025

@author: z3r0
"""

"""
agents.py
Agent Logic: MarketMaker and ForensicAccountant
Updates: Fixed ZeroDivisionError, added safe guards, ensured analyze() exists.
"""

import numpy as np
import pandas as pd
import json
from collections import deque
from typing import Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    action: str
    strength: float
    volatility_state: str
    rsi: Optional[float] = None
    macd: Optional[float] = None
    predicted_price: Optional[float] = None

@dataclass
class HealthScore:
    score: float
    npl_ratio: Optional[float] = None
    liquidity_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_growth: Optional[float] = None
    notes: str = ""

class MarketMaker:
    def __init__(self, ann_model, window_size: int = 30, rsi_period: int = 14):
        self.ann_model = ann_model
        self.window_size = window_size
        self.rsi_period = rsi_period
        self.price_window = deque(maxlen=window_size)
        # Initialize min/max with safe defaults
        self.price_min = float('inf')
        self.price_max = float('-inf')
    
    def receive_tick(self, price: float) -> Optional[Signal]:
        if price <= 0: return None # Ignore bad data
        
        self.price_window.append(price)
        
        if len(self.price_window) < 20:
            return None
        
        # Convert to Series
        prices_series = pd.Series(list(self.price_window))
        
        # 1. RSI (Safe)
        delta = prices_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        loss = loss.replace(0, 1e-9) # Prevent divide by zero
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

        # 2. MACD
        ema12 = prices_series.ewm(span=12, adjust=False).mean()
        ema26 = prices_series.ewm(span=26, adjust=False).mean()
        macd_line = float((ema12 - ema26).iloc[-1])

        # 3. ANN Prediction
        predicted_price = None
        if len(self.price_window) >= 6:
            predicted_price = self._query_ann(np.array(self.price_window))
        
        # 4. Volatility
        volatility_val = prices_series.pct_change().std()
        if pd.isna(volatility_val): volatility_val = 0.0
        
        if volatility_val < 0.01: volatility_state = 'LOW'
        elif volatility_val < 0.03: volatility_state = 'MEDIUM'
        else: volatility_state = 'HIGH'
        
        action, strength = self._generate_signal(price, rsi, macd_line, predicted_price)
        
        return Signal(action, strength, volatility_state, rsi, macd_line, predicted_price)
    
    def _query_ann(self, prices: np.ndarray) -> Optional[float]:
        try:
            last_5 = prices[-5:]
            
            # Robust Min/Max scaling
            local_min = last_5.min()
            local_max = last_5.max()
            
            # Update historical bounds
            self.price_min = min(self.price_min, local_min)
            self.price_max = max(self.price_max, local_max)
            
            denom = self.price_max - self.price_min
            if denom < 1e-8: denom = 1.0 # Prevent divide by zero if flatline
            
            normalized = (last_5 - self.price_min) / denom
            
            import torch
            input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
            
            self.ann_model.eval()
            with torch.no_grad():
                normalized_pred = self.ann_model(input_tensor).item()
            
            denormalized = normalized_pred * denom + self.price_min
            return float(denormalized)
        except Exception as e:
            return None
    
    def _generate_signal(self, current_price, rsi, macd, predicted_price):
        signals = []
        if rsi < 30: signals.append(('BUY', (30-rsi)*3))
        elif rsi > 70: signals.append(('SELL', (rsi-70)*3))
        
        if macd > 0: signals.append(('BUY', abs(macd)*5))
        elif macd < 0: signals.append(('SELL', abs(macd)*5))
        
        if predicted_price and current_price > 0:
            change = ((predicted_price - current_price) / current_price) * 100
            if change > 1.0: signals.append(('BUY', change * 10))
            elif change < -1.0: signals.append(('SELL', abs(change) * 10))
            
        if not signals: return 'HOLD', 0.0
        
        buy_power = sum(s[1] for s in signals if s[0] == 'BUY')
        sell_power = sum(s[1] for s in signals if s[0] == 'SELL')
        
        if buy_power > sell_power: return 'BUY', min(100, buy_power)
        elif sell_power > buy_power: return 'SELL', min(100, sell_power)
        return 'HOLD', 0.0

class ForensicAccountant:
    def __init__(self):
        self.report_data = None
    
    def load_report(self, json_path: str):
        try:
            with open(json_path, 'r') as f:
                self.report_data = json.load(f)
        except Exception:
            self.report_data = None
            
    def _get_nested(self, keys, default=0):
        if not self.report_data: return default
        data = self.report_data
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, {})
            else:
                return default
        if isinstance(data, dict) and 'value' in data: return float(data['value'])
        try: return float(data) if data is not None else default
        except: return default

    # CRITICAL: Ensure this method exists
    def analyze(self) -> HealthScore:
        if not self.report_data:
            return HealthScore(score=50.0, notes="No Data Loaded")

        score = 50.0
        notes = []
        
        npl = self._get_nested(['sacco_metrics', 'npl_ratio'], 0.0)
        if npl > 1.0: npl = npl / 100.0 
            
        liquidity_assets = self._get_nested(['financials', 'balance_sheet', 'current_assets'], 0)
        liquidity_liabs = self._get_nested(['financials', 'balance_sheet', 'current_liabilities'], 1)
        liquidity = liquidity_assets / liquidity_liabs if liquidity_liabs > 0 else 0.0
        
        if liquidity > 1.5: score += 10
        elif liquidity < 1.0 and liquidity > 0: 
            score -= 15
            notes.append("Liquidity Strain")
            
        if npl > 0.10: 
            score -= 20
            notes.append(f"High NPL ({npl:.1%})")
        
        return HealthScore(
            score=max(0, min(100, score)),
            npl_ratio=npl,
            liquidity_ratio=liquidity,
            debt_to_equity=0.0,
            revenue_growth=0.05,
            notes="; ".join(notes)
        )