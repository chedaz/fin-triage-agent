#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 05:22:16 2025

@author: z3r0
"""

"""
agents.py
Agent Logic: MarketMaker and ForensicAccountant
Updates: Fixed NPL Math Bug (13.6 -> 13.6% not 1360%)
"""

import numpy as np
import pandas as pd
import json
from collections import deque
from typing import Optional
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
    risk_free_rate: Optional[float] = None
    sentiment: str = "NEUTRAL"
    notes: str = ""

class MarketMaker:
    def __init__(self, ann_model, window_size: int = 30, rsi_period: int = 14):
        self.ann_model = ann_model
        self.window_size = window_size
        self.rsi_period = rsi_period
        self.price_window = deque(maxlen=window_size)
        self.price_min = float('inf')
        self.price_max = float('-inf')
    
    def receive_tick(self, price: float) -> Optional[Signal]:
        if price <= 0: return None
        self.price_window.append(price)
        if len(self.price_window) < 20: return None
        
        prices_series = pd.Series(list(self.price_window))
        
        # Indicators
        delta = prices_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().replace(0, 1e-9)
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs)).iloc[-1]) if not pd.isna((100 - (100 / (1 + rs))).iloc[-1]) else 50.0

        ema12 = prices_series.ewm(span=12, adjust=False).mean()
        ema26 = prices_series.ewm(span=26, adjust=False).mean()
        macd_line = float((ema12 - ema26).iloc[-1])

        predicted_price = None
        if len(self.price_window) >= 6:
            predicted_price = self._query_ann(np.array(self.price_window))
        
        volatility_val = prices_series.pct_change().std()
        if pd.isna(volatility_val): volatility_val = 0.0
        volatility_state = 'HIGH' if volatility_val > 0.03 else 'MEDIUM' if volatility_val > 0.01 else 'LOW'
        
        action, strength = self._generate_signal(price, rsi, macd_line, predicted_price)
        return Signal(action, strength, volatility_state, rsi, macd_line, predicted_price)
    
    def _query_ann(self, prices: np.ndarray) -> Optional[float]:
        try:
            last_5 = prices[-5:]
            self.price_min = min(self.price_min, last_5.min())
            self.price_max = max(self.price_max, last_5.max())
            denom = self.price_max - self.price_min
            if denom < 1e-8: denom = 1.0
            normalized = (last_5 - self.price_min) / denom
            
            import torch
            input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
            self.ann_model.eval()
            with torch.no_grad():
                normalized_pred = self.ann_model(input_tensor).item()
            return float(normalized_pred * denom + self.price_min)
        except: return None
    
    def _generate_signal(self, current_price, rsi, macd, predicted_price):
        signals = []
        if rsi < 30: signals.append(('BUY', (30-rsi)*3))
        elif rsi > 70: signals.append(('SELL', (rsi-70)*3))
        if macd > 0: signals.append(('BUY', abs(macd)*5))
        elif macd < 0: signals.append(('SELL', abs(macd)*5))
        
        if predicted_price:
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
    
    def load_report(self, json_path_or_data):
        if isinstance(json_path_or_data, dict):
            self.report_data = json_path_or_data
        else:
            try:
                with open(json_path_or_data, 'r') as f:
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

    def analyze(self) -> HealthScore:
        if not self.report_data:
            return HealthScore(score=50.0, notes="No Data")

        doc_type = self._get_nested(['meta', 'doc_type'], "Unknown")
        score = 50.0
        notes = []
        risk_free_rate = None
        
        # --- FIX: Robust NPL Normalization ---
        raw_npl = self._get_nested(['sacco_metrics', 'npl_ratio'], 0.0)
        
        # Heuristic: If NPL > 1.0, assume it's a percentage (e.g., 13.6) -> convert to 0.136
        # If NPL <= 1.0, assume it's a decimal (e.g., 0.136) -> keep as 0.136
        if raw_npl > 1.0: 
            npl = raw_npl / 100.0
        else:
            npl = raw_npl

        # --- Logic for Banks/Saccos ---
        if "Quarterly" in doc_type or "Financial" in doc_type or "Sacco" in doc_type:
            revenue = self._get_nested(['financials', 'income_statement', 'revenue'])
            profit = self._get_nested(['financials', 'income_statement', 'net_profit'])
            
            if revenue > 0:
                margin = (profit / revenue) * 100
                if margin > 25: score += 20
                elif margin < 10: score -= 10
            
            if npl > 0.10: 
                score -= 20
                notes.append(f"High NPL ({npl:.1%})")
                
            # Check risks
            risks = self.report_data.get('narrative', {}).get('risks', [])
            if len(risks) > 0: score -= 5

        # --- Logic for Macro/Auctions ---
        elif "Auction" in doc_type:
            tbill_91 = self._get_nested(['cbk_metrics', '91_day_t_bill_rate'])
            perf_rate = self._get_nested(['cbk_metrics', 'performance_rate'])
            
            if tbill_91 == 0:
                 bond_rates = self._get_nested(['cbk_metrics', 'bond_auction', 'weighted_average_rates'])
                 if isinstance(bond_rates, list) and bond_rates:
                     tbill_91 = bond_rates[0]
            
            risk_free_rate = tbill_91
            score = 100
            notes.append(f"Market Yield: {risk_free_rate}%")

        return HealthScore(
            score=max(0, min(100, score)),
            npl_ratio=npl,
            liquidity_ratio=1.5,
            risk_free_rate=risk_free_rate,
            sentiment="NEGATIVE" if score < 40 else "POSITIVE",
            notes="; ".join(notes)
        )