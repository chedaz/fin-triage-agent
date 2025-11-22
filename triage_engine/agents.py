#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 05:22:16 2025

@author: z3r0
"""

"""
agents.py
Agent Logic: MarketMaker and ForensicAccountant
Updates: ForensicAccountant now parses specific Finance Parser schemas (Equity, CBK, Sacco).
"""

import numpy as np
import pandas as pd
import json
from collections import deque
from typing import Optional, Dict, List
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
    risk_free_rate: Optional[float] = None  # New field for CBK data
    sentiment: str = "NEUTRAL"
    notes: str = ""

# ... MarketMaker Class remains the same as previous step ...
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
        return ('BUY', min(100, buy_power)) if buy_power > sell_power else ('SELL', min(100, sell_power)) if sell_power > buy_power else ('HOLD', 0.0)

class ForensicAccountant:
    def __init__(self):
        self.report_data = None
    
    def load_report(self, json_path_or_data):
        """Accepts file path or direct dict data"""
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
        # Handle "value" wrapper seen in Equity Group JSON
        if isinstance(data, dict) and 'value' in data: return float(data['value'])
        try: return float(data) if data is not None else default
        except: return default

    def analyze(self) -> HealthScore:
        if not self.report_data:
            return HealthScore(score=50.0, notes="No Data")

        # Determine Document Type from Parser Metadata
        doc_type = self._get_nested(['meta', 'doc_type'], "Unknown")
        entity = self._get_nested(['meta', 'entity_legal_name'], "Unknown Entity")
        
        score = 50.0
        notes = []
        risk_free_rate = None
        
        # --- 1. CORPORATE/BANK ANALYSIS (e.g., Equity Group) ---
        if "Quarterly" in doc_type or "Financial" in doc_type:
            revenue = self._get_nested(['financials', 'income_statement', 'revenue'])
            profit = self._get_nested(['financials', 'income_statement', 'net_profit'])
            
            # Profitability
            if revenue > 0:
                margin = (profit / revenue) * 100
                if margin > 25: 
                    score += 20
                    notes.append(f"High Net Margin ({margin:.1f}%)")
                elif margin < 10:
                    score -= 10
                    notes.append(f"Low Margin ({margin:.1f}%)")
            
            # Risks from Narrative
            risks = self.report_data.get('narrative', {}).get('risks', [])
            if len(risks) > 0:
                score -= (len(risks) * 5)
                notes.append(f"{len(risks)} Risk Factors Identified (e.g., {risks[0][:30]}...)")

        # --- 2. CBK AUCTION ANALYSIS ---
        elif "Auction" in doc_type:
            # Extract T-Bill/Bond Rates
            tbill_91 = self._get_nested(['cbk_metrics', '91_day_t_bill_rate'])
            perf_rate = self._get_nested(['cbk_metrics', 'performance_rate'])
            
            # Check bond auction weighted average if T-bill missing
            if tbill_91 == 0:
                 # Try bond auction
                 bond_rates = self._get_nested(['cbk_metrics', 'bond_auction', 'weighted_average_rates'])
                 if isinstance(bond_rates, list) and bond_rates:
                     tbill_91 = bond_rates[0] # Use the first bond rate as proxy
            
            risk_free_rate = tbill_91
            
            # Liquidity Check (Subscription Rate)
            # Note: Parser output says 2.897 for 289%, or 0.976 for 97%
            if perf_rate > 1.5:
                notes.append(f"High Market Liquidity (Sub: {perf_rate:.1%})")
            elif perf_rate < 0.8:
                notes.append(f"Liquidity Tightness (Sub: {perf_rate:.1%})")
                
            score = 100 # CBK data is "fact", score represents reliability/impact
            notes.append(f"Market Benchmark Set: {risk_free_rate}%")

        # --- 3. SACCO SECTOR ANALYSIS ---
        elif "Supervision Report" in doc_type:
            npl = self._get_nested(['sacco_metrics', 'npl_ratio'])
            if npl > 8.0:
                score -= 15
                notes.append(f"Sector NPL Elevated ({npl}%)")
            
            assets = self._get_nested(['financials', 'balance_sheet', 'total_assets'])
            if assets > 1000: # Trillions scale check based on parser metadata
                score += 10
                notes.append("Sector Assets > 1T")

        return HealthScore(
            score=max(0, min(100, score)),
            npl_ratio=self._get_nested(['sacco_metrics', 'npl_ratio']),
            liquidity_ratio=1.5, # Default if not found
            risk_free_rate=risk_free_rate,
            sentiment="NEGATIVE" if score < 40 else "POSITIVE",
            notes=f"[{entity}]: " + "; ".join(notes)
        )