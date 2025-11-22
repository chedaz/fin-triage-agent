#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 05:21:06 2025

@author: z3r0
"""

"""
stream.py
Playback Engine for Historical CSV Data
Updates: Added column normalization to handle NSE 'Mean Price'/'Price' headers
"""
import pandas as pd
import time
from pathlib import Path
from typing import Generator, Dict
import logging

logger = logging.getLogger(__name__)

class MarketStream:
    def __init__(self, data_dir: str = './data'):
        self.data_dir = Path(data_dir)
        
        # Mapping common NSE/Financial headers to standard OHLCV
        self.column_map = {
            'Mean Price': 'Close',
            'Price': 'Close',
            'Adj Close': 'Close',
            'Previous': 'Close',      # Fallback if no close
            'Day High': 'High',
            'Day Low': 'Low',
            'Tot Vol': 'Volume',
            'Vol': 'Volume',
            'Code': 'Ticker'
        }
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to Open, High, Low, Close, Volume"""
        # Clean whitespace and title case existing columns
        df.columns = df.columns.str.strip().str.title()
        
        # Rename based on map
        # Note: Title casing turns 'Mean Price' -> 'Mean Price', but 'CLOSE' -> 'Close'
        # We need to match the title-cased keys in self.column_map if they match exactly,
        # or do a flexible lookup.
        
        new_names = {}
        for col in df.columns:
            # Check exact map
            if col in self.column_map:
                new_names[col] = self.column_map[col]
            # Check title case map (e.g., 'Mean price' -> 'Mean Price')
            elif col.title() in self.column_map:
                new_names[col] = self.column_map[col.title()]
        
        df.rename(columns=new_names, inplace=True)
        
        # CRITICAL: Ensure 'Close' exists
        if 'Close' not in df.columns:
            # If we have Open but no Close, use Open
            if 'Open' in df.columns:
                df['Close'] = df['Open']
            # If we have nothing, look for ANY float column
            else:
                float_cols = df.select_dtypes(include=['float', 'int']).columns
                if len(float_cols) > 0:
                    logger.warning(f"No 'Close' column found. Defaulting to '{float_cols[0]}'")
                    df['Close'] = df[float_cols[0]]
        
        # Fill missing OHLC if Close exists
        if 'Close' in df.columns:
            if 'Open' not in df.columns: df['Open'] = df['Close']
            if 'High' not in df.columns: df['High'] = df['Close']
            if 'Low' not in df.columns: df['Low'] = df['Close']
            if 'Volume' not in df.columns: df['Volume'] = 0
            
        return df

    def playback(
        self,
        ticker: str,
        speed: float = 0.1,
        start_date: str = None,
        end_date: str = None
    ) -> Generator[Dict, None, None]:
        
        try:
            df = self.load_full_history(ticker)
        except FileNotFoundError:
            return

        # Filter by date range if specified
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        
        logger.info(f"Streaming {len(df)} ticks for {ticker}")
        
        for idx, row in df.iterrows():
            yield {
                'Date': row['Date'],
                'Open': float(row['Open']),
                'High': float(row['High']),
                'Low': float(row['Low']),
                'Close': float(row['Close']),
                'Volume': int(row['Volume'])
            }
            if speed > 0:
                time.sleep(speed)
    
    def load_full_history(self, ticker: str) -> pd.DataFrame:
        csv_path = self.data_dir / f"{ticker}.csv"
        
        # Fallback: check if user provided file without extension or with different case
        if not csv_path.exists():
            # Try finding any file starting with ticker
            candidates = list(self.data_dir.glob(f"{ticker}*.csv"))
            if candidates:
                csv_path = candidates[0]
            else:
                raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # 1. Normalize Columns immediately
        df = self._normalize_columns(df)
        
        # 2. Fix Date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        else:
             raise ValueError(f"CSV {csv_path} missing 'Date' column")
        
        # 3. Numeric Conversion
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # 4. Fill Gaps
        df = df.ffill().bfill().fillna(0.0)
        
        return df