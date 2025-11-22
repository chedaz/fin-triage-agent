#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 05:21:06 2025

@author: z3r0
"""

"""
stream.py
Playback Engine for Historical CSV Data
Simulates live market feed by streaming row-by-row
Per Multi-Agent Triage Engine Design.pdf
"""

import pandas as pd
import time
from pathlib import Path
from typing import Generator, Dict
import logging

logger = logging.getLogger(__name__)


class MarketStream:
    """
    Mock streaming data source that replays historical CSV data
    Simulates "Hot" pipeline without Kafka (per user constraint)
    """
    
    def __init__(self, data_dir: str = './data'):
        """
        Initialize the market stream
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
    def load_full_history(self, ticker: str) -> pd.DataFrame:
        csv_path = self.data_dir / f"{ticker}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.title()
        
        # Fix Date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        
        # Fix Numeric columns & Missing values (Updated for new Pandas)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Use new ffill/bfill syntax
        df = df.ffill().bfill().fillna(0.0)
        
        return df
    
    def playback(
        self,
        ticker: str,
        speed: float = 0.1,
        start_date: str = None,
        end_date: str = None
    ) -> Generator[Dict, None, None]:
        """
        Stream historical data row-by-row to simulate live feed
        
        Args:
            ticker: Stock ticker symbol (matches CSV filename)
            speed: Delay between ticks in seconds (0.1 = 10 ticks/sec)
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
        
        Yields:
            Dict containing: {
                'Date': datetime,
                'Open': float,
                'High': float,
                'Low': float,
                'Close': float,
                'Volume': int
            }
        """
        csv_path = self.data_dir / f"{ticker}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found for ticker: {ticker}")
        
        logger.info(f"Loading historical data for {ticker} from {csv_path}")
        
        # Read CSV with data normalization
        df = pd.read_csv(csv_path)
        
        # Normalize column names (handle variations)
        df.columns = df.columns.str.strip().str.title()
        
        # Ensure 'Date' column exists and convert to datetime
        if 'Date' not in df.columns:
            raise ValueError(f"CSV must contain 'Date' column. Found: {df.columns.tolist()}")
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        # Filter by date range if specified
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Fill NaN values with forward fill, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure numeric columns are floats
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        logger.info(f"Streaming {len(df)} ticks for {ticker} from {df['Date'].min()} to {df['Date'].max()}")
        
        # Stream row-by-row
        for idx, row in df.iterrows():
            tick = {
                'Date': row['Date'],
                'Open': float(row.get('Open', 0)),
                'High': float(row.get('High', 0)),
                'Low': float(row.get('Low', 0)),
                'Close': float(row.get('Close', 0)),
                'Volume': int(row.get('Volume', 0))
            }
            
            yield tick
            
            # Simulate real-time delay
            if speed > 0:
                time.sleep(speed)
    
    # def load_full_history(self, ticker: str) -> pd.DataFrame:
    #     """
    #     Load entire CSV into memory (for batch training)
        
    #     Args:
    #         ticker: Stock ticker symbol
        
    #     Returns:
    #         DataFrame with OHLCV data
    #     """
    #     csv_path = self.data_dir / f"{ticker}.csv"
        
    #     if not csv_path.exists():
    #         raise FileNotFoundError(f"CSV not found: {csv_path}")
        
    #     df = pd.read_csv(csv_path)
    #     df.columns = df.columns.str.strip().str.title()
    #     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    #     df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        
    #     # Normalize numeric columns
    #     numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    #     for col in numeric_cols:
    #         if col in df.columns:
    #             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill').fillna(0.0)
        
    #     logger.info(f"Loaded {len(df)} records for {ticker}")
        
    #     return df