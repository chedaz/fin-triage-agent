#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 08:10:31 2025

@author: z3r0
"""

"""
MCP Server for Triage Engine
"""
from mcp.server.fastmcp import FastMCP
import json
import tempfile
import os
from triage_engine.agents import MarketMaker, ForensicAccountant
from triage_engine.optimizer import optimize_portfolio
from triage_engine.stream import MarketStream
from triage_engine.models import StockPredictorANN

mcp = FastMCP("Triage Engine Server")

@mcp.tool()
def consult_market_maker(ticker: str) -> str:
    try:
        ann_model = StockPredictorANN()
        market_maker = MarketMaker(ann_model) # Positional arg
        stream = MarketStream()
        try:
            history = stream.load_full_history(ticker)
        except FileNotFoundError:
            return f"Error: No history found for {ticker}"

        recent_history = history.tail(60)
        if len(recent_history) == 0: return f"Error: Empty data for {ticker}"
        
        last_signal = None
        for _, row in recent_history.iterrows():
            price = row.get('close') or row.get('Close') or 0
            last_signal = market_maker.receive_tick(float(price))
        
        if last_signal is None:
            return "Signal: HOLD | Strength: 0 | ANN_Pred: N/A"
        
        ann_pred = last_signal.predicted_price
        ann_str = f"{ann_pred:.2f}" if ann_pred else "N/A"
        
        return (
            f"Signal: {last_signal.action} | "
            f"Strength: {int(last_signal.strength)} | "
            f"Volatility: {last_signal.volatility_state} | "
            f"ANN_Pred: {ann_str}"
        )
    except Exception as e:
        return f"ERROR in Market Maker: {str(e)}"

@mcp.tool()
def consult_forensic_accountant(ticker: str, report_context: str = "") -> str:
    try:
        accountant = ForensicAccountant()
        
        # SAFEGUARD: If context is not a valid file path or JSON, use mock
        use_mock = True
        report_path = ""
        
        if report_context and os.path.exists(report_context):
             report_path = report_context
             use_mock = False
        elif report_context and report_context.strip().startswith("{"):
             try:
                 # Write JSON string to temp file
                 data = json.loads(report_context)
                 with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                     json.dump(data, f)
                     report_path = f.name
                     use_mock = False
             except:
                 pass

        if use_mock:
            # Generate Mock Data if no valid file provided
            mock_report = {
                "ticker": ticker,
                "sacco_metrics": {"npl_ratio": 12.5},
                "financials": {
                    "balance_sheet": {
                        "current_assets": 12000000, 
                        "current_liabilities": 8000000
                    }
                }
            }
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(mock_report, f)
                report_path = f.name

        accountant.load_report(report_path)
        
        if not hasattr(accountant, 'analyze'):
            return "ERROR: ForensicAccountant class malformed (analyze method missing)"
            
        health_score = accountant.analyze()
        
        # Cleanup
        if use_mock:
            try: os.unlink(report_path)
            except: pass
        
        return (
            f"Score: {health_score.score}/100 | "
            f"NPL: {health_score.npl_ratio:.1%} | "
            f"Liquidity: {health_score.liquidity_ratio:.2f} | "
            f"Notes: {health_score.notes}"
        )
    except Exception as e:
        return f"ERROR in Forensic Accountant: {str(e)}"

@mcp.tool()
def run_portfolio_optimization(tickers: str, risk_aversion: float = 0.5) -> str:
    return f"Optimal Weights: Equal Weight for {tickers} (MVP Mock)"

if __name__ == "__main__":
    mcp.run(transport='stdio')