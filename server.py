#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 08:10:31 2025

@author: z3r0
"""

"""
MCP Server for Triage Engine
Updates: Added consult_macro_analyst tool.
"""
from mcp.server.fastmcp import FastMCP
import json
import tempfile
import os
import torch
import logging

from triage_engine.agents import MarketMaker, ForensicAccountant
from triage_engine.stream import MarketStream
from triage_engine.models import StockPredictorANN

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("mcp_server")

mcp = FastMCP("Triage Engine Server")

def load_trained_model(ticker: str):
    model = StockPredictorANN()
    model_path = f"models/{ticker}_model.pth"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
    return model

@mcp.tool()
def consult_market_maker(ticker: str) -> str:
    try:
        ann_model = load_trained_model(ticker)
        market_maker = MarketMaker(ann_model)
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
        
        ann_str = f"{last_signal.predicted_price:.2f}" if last_signal.predicted_price else "N/A"
        
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
        use_mock = True
        report_path = ""
        
        if report_context and os.path.exists(report_context):
             report_path = report_context
             use_mock = False
        elif report_context and report_context.strip().startswith("{"):
             try:
                 data = json.loads(report_context)
                 with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                     json.dump(data, f)
                     report_path = f.name
                     use_mock = False
             except: pass

        if use_mock:
            mock_report = {"meta": {"doc_type": "Financial"}, "sacco_metrics": {"npl_ratio": 12.5}}
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(mock_report, f)
                report_path = f.name

        accountant.load_report(report_path)
        health_score = accountant.analyze()
        
        if use_mock:
            try: os.unlink(report_path)
            except: pass
        
        return (
            f"Score: {health_score.score:.1f}/100 | "
            f"NPL: {health_score.npl_ratio:.1%} | "
            f"Notes: {health_score.notes}"
        )
    except Exception as e:
        return f"ERROR in Forensic Accountant: {str(e)}"

@mcp.tool()
def consult_macro_analyst(report_context: str) -> str:
    """
    Consult the Macro Analyst for Bond Market and Risk-Free Rate analysis.
    """
    try:
        # Reuse Forensic class logic
        analyst = ForensicAccountant()
        
        # Handle File Path
        if report_context and os.path.exists(report_context):
             analyst.load_report(report_context)
        elif report_context and report_context.strip().startswith("{"):
             try:
                 data = json.loads(report_context)
                 analyst.load_report(data) # Load dict directly
             except: return "Error: Invalid JSON"
        else:
            return "Error: Please provide a valid file path to the macro report."

        score = analyst.analyze()
        
        rf_str = f"{score.risk_free_rate}%" if score.risk_free_rate else "N/A"
        
        return (
            f"Risk-Free Rate: {rf_str} | "
            f"Sentiment: {score.sentiment} | "
            f"Notes: {score.notes}"
        )
    except Exception as e:
        return f"ERROR in Macro Analyst: {str(e)}"

@mcp.tool()
def run_portfolio_optimization(tickers: str, risk_aversion: float = 0.5) -> str:
    return f"Optimal Weights: Equal Weight for {tickers} (MVP Mock)"

if __name__ == "__main__":
    mcp.run(transport='stdio')