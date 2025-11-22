#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 09:23:29 2025

@author: z3r0
"""

"""
run_parser_simulation.py
Tests the integration of Finance Parser outputs into the Triage Engine.
"""
import asyncio
import os
import json
import tempfile
from orchestrator import TriageOrchestrator
import subprocess
from mock_data import EQUITY_GROUP_Q3, CBK_AUCTION_OCT, CBK_BOND_NOV, SACCO_SECTOR_24

def load_dotenv(path=".env"):
    command = f"set -a && source {path} && env"
    result = subprocess.run(
        ["/bin/bash", "-c", command],
        capture_output=True,
        text=True
        )
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        os.environ[key] = value
        
load_dotenv()

async def simulate_scenario(orchestrator, name, ticker, json_data, query):
    print(f"\n--- SCENARIO: {name} ---")
    print(f"📄 Injecting Parser Output for: {ticker}")
    
    # 1. Write the Python Dict to a Temp JSON File (Simulating the Parser saving a file)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_data, f)
        temp_path = f.name
    
    # 2. Construct the Orchestrator Query
    # We explicitly tell the Orchestrator where to look, simulating the system passing the path
    full_query = f"{query} Use the financial report located at: {temp_path}"
    
    print(f"🤖 User Query: {query}")
    print("⏳ Triage Engine thinking...")
    
    try:
        response = await orchestrator.query(full_query)
        print("\n📝 TRIAGE RESPONSE:")
        print(response)
    finally:
        os.unlink(temp_path)

async def main():
    api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set LLM_API_KEY")
        return

    # Initialize
    orchestrator = TriageOrchestrator(api_key)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(current_dir, "server.py")
    await orchestrator.connect_to_server(server_path)

    # --- SIMULATION 1: Corporate Analysis (Equity Group) ---
    await simulate_scenario(
        orchestrator,
        "Equity Group Q3 Analysis",
        "EQTY",
        EQUITY_GROUP_Q3,
        "Analyze Equity Group's health based on the Q3 update."
    )

    # --- SIMULATION 2: Macro Analysis (CBK Bond) ---
    # Here the agent should pick up the high yield (12.34%) and oversubscription
    await simulate_scenario(
        orchestrator,
        "CBK Bond Auction Analysis",
        "CBK",
        CBK_BOND_NOV,
        "What is the current sentiment in the bond market and the risk-free rate?"
    )

    await orchestrator.disconnect()

if __name__ == "__main__":
    asyncio.run(main())