#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 06:32:54 2025

@author: z3r0
"""

"""
Triage Engine MCP System Launcher
Orchestrates the startup of both server and client for end-to-end execution.
"""
import asyncio
import os
import sys
# Import the Factory Function (wrapper), not the specific classes
from orchestrator import TriageOrchestrator
import subprocess


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

async def main():
    """
    Main entry point - starts the full system and runs a test query.
    """
    print("="*70)
    print("  TRIAGE ENGINE - DISTRIBUTED MULTI-AGENT SYSTEM (MCP)")
    print("="*70)
    print()
    
    # Check for API key (DeepSeek/OpenAI or Gemini/Google)
    # We check specific keys first, then fall back to ANTHROPIC_API_KEY for backward compat
    
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("❌ ERROR: LLM_API_KEY environment variable not set")
        print("   For DeepSeek: export LLM_API_KEY='sk-...'")
        print("   For Gemini:   export LLM_API_KEY='AIza...'")
        sys.exit(1)
        
    # Get server path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(current_dir, "server.py")
    
    if not os.path.exists(server_path):
        print(f"❌ ERROR: server.py not found at {server_path}")
        sys.exit(1)
        
    print(f"📁 Server script: {server_path}")
    print(f"🔌 Transport: stdio")
    print()
    
    # Initialize orchestrator using the Factory Function
    # This automatically picks DeepSeek/Gemini based on LLM_PROVIDER env var
    try:
        print("🚀 Initializing Triage Manager (Orchestrator)...")
        orchestrator = TriageOrchestrator(api_key)
    except Exception as e:
        print(f"❌ ERROR Initializing Orchestrator: {e}")
        sys.exit(1)
        
    # Connect to MCP server (this starts the server subprocess)
    print("🔗 Connecting to MCP Server...")
    await orchestrator.connect_to_server(server_path)
    print()
    
    # Run test query
    print("="*70)
    print("  TEST EXECUTION")
    print("="*70)
    print()
    
    test_query = "Analyze EQTY status."
    print(f"📊 QUERY: {test_query}")
    print()
    print("⏳ Processing (calling Market Maker + Forensic Accountant)...")
    print()
    
    try:
        response = await orchestrator.query(test_query)
        
        print("="*70)
        print("  TRIAGE MANAGER DECISION")
        print("="*70)
        print()
        print(response)
        print()
            
    except Exception as e:
        print(f"\n❌ ERROR during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("="*70)
        print("🛑 Shutting down...")
        await orchestrator.disconnect()
        print("✓ Disconnected from MCP Server")
        print("✓ Session closed")
        print()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)