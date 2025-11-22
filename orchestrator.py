#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 07:42:15 2025

@author: z3r0
"""

"""
Triage Manager Orchestrator (Universal)
Supports: DeepSeek (via OpenAI SDK), Gemini (Google GenAI), and Anthropic.
Fixed: Correctly handles MCP stdio_client context manager lifecycle.
"""
import asyncio
import os
import json
import sys

# --- PROVIDER IMPORTS ---
try:
    from openai import AsyncOpenAI
except ImportError:
    pass  # DeepSeek optional

try:
    import google.generativeai as genai
    from google.generativeai.types import Tool as GeminiTool, FunctionDeclaration
    from google.ai.generativelanguage import Part, FunctionResponse
except ImportError:
    pass  # Gemini optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- SYSTEM PROMPT ---
TRIAGE_MANAGER_PROMPT = """You are the **Triage Manager**, a Senior Portfolio Manager.
YOUR IDENTITY:
- Decisive, analytical, and risk-aware.
- Prioritize capital preservation.
- NEVER access raw data directly; rely exclusively on your tools.

YOUR TOOLS:
1. consult_market_maker: Technicals (price, momentum, ANN).
2. consult_forensic_accountant: Fundamentals (health, flags).
3. run_portfolio_optimization: Allocation strategy.

DECISION MATRIX:
- MM SELL + Forensic BUY -> HOLD/SELL (Respect the crash).
- MM BUY + Forensic SELL -> HOLD (Fundamental risk).
- Both BUY -> STRONG BUY.

OUTPUT: 3-section executive summary (Assessment, Recommendation, Rationale). No code."""


class BaseOrchestrator:
    def __init__(self):
        self.session = None
        self.stdio = None
        self.write = None
        self._stdio_ctx = None  # Internal context manager for transport

    async def connect_to_server(self, server_script_path: str):
        """Standard MCP Connection Logic"""
        server_params = StdioServerParameters(command="python", args=[server_script_path], env=None)
        
        # FIX: stdio_client is a context manager. We must manually enter it 
        # to keep the connection open across method calls.
        self._stdio_ctx = stdio_client(server_params)
        self.stdio, self.write = await self._stdio_ctx.__aenter__()

        # Initialize Session
        self.session = ClientSession(self.stdio, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        print(f"✓ Connected to MCP Server via {self.__class__.__name__}")

    async def disconnect(self):
        """Clean up both Session and Transport contexts"""
        if self.session: 
            await self.session.__aexit__(None, None, None)
        
        # FIX: Properly exit the stdio context
        if self._stdio_ctx:
            await self._stdio_ctx.__aexit__(None, None, None)


class DeepSeekOrchestrator(BaseOrchestrator):
    def __init__(self, api_key: str):
        super().__init__()
        if 'openai' not in sys.modules:
            raise ImportError("Please install openai: pip install openai")
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.conversation_history = [{"role": "system", "content": TRIAGE_MANAGER_PROMPT}]

    async def query(self, user_question: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_question})
        
        # 1. Fetch Tools from MCP & Convert to OpenAI Format
        mcp_tools = await self.session.list_tools()
        openai_tools = []
        for tool in mcp_tools.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })

        # 2. Tool Loop
        while True:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.conversation_history,
                tools=openai_tools
            )
            
            message = response.choices[0].message
            self.conversation_history.append(message) 

            if not message.tool_calls:
                return message.content

            for tool_call in message.tool_calls:
                fname = tool_call.function.name
                fargs = json.loads(tool_call.function.arguments)
                
                print(f"\n🔧 DeepSeek requesting: {fname}")
                try:
                    result = await self.session.call_tool(fname, fargs)
                    
                    # Extract text content properly from MCP result
                    content_str = "".join([c.text for c in result.content if hasattr(c, "text")])
                    print(f"   Result: {content_str[:100]}...")
                except Exception as e:
                    content_str = f"Tool execution error: {str(e)}"
                    print(f"   Error: {content_str}")

                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content_str
                })


class GeminiOrchestrator(BaseOrchestrator):
    def __init__(self, api_key: str):
        super().__init__()
        if 'google.generativeai' not in sys.modules:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        genai.configure(api_key=api_key)
        self.model = None 
        self.chat = None

    async def connect_to_server(self, server_script_path: str):
        await super().connect_to_server(server_script_path)
        
        # Convert MCP Tools to Gemini Functions
        mcp_tools = await self.session.list_tools()
        gemini_funcs = []
        for tool in mcp_tools.tools:
            gemini_funcs.append(FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema
            ))
            
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            tools=[GeminiTool(function_declarations=gemini_funcs)],
            system_instruction=TRIAGE_MANAGER_PROMPT
        )
        self.chat = self.model.start_chat(enable_automatic_function_calling=False)

    async def query(self, user_question: str) -> str:
        response = self.chat.send_message(user_question)
        
        while True:
            try:
                part = response.parts[0]
            except IndexError:
                return response.text

            if fn := part.function_call:
                fname = fn.name
                fargs = dict(fn.args)
                print(f"\n🔧 Gemini requesting: {fname}")
                
                try:
                    result = await self.session.call_tool(fname, fargs)
                    content_str = "".join([c.text for c in result.content if hasattr(c, "text")])
                    print(f"   Result: {content_str[:100]}...")
                except Exception as e:
                    content_str = f"Error: {str(e)}"
                
                # Send result back
                tool_response = Part(function_response=FunctionResponse(name=fname, response={'result': content_str}))
                response = self.chat.send_message([tool_response])
            else:
                return response.text

# --- FACTORY FUNCTION FOR RUN.PY ---
def TriageOrchestrator(api_key: str):
    """Factory that returns the correct class based on env var"""
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    
    if provider == "deepseek":
        print(f"🤖 Initializing DeepSeek Orchestrator")
        return DeepSeekOrchestrator(api_key)
    elif provider == "gemini":
        print(f"🤖 Initializing Gemini Orchestrator")
        return GeminiOrchestrator(api_key)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'deepseek' or 'gemini'.")