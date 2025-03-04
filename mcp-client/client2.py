import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Union

import nest_asyncio
from colorama import Fore, Style, init
from mcp import ClientSession, StdioServerParameters

init(autoreset=True)  # Initialize colorama with autoreset=True

from dotenv import load_dotenv
from mcp.client.stdio import stdio_client
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import Tool, ToolDefinition

load_dotenv()  


# The following function is required to enable capture of the local variables within the 
# server config parsing loop 
def create_tool_fn(server_agent: Agent, tool_name: str, sessions: Dict[str, ClientSession], server_name: str ):
    
    async def tool_func(ctx: RunContext[Any], **kwargs) -> str:
        print("!"*100)
        print("#"*100)
        print(kwargs)
        print(f"Tool name: {tool_name}")
        print("!"*100)

        # agent_response = server_agent.run_sync(str_arg)
        agent_response = await sessions[server_name].call_tool(tool_name, kwargs)
        print(f"\nServer agent response: {agent_response}")
        return f"Tool {tool_name} called with {kwargs}. Agent response: {agent_response}"
    
    return tool_func

class MCPClient:
    def __init__(self):
        # Initialize sessions and agents dictionaries
        self.sessions: Dict[str, ClientSession] = {}  # Dictionary to store {server_name: session}
        self.exit_stack = AsyncExitStack()
        self.available_tools = []  # List to store all available tools across servers
        self.dynamic_tools: List[Tool] = []  # List to store dynamic pydantic tools
        
        self.server_names = []
        self.super_agent = None
        

    async def connect_to_server(self):
        """Connect to an MCP server using config.json settings"""
        print("\nLoading config.json...")
        with open('config.json') as f:
            config = json.load(f)
        
        print("\nAvailable servers in config:", list(config['mcpServers'].keys()))
        print("\nFull config content:", json.dumps(config, indent=2))
        
        # Connect to all servers in config
        print("Adding Server names")
        self.server_names.extend(list(config["mcpServers"].keys()))
        print("Looping over servers")
        for server_name, server_config in config['mcpServers'].items():
            print(f"\nAttempting to load {server_name} server config...")
            print("Server config found:", json.dumps(server_config, indent=2))
            
            server_params = StdioServerParameters(
                command=server_config['command'],
                args=server_config['args'],
                env=None
            )
            print("\nCreated server parameters:", server_params)
            
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            
            await session.initialize()
            
            # Store session with server name as key
            self.sessions[server_name] = session
            
            # Create and store an Agent for this server
            # This agent wrapper ensure the schema translation happens when the tool is called...
            server_agent: Agent = Agent(
                'openai:gpt-4',
                system_prompt=(
                    f"You are an AI assistant that helps interact with the {server_name} server. "
                    "You will use the available tools to process requests and provide responses."
                )
            )
            
            # List available tools for this server
            response = await session.list_tools()
            
            server_tools = [{
                "name": f"{server_name}__{tool.name}",
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in response.tools]
            
            print(f"\nAvailable tools for {server_tools}")

            # Add server's tools to overall available tools
            self.available_tools.extend(server_tools)

            server_agent: Agent = Agent(
                'openai:gpt-4',
                system_prompt=(
                    f"You are an AI assistant that helps interact with the {server_name} server. "
                    "You will use the available tools to process requests and provide responses."
                )
            )

            # Create corresponding dynamic pydantic tools
            for tool in response.tools:

                # Create a function that matches the tool's schema and uses server_agent
                tool_func = create_tool_fn(server_agent=server_agent, tool_name=tool.name, sessions=self.sessions, server_name=server_name)

                dynamic_tool = Tool(
                    tool_func,
                    name=f"{server_name}__{tool.name}",
                    description=tool.description,
                    max_retries=3,
                )
                dynamic_tool._parameters_json_schema = tool.inputSchema 
                self.dynamic_tools.append(dynamic_tool)
                print(f"\nAdded dynamic tool: {dynamic_tool.name}")
                print(f"Description: {dynamic_tool.description}")
                print(f"Function: {dynamic_tool.function}")
                print(f"Prepare function: {dynamic_tool.prepare}")
            
            self.super_agent: Agent = Agent(
                'openai:gpt-4o',
                tools=self.dynamic_tools,
                retries=3,
            )
            @self.super_agent.system_prompt
            def super_agent_system_prompt():
                print("Super agent system prompt called")
                print("Server names", self.server_names)
                return (
                    f"You are an AI assistant that helps interact with the {', '.join(self.server_names)} servers. "
                    "You will use the available tools to process requests and provide responses."
                )
            # print(self.super_agent._function_tools)
            print(f"\nConnected to server {server_name} with tools:", 
                  [tool["name"] for tool in server_tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""

        print("-"*100)
        # print(self.super_agent)
        # print("-"*100)
        # exit(1)
        result = self.super_agent.run_sync(query)
        for message in result._all_messages:
            print(f"{Fore.GREEN}{message}")
        print("="*100)

        return result.data

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print(f"{Fore.WHITE}\nMCP Client Started!")
        print(f"{Fore.WHITE}Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input(f"\n{Fore.RED}Query: {Fore.LIGHTGREEN_EX}").strip()
                
                if query.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\nGoodbye!")
                    break
                    
                response = await self.process_query(query)
                print(f"\n{Fore.YELLOW}{response}")
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"\n{Fore.RED}Error: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    nest_asyncio.apply()
    asyncio.run(main())
