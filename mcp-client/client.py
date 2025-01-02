import asyncio
import json
from typing import Optional, Dict, List, Union, Any
from contextlib import AsyncExitStack
from colorama import init, Fore, Style

from mcp import ClientSession, StdioServerParameters
init(autoreset=True)  # Initialize colorama with autoreset=True
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import Tool, ToolDefinition
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize sessions and agents dictionaries
        self.sessions: Dict[str, ClientSession] = {}  # Dictionary to store {server_name: session}
        self.agents: Dict[str, Agent] = {}  # Dictionary to store {server_name: agent}
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.available_tools = []  # List to store all available tools across servers
        self.dynamic_tools: List[Tool] = []  # List to store dynamic pydantic tools

    async def connect_to_server(self):
        """Connect to an MCP server using config.json settings"""
        print("\nLoading config.json...")
        with open('config.json') as f:
            config = json.load(f)
        
        print("\nAvailable servers in config:", list(config['mcpServers'].keys()))
        print("\nFull config content:", json.dumps(config, indent=2))
        
        # Connect to all servers in config
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
            server_agent: Agent = Agent(
                'openai:gpt-4',
                system_prompt=(
                    f"You are an AI assistant that helps interact with the {server_name} server. "
                    "You will use the available tools to process requests and provide responses."
                )
            )
            self.agents[server_name] = server_agent
            
            # List available tools for this server
            response = await session.list_tools()
            server_tools = [{
                "name": f"{server_name}__{tool.name}",
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in response.tools]
            
            # Add server's tools to overall available tools
            self.available_tools.extend(server_tools)

            # Create corresponding dynamic pydantic tools
            for tool in response.tools:
                async def prepare_tool(
                    ctx: RunContext[str], 
                    tool_def: ToolDefinition,
                    tool_name: str = tool.name,
                    server: str = server_name
                ) -> Union[ToolDefinition, None]:
                    # Customize tool definition based on server context
                    tool_def.name = f"{server}__{tool_name}"
                    tool_def.description = f"Tool from {server} server: {tool.description}"
                    return tool_def

                # Create a function that matches the tool's schema and uses server_agent
                async def tool_func(ctx: RunContext[Any], str_arg) -> str:
                    agent_response = await server_agent.run_sync(str_arg)
                    print(f"\nServer agent response: {agent_response}")
                    return f"Tool {tool.name} called with {str_arg}. Agent response: {agent_response}"

                dynamic_tool = Tool(
                    tool_func,
                    prepare=prepare_tool,
                    name=f"{server_name}__{tool.name}",
                    description=tool.description
                )
                self.dynamic_tools.append(dynamic_tool)
                print(f"\nAdded dynamic tool: {dynamic_tool.name}")
                print(f"Description: {dynamic_tool.description}")
                print(f"Function: {dynamic_tool.function}")
                print(f"Prepare function: {dynamic_tool.prepare}")
            
            print(f"\nConnected to server {server_name} with tools:", 
                  [tool["name"] for tool in server_tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Initial Claude API call with all available tools
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=self.available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                # Parse server name and tool name from the full tool name
                full_tool_name = content.name
                server_name, tool_name = full_tool_name.split('__', 1)
                tool_args = content.input
                
                # Get the appropriate session and execute tool call
                if server_name not in self.sessions:
                    raise ValueError(f"Unknown server: {server_name}")
                    
                result = await self.sessions[server_name].call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                      "role": "assistant",
                      "content": content.text
                    })
                messages.append({
                    "role": "user", 
                    "content": result.content
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                )

                final_text.append(f"{Style.BRIGHT}{Fore.CYAN}{response.content[0].text}")

        return "\n".join(final_text)

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
    asyncio.run(main())
