import requests
import random
from mcp.server import Server
import mcp.types as types

server = Server("jokes")

def get_dad_joke():
    url = "https://icanhazdadjoke.com/"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("joke", "No joke found.")
    return "Failed to fetch a dad joke."

def get_general_joke():
    url = "https://official-joke-api.appspot.com/jokes/random"
    response = requests.get(url)
    if response.status_code == 200:
        joke_data = response.json()
        return f"{joke_data.get('setup', 'No setup')} - {joke_data.get('punchline', 'No punchline')}"
    return "Failed to fetch a general joke."

@server.list_tools()
async def list_joke_tools() -> list[types.Tool]:
    """List available joke tools."""
    return [
        types.Tool(
            name="dad-jokes",
            description="Get a random dad joke",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="jokes",
            description="Get a random general joke",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

@server.call_tool()
async def call_joke_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    """Call the appropriate joke function based on the tool name."""
    if name == "dad-jokes":
        joke = get_dad_joke()
        return [types.TextContent(type="text", text=joke)]
    elif name == "jokes":
        joke = get_general_joke()
        return [types.TextContent(type="text", text=joke)]
    else:
        raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    from mcp.server import NotificationOptions, InitializationOptions

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="jokes",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    asyncio.run(main())
