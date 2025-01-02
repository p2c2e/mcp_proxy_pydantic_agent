
## Sample to show to integrate MCP (Model Context Protocol) servers with Pydantic.AI


Parts of this example uses content from : https://github.com/modelcontextprotocol/quickstart-resources.git - Esp. the weather 'server' code

Code uses two different LLMs just for demonstration. The Proxy Agent uses gpt-4o and the tool uses sonnet. 
So, export OPENAI_API_KEY as well as ANTHROPIC_API_KEY - OR - modify the code to suit your models

The pyproject.toml assumes you are using 'uv' package manager

### Steps to run
1. Clone this repo
1. uv sync
3. cd mc-client
2. uv run client.py

Now, try interacting with some questions like:

> What is the time in NY when it is 7:30pm in Bangalore?

> What is the Weather currently in Chicago?

(and quit when done)

