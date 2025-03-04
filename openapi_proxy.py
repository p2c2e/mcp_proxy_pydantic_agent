from pydantic_ai import Agent

def create_openapi_agent(openapi_json):
    """
    Create an Agent instance from an OpenAPI JSON object.
    This instance will expose all the methods implemented on the openapi_json
    object wrapped in agent.tool_plan decorators.
    """
    agent = Agent()

    for path, methods in openapi_json.get('paths', {}).items():
        for method, details in methods.items():
            # Determine the prefix based on the HTTP method
            prefix = {
                'get': 'get_',
                'put': 'update_',
                'delete': 'delete_',
                'post': 'post_'
            }.get(method.lower(), '')

            # Create a function name with the appropriate prefix
            function_name = f"{prefix}{details.get('operationId', 'unknown_operation')}"

            # Define a tool plan function
            @agent.tool_plan(name=function_name)
            async def tool_function(*args, **kwargs):
                # Placeholder for actual implementation
                return f"Executing {function_name} with args: {args}, kwargs: {kwargs}"

            # Assign the function to the agent
            setattr(agent, function_name, tool_function)

    return agent
