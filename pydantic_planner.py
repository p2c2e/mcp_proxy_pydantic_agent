from typing import Dict, List, Tuple, Any, Annotated, TypedDict, Optional, Literal
from enum import Enum
import os
from langgraph.graph import StateGraph, END
from pydantic_ai import Agent

from pydantic import BaseModel, Field
import json

# Configure models
PLANNER_MODEL = Agent("openai:gpt-4o")

@PLANNER_MODEL.system_prompt
def planner_system_prompt():
    return "You are a planning agent. Your job is to break down a complex query into a series of steps that can be executed by specialized agents. Create a clear, ordered list of steps."

AGENT_MODEL = Agent("openai:gpt-4o")
VALIDATOR_MODEL = Agent("openai:gpt-4o")

# Define state structure
class AgentState(TypedDict):
    user_query: str
    plan: Optional[List[str]]
    current_step: Optional[int]
    execution_history: List[Dict[str, Any]]
    max_iterations: int
    current_iteration: int
    final_answer: Optional[str]
    is_solved: bool

# Define tools
@AGENT_MODEL.tool_plain
def search(query: str) -> str:
    """Search the web for information about a query"""
    # This is a mock implementation - in a real system you would integrate with a search API
    return f"Search results for: {query}. Found information about {query} including key details."

@AGENT_MODEL.tool_plain
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression"""
    try:
        print("@"*50)
        print("@"*50)
        print("@"*50)
        print("@"*50)
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {str(e)}"

@AGENT_MODEL.tool_plain
def get_current_weather(location: str) -> str:
    """Get the current weather for a location"""
    print("!"*50)
    print("!"*50)
    print("!"*50)
    return f"The weather in {location} is currently sunny with a temperature of 72°F. Tomorrow there is a 80% chance of precipitation"


# Define nodes for the graph
def initialize_state(state: AgentState) -> AgentState:
    """Initialize the state with defaults"""
    print("Initializing state with user query:", state["user_query"])
    print("Finalizing response for query:", state["user_query"])
    return {
        "user_query": state["user_query"],
        "plan": None,
        "current_step": None,
        "execution_history": [],
        "max_iterations": state.get("max_iterations", 3),
        "current_iteration": 0,
        "final_answer": None,
        "is_solved": False
    }

def create_plan(state: AgentState) -> AgentState:
    """Create a plan of steps to solve the query"""

    plan_request = f"Create a plan to answer this query: {state['user_query']}"
    
    print("Creating plan for query:", state["user_query"])
    response = PLANNER_MODEL.run_sync(plan_request)
    print("Plan created:", response.data)
    
    # Parse the response to extract the plan steps
    # This assumes the model returns a numbered list
    raw_plan = response.data
    
    # Simple parsing approach - in production you might want more robust extraction
    plan_steps = []
    for line in raw_plan.split('\n'):
        line = line.strip()
        # Check if line starts with a number (like "1." or "1-")
        if line and (line[0].isdigit() or (len(line) > 2 and line[0].isdigit() and line[1] in ['.', '-', ')'])):
            # Strip the number prefix
            step = line.split(".", 1)[-1].strip() if "." in line[:3] else line.split(")", 1)[-1].strip() if ")" in line[:3] else line.split("-", 1)[-1].strip()
            plan_steps.append(step)
    
    # If parsing failed, use the whole response as a single step
    if not plan_steps:
        plan_steps = [raw_plan]
    
    return {
        **state,
        "plan": plan_steps,
        "current_step": 0
    }

def execute_step(state: AgentState) -> AgentState:
    """Execute the current step in the plan using appropriate tools"""
    current_step = state["current_step"]
    plan_steps = state["plan"]
    
    if current_step >= len(plan_steps):
        return {**state, "current_step": len(plan_steps)}
    
    current_task = plan_steps[current_step]
    print(f"Executing step {current_step + 1}/{len(plan_steps)}: {current_task}")
    
    # Construct the agent prompt with history and current task
    history_context = ""
    if state["execution_history"]:
        history_context = "Previous steps execution:\n" + "\n".join([
            f"Step: {item['step']}\nAction: {item['action']}\nResult: {item['result']}"
            for item in state["execution_history"]
        ])
    
    request = f"""You are a helpful assistant with access to tools. 
                You need to execute this task: {current_task}
                You should use tools when appropriate and respond with your findings.
                {history_context}

                Always structure your thinking step by step, and use tools when appropriate.            
                Execute this task: {current_task}. This is part of answering the original query: '{state['user_query']}'"""

    # First, get the agent's plan
    agent_response = AGENT_MODEL.run_sync(request)
    print("Agent response:", agent_response.data)
    
    agent_thinking = agent_response.data
    
    tool_calls = []
    tool_results = []
    # Create a summary of the execution
    execution_summary = {
        "step": current_task,
        "action": f"Used tools: {', '.join([call['name'] for call in tool_calls])}" if tool_calls else "Analysis without tools",
        "result": "; ".join([f"{r['tool']}: {r['result']}" for r in tool_results]) if tool_results else agent_thinking
    }
    
    # Update the state
    return {
        **state,
        "current_step": current_step + 1,
        "execution_history": state["execution_history"] + [execution_summary]
    }

def validate_solution(state: AgentState) -> AgentState:
    """Validate if the query has been solved or if we need another iteration"""
    # Prepare context from execution history
    execution_context = "\n".join([
        f"Step {i+1}: {item['step']}\nAction: {item['action']}\nResult: {item['result']}"
        for i, item in enumerate(state["execution_history"])
    ])
    

    request = f"""You are a validation agent that determines if a query has been fully answered.
        Review the execution history and determine if the original query has been adequately solved.
        If the query is solved, provide the final answer.
        If not, explain what information is still missing.
        Original query: {state['user_query']}

        Execution history:
        {execution_context}

        Has the query been adequately answered? Answer YES or NO, followed by your reasoning:
        1. If YES, provide a concise final answer that directly addresses the original query.
        2. If NO, explain what information is still missing or what additional steps should be taken."""
    
    print("Validating solution for query:", state["user_query"])
    response = VALIDATOR_MODEL.run_sync(request)
    print("Validation response:", response.data)
    response_text = response.data
    
    # Check if the validator thinks we've solved the query
    is_solved = response_text.strip().upper().startswith("YES")
    
    # Extract the final answer if solved, or set up for another iteration
    if is_solved:
        # Extract the final answer - everything after the "YES" part
        final_answer = response_text.split("YES", 1)[1].strip()
        if not final_answer:
            # If split didn't work, use the whole response
            final_answer = response_text
    else:
        final_answer = None
    
    # Update the state
    new_state = {
        **state,
        "is_solved": is_solved,
        "final_answer": final_answer,
        "current_iteration": state["current_iteration"] + 1
    }
    
    # Reset for next iteration if needed
    if not is_solved and new_state["current_iteration"] < new_state["max_iterations"]:
        new_state["current_step"] = 0
    
    return new_state

def should_continue(state: AgentState) -> Literal["continue_iteration", "finalize"]:
    """Determine if we should continue iterating or finalize"""
    if state["is_solved"]:
        return "finalize"
    
    if state["current_iteration"] >= state["max_iterations"]:
        return "finalize"
    
    return "continue_iteration"

def finalize_response(state: AgentState) -> AgentState:
    """Create the final response to return to the user"""
    if state["is_solved"]:
        final_result = state["final_answer"]
    else:
        # We've reached max iterations without solving
        execution_context = "\n".join([
            f"Step {i+1}: {item['step']}\nAction: {item['action']}\nResult: {item['result']}"
            for i, item in enumerate(state["execution_history"])
        ])
        
        request = f"""You are a helpful assistant. Based on the execution history, 
            provide the best possible answer to the original query, even if incomplete.
            Acknowledge what information might still be missing.
            Original query: {state['user_query']}

            Execution history (after {state['current_iteration']} iterations):
            {execution_context}

            Please provide the best possible answer to the original query based on the information gathered."""
        
        response = AGENT_MODEL.run_sync(request)
        final_result = response.data
    
    return {
        **state,
        "final_answer": final_result
    }

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("initialize", initialize_state)
workflow.add_node("create_plan", create_plan)
workflow.add_node("execute_step", execute_step)
workflow.add_node("validate_solution", validate_solution)
workflow.add_node("finalize", finalize_response)

# Add edges
workflow.add_edge("initialize", "create_plan")
workflow.add_edge("create_plan", "execute_step")

# workflow.add_edge("execute_step", END)
workflow.add_conditional_edges(
    "execute_step",
    lambda state: "execute_step" if state["current_step"] < len(state["plan"]) else "validate_solution",
    {
        "execute_step": "execute_step",
        "validate_solution": "validate_solution"
    }
)
workflow.add_conditional_edges(
    "validate_solution",
    should_continue,
    {
        "continue_iteration": "create_plan",
        "finalize": "finalize"
    }
)
workflow.add_edge("finalize", END)

# Set the entry point
workflow.set_entry_point("initialize")

# Compile the graph
app = workflow.compile()

# Function to run the workflow
def run_agent(query: str, max_iterations: int = 3) -> str:
    """Run the agent workflow on a user query"""
    initial_state = {
        "user_query": query,
        "max_iterations": max_iterations
    }
    
    # Execute the workflow
    result = app.invoke(initial_state)
    
    # Return the final answer
    return result["final_answer"]

# Example usage
if __name__ == "__main__":
    user_query = "What will the weather be like in New York tomorrow, and should I bring an umbrella? Also, calculate 25% of $120."
    # user_query = "What is 15 times 15? Now multiply that with the latitude of New Delhi"
    # user_query = "What is 15 times 15? What is the weather in New Delhi?"
    response = run_agent(user_query, max_iterations=3)
    print(f"Query: {user_query}")
    print(f"Response: {response}")
