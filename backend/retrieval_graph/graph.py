"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing & routing user queries, generating research plans to answer user questions,
conducting research, and formulating responses.
"""

from typing import Any, Literal, TypedDict, cast
import asyncio

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from backend.retrieval_graph.configuration import AgentConfiguration
from backend.retrieval_graph.researcher_graph.graph import graph as researcher_graph
from backend.retrieval_graph.state import AgentState, InputState, Router
from backend.utils import format_docs, load_chat_model, build_multimodal_messages



async def analyze_query_risk_and_route(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    # allow skipping the router for testing
    if state.router and state.router["logic"]:
        return {"router": state.router}

    configuration = AgentConfiguration.from_runnable_config(config)
    structured_output_kwargs = (
        {"method": "function_calling"} if "openai" in configuration.query_model else {}
    )
    model = load_chat_model(configuration.query_model).with_structured_output(
        Router, **structured_output_kwargs
    )
    messages = [
        {"role": "system", "content": configuration.guardrail_system_prompt}
    ] + state.messages
    response = cast(Router, await model.ainvoke(messages))
    return {"router": response}


def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "guardrail_response"]:

    _type = state.router["type"]
    if _type == "restaurant":
        return "create_research_plan"
    elif _type == "sensitive":
        return "guardrail_response"
    else:
        raise ValueError(f"Unknown router type {_type}")


async def guardrail_response(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.guardrail_response_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str]]:

    class Plan(TypedDict):
        """Generate research plan."""

        steps: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    structured_output_kwargs = (
        {"method": "function_calling"} if "openai" in configuration.query_model else {}
    )
    model = load_chat_model(configuration.query_model).with_structured_output(
        Plan, **structured_output_kwargs
    )
    messages = [
        {"role": "system", "content": configuration.research_plan_system_prompt}
    ] + state.messages
    response = cast(
        Plan, await model.ainvoke(messages, {"tags": ["langsmith:nostream"]})
    )
    return {
        "steps": response["steps"],
        "documents": "delete",
        "query": state.messages[-1].content,
    }


async def conduct_research(state: AgentState) -> dict[str, Any]:
    result = await researcher_graph.ainvoke({"question": state.steps[0]})
    return {"documents": result["documents"], "steps": state.steps[1:]}


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    # TODO: add a re-ranker here
    top_k = 20
    context = format_docs(state.documents[:top_k])
    prompt = configuration.response_system_prompt.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state.messages

    # If using OpenAI/gpt-4o, build multimodal message payload
    if configuration.response_model.startswith("openai/gpt-4o"):
        # Build messages preserving system and history; move context (with images) into a multimodal user message
        openai_messages = await asyncio.to_thread(build_multimodal_messages, prompt, state.messages)
        response = await model.ainvoke(openai_messages)
    else:
        response = await model.ainvoke(messages)
    return {"messages": [response], "answer": getattr(response, "content", str(response))}


# Define the graph


builder = StateGraph(AgentState, input_schema=InputState, config_schema=AgentConfiguration)
builder.add_node(analyze_query_risk_and_route)
builder.add_node(guardrail_response)
builder.add_node(create_research_plan)
builder.add_node(conduct_research)
builder.add_node(respond)

builder.add_edge(START, "analyze_query_risk_and_route")
builder.add_conditional_edges("analyze_query_risk_and_route", route_query)
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
builder.add_edge("respond", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"
