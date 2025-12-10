# agent_graph/__init__.py

"""
AgentGraph package initializer.

This package provides:
- create_graph(): the compiled LangGraph pipeline
- AgentState: shared state wrapper for all nodes
- Router, Planner, Retriever, Critic, Answerer
- MCPClientWrapper for tool calling

Legacy function-based nodes (router_node, planner_node, etc.)
and build_graph/run_pipeline are removed and should NOT be used.
"""

from .graph import create_graph
from .state import AgentState
from .router import Router
from .planner import Planner
from .retriever import Retriever
from .critic import Critic
from .answerer import Answerer
from .mcp_client import MCPClientWrapper

__all__ = [
    "create_graph",
    "AgentState",
    "Router",
    "Planner",
    "Retriever",
    "Critic",
    "Answerer",
    "MCPClientWrapper",
]
