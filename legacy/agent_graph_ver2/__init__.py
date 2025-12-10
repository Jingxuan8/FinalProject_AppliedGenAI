# agent_graph/__init__.py

"""
agent_graph package initializer.

Exports:
- build_graph
- run_pipeline
- MCPClient
- all agent nodes
"""

from agent_graph.graph import build_graph, run_pipeline
from agent_graph.mcp_client import MCPClientWrapper


from agent_graph.router import router_node
from agent_graph.planner import planner_node
from agent_graph.retriever import retriever_node
from agent_graph.critic import critic_node
from agent_graph.answerer import answerer_node

__all__ = [
    "build_graph",
    "run_pipeline",
    "MCPClientWrapper",
    "router_node",
    "planner_node",
    "retriever_node",
    "critic_node",
    "answerer_node",
]
