import logging
from langgraph.graph import StateGraph, END
from agents.planner  import run_planner
from agents.reviewer import run_reviewer

logger = logging.getLogger(__name__)

def build_workflow():
    """Build and compile the planner → reviewer LangGraph workflow."""
    graph = StateGraph(dict)
    graph.add_node("planner",  run_planner)
    graph.add_node("reviewer", run_reviewer)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "reviewer")
    graph.add_edge("reviewer", END)
    return graph.compile()