"""credence-router: transparent, cost-optimal tool routing via EU maximisation."""

from credence_router.answer import Answer
from credence_router.router import Router
from credence_router.tool import Tool

__all__ = ["Answer", "Router", "Tool", "create_react_agent"]


def create_react_agent(*args, **kwargs):
    """LangGraph-compatible create_react_agent with VOI-based routing.

    Requires: pip install credence-router[compat]
    """
    from credence_router.compat.prebuilt import create_react_agent as _cra

    return _cra(*args, **kwargs)
