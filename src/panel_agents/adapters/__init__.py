"""
Adapters translating agent frameworks into panel_agents events.

Each adapter lives in its own module so that importing panel_agents
never requires an agent framework, e.g.
``from panel_agents.adapters.pydantic_ai import AgentRunner``.
"""
