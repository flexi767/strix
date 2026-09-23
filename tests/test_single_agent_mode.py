"""Single-agent policy rejects fresh and resumed children before any side effects."""

from unittest.mock import AsyncMock, Mock

import pytest
from agents import RunConfig

from strix.agents.factory import build_strix_agent
from strix.core.agents import AgentCoordinator
from strix.core.execution import respawn_subagents, spawn_child_agent


@pytest.mark.asyncio
async def test_single_agent_blocks_spawn_before_factory_or_registration(monkeypatch, tmp_path):
    monkeypatch.setenv("STRIX_SINGLE_AGENT", "1")
    coordinator = AgentCoordinator()
    await coordinator.register("root", "root", parent_id=None)
    factory = Mock(side_effect=AssertionError("Must not construct a child"))
    result = await spawn_child_agent(
        coordinator=coordinator,
        factory=factory,
        agents_db_path=tmp_path / "agents.db",
        sessions_to_close=[],
        run_config=RunConfig(),
        max_turns=1,
        interactive=False,
        parent_ctx={"agent_id": "root"},
        name="child",
        task="review",
        skills=[],
        parent_history=[],
    )
    assert result["success"] is False
    assert "Single-agent" in result["error"]
    factory.assert_not_called()
    assert list(coordinator.statuses) == ["root"]


@pytest.mark.asyncio
async def test_single_agent_rejects_resumed_children(monkeypatch, tmp_path):
    monkeypatch.setenv("STRIX_SINGLE_AGENT", "1")
    coordinator = AgentCoordinator()
    await coordinator.register("root", "root", parent_id=None)
    await coordinator.register("child", "child", parent_id="root")
    factory = Mock()
    with pytest.raises(RuntimeError, match="cannot resume"):
        await respawn_subagents(
            coordinator=coordinator,
            factory=factory,
            agents_db_path=tmp_path / "agents.db",
            sessions_to_close=[],
            run_config=RunConfig(),
            max_turns=1,
            interactive=False,
            parent_ctx={"agent_id": "root"},
            root_id="root",
        )
    factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [None, "0"])
async def test_default_spawning_is_unchanged(monkeypatch, tmp_path, policy):
    if policy is None:
        monkeypatch.delenv("STRIX_SINGLE_AGENT", raising=False)
    else:
        monkeypatch.setenv("STRIX_SINGLE_AGENT", policy)
    coordinator = AgentCoordinator()
    await coordinator.register("root", "root", parent_id=None)
    starter = AsyncMock()
    monkeypatch.setattr("strix.core.execution._start_child_runner", starter)
    result = await spawn_child_agent(
        coordinator=coordinator,
        factory=Mock(),
        agents_db_path=tmp_path / "agents.db",
        sessions_to_close=[],
        run_config=RunConfig(),
        max_turns=1,
        interactive=False,
        parent_ctx={"agent_id": "root"},
        name="child",
        task="review",
        skills=[],
        parent_history=[],
    )
    assert result["success"] is True
    starter.assert_awaited_once()
    assert len(coordinator.statuses) == 2


def test_single_agent_root_can_test_without_delegation(monkeypatch):
    monkeypatch.setenv("STRIX_SINGLE_AGENT", "1")
    agent = build_strix_agent(is_root=True, scan_mode="standard", is_whitebox=True)
    assert "create_agent" not in [tool.name for tool in agent.tools]
    assert "finish_scan" in [tool.name for tool in agent.tools]
    assert "<single_agent_policy>" in agent.instructions
    assert "HTTP/browser requests" in agent.instructions
    assert "<root_agent>" not in agent.instructions
    assert agent.capabilities


def test_default_root_still_supports_delegation(monkeypatch):
    monkeypatch.delenv("STRIX_SINGLE_AGENT", raising=False)
    agent = build_strix_agent(is_root=True, scan_mode="standard")
    assert "create_agent" in [tool.name for tool in agent.tools]
    assert "<single_agent_policy>" not in agent.instructions
