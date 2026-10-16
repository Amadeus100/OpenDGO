from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from opendgo_q1.topologies import normalize_edges

from .util import stable_int


@dataclass(frozen=True)
class FaultPlan:
    pressure_percent: int
    initial_component_count: int
    requested_failure_count: int
    failed_agents: tuple[int, ...]
    failed_channels: tuple[tuple[int, int], ...]
    realized_failure_count: int
    realized_pressure_percent: float
    sampling_key: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["failed_agents"] = list(self.failed_agents)
        value["failed_channels"] = [list(channel) for channel in self.failed_channels]
        return value


def build_fault_plan(
    edges: Iterable[Iterable[int]], pressure_percent: int, *,
    task_id: str, task_structure: str, organization: str, seed: int,
) -> FaultPlan:
    normalized = normalize_edges(edges)
    channels = sorted(
        [(left, right) for left, right in normalized] + [(right, left) for left, right in normalized]
    )
    initial_count = 6 + len(channels)
    requested = int(initial_count * pressure_percent / 100 + 0.5)
    key = f"{task_id}|{task_structure}|{organization}|{seed}|{pressure_percent}"
    failed_agents: set[int] = set()
    failed_channels: set[tuple[int, int]] = set()
    event_index = 0
    while len(failed_agents) + len(failed_channels) < requested:
        available_agents = [node for node in range(6) if node not in failed_agents]
        available_channels = [
            channel for channel in channels
            if channel not in failed_channels
            and channel[0] not in failed_agents
            and channel[1] not in failed_agents
        ]
        if not available_agents and not available_channels:
            break
        choose_agent = stable_int(key, "type", event_index) % 2 == 0
        if choose_agent and not available_agents:
            choose_agent = False
        elif not choose_agent and not available_channels:
            choose_agent = True
        if choose_agent:
            ranked = sorted(available_agents, key=lambda node: stable_int(key, "agent", event_index, node))
            failed_agents.add(ranked[0])
        else:
            ranked_channels = sorted(
                available_channels,
                key=lambda channel: stable_int(key, "channel", event_index, *channel),
            )
            failed_channels.add(ranked_channels[0])
        event_index += 1
    realized = len(failed_agents) + len(failed_channels)
    return FaultPlan(
        pressure_percent=pressure_percent,
        initial_component_count=initial_count,
        requested_failure_count=requested,
        failed_agents=tuple(sorted(failed_agents)),
        failed_channels=tuple(sorted(failed_channels)),
        realized_failure_count=realized,
        realized_pressure_percent=100.0 * realized / initial_count,
        sampling_key=key,
    )
