from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.graph_validator import validate_graph, GraphValidationError


@dataclass(slots=True)
class SchedulePlan:
	node_ids: list[str]
	execution_order: list[str]
	levels: list[list[str]]
	adjacency: dict[str, set[str]]
	reverse_adjacency: dict[str, set[str]]
	warnings: list[str] = field(default_factory=list)


class Scheduler:
	"""Builds a stable DAG execution schedule from graph payload."""

	def build_plan(self, graph: dict[str, Any]) -> SchedulePlan:
		report = validate_graph(graph)
		if not report.valid:
			msgs = [f"{e.code}: {e.message}" for e in report.errors]
			raise GraphValidationError("; ".join(msgs))

		levels = self._build_levels(report.adjacency, report.reverse_adjacency)
		warnings = [f"{w.code}: {w.message}" for w in report.warnings]
		return SchedulePlan(
			node_ids=list(report.node_ids),
			execution_order=list(report.topological_order),
			levels=levels,
			adjacency=report.adjacency,
			reverse_adjacency=report.reverse_adjacency,
			warnings=warnings,
		)

	def _build_levels(
		self,
		adjacency: dict[str, set[str]],
		reverse_adjacency: dict[str, set[str]],
	) -> list[list[str]]:
		indegree = {nid: len(reverse_adjacency.get(nid, set())) for nid in adjacency}
		frontier = sorted([nid for nid, d in indegree.items() if d == 0])
		levels: list[list[str]] = []
		visited_count = 0

		while frontier:
			levels.append(list(frontier))
			next_frontier: list[str] = []
			for nid in frontier:
				visited_count += 1
				for nxt in sorted(adjacency.get(nid, set())):
					indegree[nxt] -= 1
					if indegree[nxt] == 0:
						next_frontier.append(nxt)
			frontier = sorted(next_frontier)

		if visited_count != len(adjacency):
			unresolved = sorted([nid for nid, d in indegree.items() if d > 0])
			raise GraphValidationError(f"Cycle detected while building schedule levels: {unresolved}")

		return levels

