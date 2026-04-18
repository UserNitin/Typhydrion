from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


class GraphValidationError(Exception):
	"""Raised when a graph is structurally invalid for execution."""


@dataclass(slots=True)
class ValidationIssue:
	level: str  # error | warning
	code: str
	message: str
	node_id: str | None = None
	edge_index: int | None = None


@dataclass(slots=True)
class ValidationReport:
	valid: bool
	errors: list[ValidationIssue] | None = None
	warnings: list[ValidationIssue] | None = None
	node_ids: list[str] | None = None
	adjacency: dict[str, set[str]] | None = None
	reverse_adjacency: dict[str, set[str]] | None = None
	topological_order: list[str] | None = None

	def __post_init__(self) -> None:
		if self.errors is None:
			self.errors = []
		if self.warnings is None:
			self.warnings = []
		if self.node_ids is None:
			self.node_ids = []
		if self.adjacency is None:
			self.adjacency = {}
		if self.reverse_adjacency is None:
			self.reverse_adjacency = {}
		if self.topological_order is None:
			self.topological_order = []


def _safe_node_id(node: dict[str, Any], index: int) -> str:
	raw = node.get("node_id")
	if raw is None:
		return f"node_{index}"
	s = str(raw).strip()
	return s or f"node_{index}"


def _as_dict(value: Any) -> dict[str, Any]:
	if isinstance(value, dict):
		return cast(dict[str, Any], value)
	return {}


def normalize_graph(graph: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	"""Normalize payload to (nodes, edges) lists.

	Expected shape:
	{
	  "nodes": [{"node_id": "...", "title": "...", ...}],
	  "edges": [{"source": {"node_id": "..."}, "target": {"node_id": "..."}, ...}],
	}
	"""
	if not isinstance(graph, dict):
		return [], []
	graph_map = cast(dict[str, Any], graph)
	raw_nodes_obj = graph_map.get("nodes", [])
	raw_edges_obj = graph_map.get("edges", [])
	raw_nodes_list: list[Any] = cast(list[Any], raw_nodes_obj) if isinstance(raw_nodes_obj, list) else []
	raw_edges_list: list[Any] = cast(list[Any], raw_edges_obj) if isinstance(raw_edges_obj, list) else []
	nodes: list[dict[str, Any]] = [cast(dict[str, Any], n) for n in raw_nodes_list if isinstance(n, dict)]
	edges: list[dict[str, Any]] = [cast(dict[str, Any], e) for e in raw_edges_list if isinstance(e, dict)]
	return nodes, edges


def build_adjacency(
	nodes: list[Any],
	edges: list[Any],
) -> tuple[dict[str, set[str]], dict[str, set[str]], list[ValidationIssue], list[ValidationIssue], list[str]]:
	errors: list[ValidationIssue] = []
	warnings: list[ValidationIssue] = []

	node_ids: list[str] = []
	seen: set[str] = set()
	for i, node in enumerate(nodes):
		node_id = _safe_node_id(node, i)
		if node_id in seen:
			errors.append(ValidationIssue("error", "DUPLICATE_NODE_ID", f"Duplicate node_id '{node_id}'", node_id=node_id))
			continue
		seen.add(node_id)
		node_ids.append(node_id)

	adjacency: dict[str, set[str]] = {nid: set[str]() for nid in node_ids}
	reverse_adjacency: dict[str, set[str]] = {nid: set[str]() for nid in node_ids}
	edge_seen: set[tuple[str, str, str, str]] = set()

	for idx, edge in enumerate(edges):
		src = _as_dict(edge.get("source"))
		tgt = _as_dict(edge.get("target"))
		src_id = str(src.get("node_id", "") or "").strip()
		tgt_id = str(tgt.get("node_id", "") or "").strip()
		src_port = str(src.get("port_name", "") or "")
		tgt_port = str(tgt.get("port_name", "") or "")

		if not src_id or not tgt_id:
			errors.append(ValidationIssue("error", "EDGE_MISSING_ENDPOINT", "Edge must contain source.node_id and target.node_id", edge_index=idx))
			continue
		if src_id not in adjacency:
			errors.append(ValidationIssue("error", "EDGE_UNKNOWN_SOURCE", f"Edge source node_id '{src_id}' does not exist", edge_index=idx))
			continue
		if tgt_id not in adjacency:
			errors.append(ValidationIssue("error", "EDGE_UNKNOWN_TARGET", f"Edge target node_id '{tgt_id}' does not exist", edge_index=idx))
			continue
		if src_id == tgt_id:
			errors.append(ValidationIssue("error", "SELF_LOOP", f"Self-loop detected on node '{src_id}'", node_id=src_id, edge_index=idx))
			continue

		dedupe_key = (src_id, src_port, tgt_id, tgt_port)
		if dedupe_key in edge_seen:
			warnings.append(
				ValidationIssue(
					"warning",
					"DUPLICATE_EDGE",
					f"Duplicate edge {src_id}:{src_port} -> {tgt_id}:{tgt_port}",
					edge_index=idx,
				)
			)
			continue
		edge_seen.add(dedupe_key)

		adjacency[src_id].add(tgt_id)
		reverse_adjacency[tgt_id].add(src_id)

	return adjacency, reverse_adjacency, errors, warnings, node_ids


def topological_sort(adjacency: dict[str, set[str]]) -> list[str]:
	indegree = {node_id: 0 for node_id in adjacency}
	for src, tgts in adjacency.items():
		_ = src
		for tgt in tgts:
			indegree[tgt] = indegree.get(tgt, 0) + 1

	queue = sorted([nid for nid, deg in indegree.items() if deg == 0])
	ordered: list[str] = []

	while queue:
		node_id = queue.pop(0)
		ordered.append(node_id)
		for nxt in sorted(adjacency.get(node_id, set())):
			indegree[nxt] -= 1
			if indegree[nxt] == 0:
				queue.append(nxt)

	if len(ordered) != len(adjacency):
		unresolved = sorted([nid for nid, deg in indegree.items() if deg > 0])
		raise GraphValidationError(f"Cycle detected. Unresolved nodes: {unresolved}")
	return ordered


def validate_graph(graph: Any) -> ValidationReport:
	nodes, edges = normalize_graph(graph)
	adjacency, reverse_adjacency, errors, warnings, node_ids = build_adjacency(nodes, edges)

	for i, node in enumerate(nodes):
		nid = _safe_node_id(node, i)
		title = str(node.get("title", "") or "").strip()
		if not title:
			warnings.append(ValidationIssue("warning", "NODE_MISSING_TITLE", f"Node '{nid}' has no title", node_id=nid))

	topo: list[str] = []
	if not errors:
		try:
			topo = topological_sort(adjacency)
		except GraphValidationError as exc:
			errors.append(ValidationIssue("error", "CYCLE_DETECTED", str(exc)))

	return ValidationReport(
		valid=len(errors) == 0,
		errors=errors,
		warnings=warnings,
		node_ids=node_ids,
		adjacency=adjacency,
		reverse_adjacency=reverse_adjacency,
		topological_order=topo,
	)

