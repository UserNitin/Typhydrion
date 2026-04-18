from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time

from engine.graph_validator import normalize_graph
from engine.scheduler import Scheduler, SchedulePlan
from engine.node_runner import NodeRunner, NodeExecutionResult


@dataclass(slots=True)
class PipelineExecutionSummary:
	success: bool
	execution_order: list[str] = field(default_factory=list)
	plan_levels: list[list[str]] = field(default_factory=list)
	node_results: dict[str, NodeExecutionResult] = field(default_factory=dict)
	terminal_outputs: dict[str, Any] = field(default_factory=dict)
	warnings: list[str] = field(default_factory=list)
	errors: list[str] = field(default_factory=list)
	started_at: float = 0.0
	finished_at: float = 0.0

	@property
	def execution_time(self) -> float:
		if self.finished_at <= self.started_at:
			return 0.0
		return self.finished_at - self.started_at


class PipelineExecutor:
	"""Executes an entire node graph in scheduled topological order."""

	def __init__(self, scheduler: Scheduler | None = None, runner: NodeRunner | None = None) -> None:
		self._scheduler = scheduler or Scheduler()
		self._runner = runner or NodeRunner()

	def execute(
		self,
		graph: dict[str, Any],
		initial_inputs_by_node: dict[str, dict[str, Any]] | None = None,
		fail_fast: bool = True,
	) -> PipelineExecutionSummary:
		started_at = time.perf_counter()
		initial_inputs_by_node = initial_inputs_by_node or {}

		try:
			plan = self._scheduler.build_plan(graph)
		except Exception as exc:
			finished = time.perf_counter()
			return PipelineExecutionSummary(
				success=False,
				errors=[str(exc)],
				started_at=started_at,
				finished_at=finished,
			)

		nodes, edges = normalize_graph(graph)
		node_map = self._build_node_map(nodes)

		node_results: dict[str, NodeExecutionResult] = {}
		warnings: list[str] = list(plan.warnings)
		errors: list[str] = []

		for node_id in plan.execution_order:
			node_payload = node_map.get(node_id)
			if node_payload is None:
				errors.append(f"Node '{node_id}' is present in schedule but missing in payload")
				if fail_fast:
					break
				continue

			inputs = self._build_node_inputs(
				node_id=node_id,
				edges=edges,
				node_results=node_results,
				initial_inputs=initial_inputs_by_node.get(node_id, {}),
			)

			result = self._runner.run_node(node_payload, inputs)
			node_results[node_id] = result

			if not result.success:
				errors.append(f"Node '{node_id}' ({result.node_title}) failed: {result.error_message}")
				if fail_fast:
					break

		terminal_outputs = self._collect_terminal_outputs(plan, node_results)
		finished = time.perf_counter()
		return PipelineExecutionSummary(
			success=len(errors) == 0,
			execution_order=list(plan.execution_order),
			plan_levels=list(plan.levels),
			node_results=node_results,
			terminal_outputs=terminal_outputs,
			warnings=warnings,
			errors=errors,
			started_at=started_at,
			finished_at=finished,
		)

	def _build_node_map(self, nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
		out: dict[str, dict[str, Any]] = {}
		for i, node in enumerate(nodes):
			if not isinstance(node, dict):
				continue
			node_id = str(node.get("node_id") or f"node_{i}")
			out[node_id] = node
		return out

	def _build_node_inputs(
		self,
		node_id: str,
		edges: list[dict[str, Any]],
		node_results: dict[str, NodeExecutionResult],
		initial_inputs: dict[str, Any],
	) -> dict[str, Any]:
		import pandas as pd

		inputs: dict[str, Any] = dict(initial_inputs or {})
		fallback_data = inputs.get("Data")

		for edge in edges:
			if not isinstance(edge, dict):
				continue
			src = edge.get("source") if isinstance(edge.get("source"), dict) else {}
			tgt = edge.get("target") if isinstance(edge.get("target"), dict) else {}
			src_id = str(src.get("node_id", "") or "")
			tgt_id = str(tgt.get("node_id", "") or "")
			if tgt_id != node_id or not src_id:
				continue

			src_result = node_results.get(src_id)
			if src_result is None:
				continue

			src_port = str(src.get("port_name", "") or "")
			tgt_port = str(tgt.get("port_name", "") or "")
			payload = None
			if src_port and src_port in src_result.outputs:
				payload = src_result.outputs.get(src_port)
			if payload is None:
				payload = src_result.primary_output
			if payload is None:
				continue

			cols = edge.get("columns", [])
			if isinstance(cols, list) and cols and "*" not in cols and isinstance(payload, pd.DataFrame):
				keep = [c for c in cols if c in payload.columns]
				if keep:
					payload = payload[keep].copy()

			key = tgt_port or "Data"
			if key in inputs:
				# Multiple inbound links to same target key are collected as a list.
				existing = inputs[key]
				if isinstance(existing, list):
					existing.append(payload)
				else:
					inputs[key] = [existing, payload]
			else:
				inputs[key] = payload

			if fallback_data is None:
				fallback_data = payload

		if "Data" not in inputs and fallback_data is not None:
			inputs["Data"] = fallback_data
		if "Chunk" not in inputs and "Data" in inputs:
			inputs["Chunk"] = inputs["Data"]
		return inputs

	def _collect_terminal_outputs(
		self,
		plan: SchedulePlan,
		node_results: dict[str, NodeExecutionResult],
	) -> dict[str, Any]:
		terminal_ids = [nid for nid, children in plan.adjacency.items() if len(children) == 0]
		outputs: dict[str, Any] = {}
		for nid in terminal_ids:
			result = node_results.get(nid)
			if result is None:
				continue
			outputs[nid] = result.primary_output
		return outputs

