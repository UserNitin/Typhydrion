from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time

from nodes.registry import get_node_runtime
from nodes.base.node_runtime import NodeResult


@dataclass(slots=True)
class NodeExecutionResult:
	node_id: str
	node_title: str
	success: bool
	outputs: dict[str, Any] = field(default_factory=dict)
	primary_output: Any = None
	metadata: dict[str, Any] = field(default_factory=dict)
	error_message: str | None = None
	started_at: float = 0.0
	finished_at: float = 0.0

	@property
	def execution_time(self) -> float:
		if self.finished_at <= self.started_at:
			return 0.0
		return self.finished_at - self.started_at


class NodeRunner:
	"""Executes a single node payload using the registered node runtime."""

	def run_node(self, node_payload: dict[str, Any], inputs: dict[str, Any]) -> NodeExecutionResult:
		node_id = str(node_payload.get("node_id", "") or "")
		title = str(node_payload.get("title", "") or "")
		options = node_payload.get("options", {}) or {}
		started_at = time.perf_counter()

		runtime_cls = get_node_runtime(title)
		if runtime_cls is None:
			# Safe passthrough fallback so orchestration can continue.
			passthrough = inputs.get("Data")
			if passthrough is None and inputs:
				passthrough = next(iter(inputs.values()))
			finished = time.perf_counter()
			return NodeExecutionResult(
				node_id=node_id,
				node_title=title,
				success=True,
				outputs={"Data": passthrough} if passthrough is not None else {},
				primary_output=passthrough,
				metadata={"warning": f"No runtime registered for '{title}', passthrough applied"},
				started_at=started_at,
				finished_at=finished,
			)

		try:
			node_data = {"options": self._options_to_list(options)}
			runtime = runtime_cls(node_id, node_data)
			result = runtime.execute(inputs)

			if not isinstance(result, NodeResult):
				finished = time.perf_counter()
				return NodeExecutionResult(
					node_id=node_id,
					node_title=title,
					success=True,
					outputs={},
					primary_output=inputs.get("Data"),
					metadata={"warning": "Runtime returned unexpected result type"},
					started_at=started_at,
					finished_at=finished,
				)

			primary = self._primary_output(result.outputs, title)
			finished = time.perf_counter()
			meta = dict(result.metadata or {})
			meta["runtime_execution_time"] = result.execution_time
			return NodeExecutionResult(
				node_id=node_id,
				node_title=title,
				success=bool(result.success),
				outputs=dict(result.outputs or {}),
				primary_output=primary,
				metadata=meta,
				error_message=result.error_message,
				started_at=started_at,
				finished_at=finished,
			)
		except Exception as exc:
			finished = time.perf_counter()
			return NodeExecutionResult(
				node_id=node_id,
				node_title=title,
				success=False,
				outputs={},
				primary_output=None,
				metadata={},
				error_message=str(exc),
				started_at=started_at,
				finished_at=finished,
			)

	def _options_to_list(self, options: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
		if isinstance(options, list):
			return [o for o in options if isinstance(o, dict)]
		if isinstance(options, dict):
			return [{"label": k, "value": v} for k, v in options.items()]
		return []

	def _primary_output(self, outputs: dict[str, Any], title: str) -> Any:
		import pandas as pd
		import numpy as np

		if not isinstance(outputs, dict):
			return None

		if "Split" in title and "X_train" in outputs:
			val = outputs["X_train"]
			if isinstance(val, pd.Series):
				return val.to_frame()
			return val

		priority_keys = [
			"Filtered Chunk", "Filtered Data", "Clean Chunk", "Converted Chunk",
			"Scaled Features", "Encoded Features", "Data", "Raw Data",
			"Merged Data", "Features", "X_train", "Predictions",
			"Scaled Data", "Encoded Data", "Preview", "Processed Data",
			"Joined DataFrame", "Selected DataFrame", "Cluster Labels",
			"Anomaly Scores", "Anomaly Labels",
		]
		for key in priority_keys:
			if key in outputs and outputs[key] is not None:
				val = outputs[key]
				if isinstance(val, pd.Series):
					return val.to_frame()
				if isinstance(val, np.ndarray) and val.ndim <= 2:
					return pd.DataFrame(val)
				return val

		for val in outputs.values():
			if val is not None:
				if isinstance(val, pd.Series):
					return val.to_frame()
				return val
		return None

