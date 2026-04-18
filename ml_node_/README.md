# ml_node_

ml_node_ is a desktop machine learning workflow builder with a visual node-based editor.
It is designed for creating end-to-end ML pipelines by connecting nodes for data loading,
preprocessing, splitting, model training, evaluation, and export.

The application uses PySide6 (Qt for Python) for the UI and a graph execution engine for
pipeline scheduling and runtime orchestration.

## What This Project Does

- Build ML workflows visually instead of writing every step manually.
- Execute pipelines in topological order with validation and scheduling.
- Support a broad node catalog across IO, preprocessing, modeling, evaluation, and utility tasks.
- Produce artifacts in local working folders (models, logs, reports, cached data).

## Current State

Implemented and active:
- Desktop shell and node editor UI.
- Graph validation, scheduling, and pipeline execution core.
- Runtime node registry and many node categories.

Partially scaffolded (in progress):
- Some modules under advisor, data_engine, storage, resources, and app layers.
- Packaging metadata and dependency lock-in are still being completed.

## High-Level Architecture

### Entry Point

Main startup lives in src/main.py and performs the following flow:
1. Create QApplication.
2. Set application and organization name (Typhydrion).
3. Create and show MainWindow.
4. Start the Qt event loop.

### Execution Core

The execution engine is split into clear responsibilities:
- src/engine/graph_validator.py: structure checks, adjacency mapping, cycle/error reporting, topological sorting.
- src/engine/scheduler.py: scheduling and execution-level grouping based on validated graph data.
- src/engine/node_runner.py: per-node runtime resolution and normalized execution outputs.
- src/engine/pipeline_executor.py: end-to-end orchestration of node execution across the graph.

### Node Runtime System

- src/nodes/registry.py maps visual node names to runtime implementations.
- src/nodes/base/node_runtime.py defines NodeRuntime, NodeResult, and NodeContext contracts.
- Node catalog includes data IO, preprocessing, split/flow, model, training, evaluation, and utility/system nodes.

### UI Layer

- src/ui/main_window.py provides the main desktop shell and integrates key app windows.
- src/ui/windows/node_editor_window.py contains graph editing and execution interaction behavior.
- src/ui/app_settings.py centralizes typed application settings and defaults.

## Repository Layout

```
ml_node_/
	assets/              UI resources (fonts, icons, themes)
	data/                Local data artifacts (cache, chunks, samples)
	logs/                Runtime logs and diagnostics
	models/              Saved model artifacts
	reports/             Generated docs, profiles, and analysis output
	scripts/             Utility scripts
	src/                 Application source code
		advisor/           Assistant/advisor-related modules (partially scaffolded)
		app/               Application state/controller layer (partially scaffolded)
		data_engine/       Data handling pipeline components (partially scaffolded)
		engine/            Graph validation, scheduling, execution
		ml/                ML support utilities
		nodes/             Node definitions and runtime implementations
		resources/         App resources and constants (partially scaffolded)
		storage/           Persistence and artifact handling (partially scaffolded)
		ui/                Windows, widgets, settings, interactions
		utils/             Shared utilities
		main.py            Desktop app entry point
	pyproject.toml       Packaging/tool metadata (currently empty)
	requirements.txt     Dependency manifest (currently empty)
	README.md            Project documentation
```

## Setup (Development)

Python 3.10+ is recommended.

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

Because requirements.txt is currently empty, install core dependencies manually for now
(minimum: PySide6), or populate requirements.txt first.

Example:

```powershell
pip install PySide6
```

## Run the Application

From the ml_node_ directory:

```powershell
python src/main.py
```

## Data and Artifacts

The application is structured to keep generated outputs local and organized:
- data/cache and data/chunks for intermediate/working data.
- models for trained/exported model outputs.
- reports for generated summaries and profiling files.
- logs for runtime diagnostics.

## Troubleshooting

- Import errors: verify the virtual environment is active and dependencies are installed.
- UI startup errors: confirm PySide6 is installed in the active environment.
- Pipeline/runtime issues: inspect recent files in logs and reports.
- Relative path issues: run commands from the ml_node_ root directory.

## Recommended Next Improvements

1. Populate requirements.txt with pinned dependencies.
2. Populate pyproject.toml with project metadata and tooling config.
3. Add automated tests for graph_validator, scheduler, and pipeline_executor.
4. Define stable persistence contracts in storage modules.

## License

Add project license information here.
