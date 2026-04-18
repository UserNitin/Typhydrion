# Project Overview: ml_node_

## 1. What this project is
ml_node_ is a desktop machine-learning workflow builder with a visual, node-based editor.
Users create pipelines by connecting nodes for:
- data loading and transformation,
- train/test splitting,
- model training,
- evaluation and reporting,
- export/inference utilities.

The UI is built with PySide6 (Qt for Python), and pipeline execution is handled by a graph engine.

## 2. Current repository shape
Top-level structure:
- assets/: UI resources (fonts, icons, themes)
- data/: local data artifacts (cache, chunks, samples)
- logs/: runtime logs
- models/: saved model outputs
- reports/: generated docs and analysis (module map, inventory, profiles)
- scripts/: utility scripts (currently empty)
- src/: application source code

## 3. Entry point and app startup
Primary app entry point:
- src/main.py

Startup flow:
1. Create QApplication.
2. Set application/org names as Typhydrion.
3. Create and show MainWindow.
4. Start Qt event loop.

## 4. Core architecture
The project is organized by responsibility:

- ui/: windows, widgets, styling, settings integration
- nodes/: all node definitions and runtime implementations
- engine/: graph validation, scheduling, node execution, pipeline orchestration
- app/: app-level controller/state placeholders
- advisor/: AI advisor placeholders
- data_engine/: dataset/chunking placeholders
- storage/: project/model/report persistence placeholders
- resources/: static string/icon placeholders
- ml/: algorithm-level support modules
- utils/: shared utilities

## 5. Execution engine (how a pipeline runs)
### 5.1 Graph validation
File: src/engine/graph_validator.py
Responsibilities:
- normalize graph payload into nodes + edges,
- validate node IDs and edge endpoints,
- detect self-loops and duplicates,
- produce adjacency/reverse-adjacency maps,
- topologically sort DAG,
- return warnings/errors in ValidationReport.

### 5.2 Scheduling
File: src/engine/scheduler.py
Responsibilities:
- call validator,
- fail if graph has structural errors/cycles,
- build stable execution order,
- group nodes into execution levels.

### 5.3 Single node execution
File: src/engine/node_runner.py
Responsibilities:
- resolve runtime class from node title via registry,
- execute node runtime,
- normalize options format,
- choose a primary output from runtime outputs,
- capture timing and errors,
- apply safe passthrough when runtime class is missing.

### 5.4 End-to-end pipeline execution
File: src/engine/pipeline_executor.py
Responsibilities:
- build schedule plan,
- construct per-node inputs from upstream edges,
- run nodes in topological order,
- stop early or continue based on fail_fast,
- gather terminal outputs and execution summary.

## 6. Node system
### 6.1 Registry
File: src/nodes/registry.py
Defines NODE_REGISTRY mapping UI node names to runtime classes.
Provides:
- get_node_runtime(node_name)
- list_available_nodes()
- get_node_categories()

### 6.2 Node categories implemented
- IO/Data: loader, preview, merger, selector, filter, export/inference/final output
- Preprocess: missing value handling, encoding, scaling, feature selection, text preprocessing
- Split/Flow: train/val/test split, cross validation, time-series split, routing/control
- Models: classification, regression, clustering, anomaly, neural net, selector
- Training: training controller, tuner, ensemble builder
- Evaluation: metrics, visualization, reporting, explainability
- System/Utility: resource/debug/log/checkpoint/timer/loop/comment/AI advisor

### 6.3 Base runtime contract
File: src/nodes/base/node_runtime.py
Core abstractions:
- NodeRuntime: abstract base class for node logic
- NodeResult: standardized output + metadata + timing + error status
- NodeContext: warnings/errors/cache/global state helpers
Utilities:
- ensure_dataframe(...)
- safe_column_select(...)

## 7. UI layer
### 7.1 Main window
File: src/ui/main_window.py
Contains the primary desktop shell and supporting animated UI components.
Integrates:
- Node editor window,
- node properties,
- data preview/statistics/profiler windows,
- output windows,
- settings and advisor windows,
- project save worker thread helpers.

### 7.2 Node editor
File: src/ui/windows/node_editor_window.py
Provides visual graph editing and execution interactions:
- node graph scene/view,
- edge/link management,
- async worker for single-node execution,
- async worker for full pipeline execution,
- signal-based UI updates (outputs, node selection, dataset events).

### 7.3 App settings
File: src/ui/app_settings.py
Centralized QSettings wrapper with typed defaults for:
- UI preferences,
- project/autosave behavior,
- performance options,
- pipeline behavior,
- graph/dataset/training defaults,
- logging/debug flags,
- AI assistant preferences,
- advanced/export/developer options.

## 8. Data and artifacts
Working directories support pipeline artifacts and profiling:
- data/cache and data/chunks for intermediate data handling,
- models for trained model artifacts,
- reports for generated summaries and profiler outputs,
- logs for runtime diagnostics.

## 9. Project maturity snapshot
Implemented and substantial:
- UI windows and node editor,
- node runtime ecosystem,
- graph validation + scheduler + execution pipeline,
- broad registry of ML workflow nodes.

Still scaffolded/placeholder in this snapshot:
- many modules under app/, advisor/, data_engine/, storage/, resources/ show minimal or empty content,
- README.md, requirements.txt, and pyproject.toml are currently empty.

## 10. Suggested immediate improvements
1. Fill README.md with setup/run/dev instructions and architecture notes.
2. Fill requirements.txt and pyproject.toml with real dependency and packaging metadata.
3. Add automated tests for engine modules (validator, scheduler, executor).
4. Define persistence contracts in storage/ (project save/load, model/report storage).
5. Add integration tests for common pipeline graphs and failure cases.

## 11. One-line summary
This is a PySide6 desktop node-based ML workflow platform with a strong visual editor and execution core, plus partial scaffolding for surrounding services that still need project-level packaging and documentation completion.
