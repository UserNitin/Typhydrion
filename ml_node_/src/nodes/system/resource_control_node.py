"""
Resource Control and System Nodes - Manage hardware resources and system utilities.
"""
from __future__ import annotations

from typing import Any
import os
import time
from datetime import datetime
import json
from pathlib import Path

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe

# Try to import psutil for resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ResourceControlNode(NodeRuntime):
    """
    Control and monitor hardware resource usage:
    - CPU limits
    - Memory caps
    - GPU selection
    - Priority settings
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        max_cpu = self.get_option("Max CPU %", 80)
        max_gpu = self.get_option("Max GPU %", 70)
        memory_cap_mb = self.get_option("Memory Cap (MB)", 4096)
        gpu_device = self.get_option("GPU Device", "Auto")
        priority = self.get_option("Priority", "Normal")
        memory_mapping = self.get_option("Memory Mapping", True)
        
        resource_limits = {
            "max_cpu_percent": max_cpu,
            "max_gpu_percent": max_gpu,
            "memory_cap_mb": memory_cap_mb,
            "gpu_device": gpu_device,
            "priority": priority,
            "memory_mapping": memory_mapping,
        }
        
        # Current resource usage
        current_usage = self._get_current_usage()
        
        # Apply settings (if possible)
        applied_settings = self._apply_settings(resource_limits)
        
        return NodeResult(
            outputs={
                "Resource Limits": resource_limits,
            },
            metadata={
                "current_usage": current_usage,
                "settings_applied": applied_settings,
            }
        )
    
    def _get_current_usage(self) -> dict:
        """Get current resource usage."""
        usage = {
            "cpu_percent": 0,
            "memory_percent": 0,
            "memory_used_mb": 0,
            "memory_available_mb": 0,
        }
        
        if PSUTIL_AVAILABLE:
            usage["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            usage["memory_percent"] = mem.percent
            usage["memory_used_mb"] = mem.used / 1024 / 1024
            usage["memory_available_mb"] = mem.available / 1024 / 1024
        
        return usage
    
    def _apply_settings(self, limits: dict) -> dict:
        """Apply resource limit settings."""
        applied = {}
        
        # Set number of threads for numpy/sklearn
        try:
            n_threads = max(1, int(os.cpu_count() * limits["max_cpu_percent"] / 100))
            os.environ["OMP_NUM_THREADS"] = str(n_threads)
            os.environ["MKL_NUM_THREADS"] = str(n_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
            applied["cpu_threads"] = n_threads
        except Exception:
            pass
        
        # GPU device selection
        if limits["gpu_device"] != "Auto":
            if limits["gpu_device"] == "CPU Only":
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                applied["gpu"] = "disabled"
            else:
                gpu_id = limits["gpu_device"].replace("GPU ", "")
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
                applied["gpu"] = gpu_id
        
        return applied


class DebugInspectorNode(NodeRuntime):
    """Inspect data and debug pipeline."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Any Data")
        
        show_shape = self.get_option("Show Shape", True)
        show_types = self.get_option("Show Types", True)
        show_stats = self.get_option("Show Stats", True)
        show_sample = self.get_option("Show Sample", True)
        sample_rows = self.get_option("Sample Rows", 5)
        breakpoint_enabled = self.get_option("Breakpoint", False)
        
        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "data_type": str(type(data)),
        }
        
        if data is None:
            debug_info["status"] = "No data received"
            return NodeResult(outputs={"Passthrough": None}, metadata=debug_info)
        
        # Try to convert to DataFrame for inspection
        try:
            df = ensure_dataframe(data)
            
            if show_shape:
                debug_info["shape"] = df.shape
                debug_info["rows"] = len(df)
                debug_info["columns"] = len(df.columns)
            
            if show_types:
                debug_info["dtypes"] = {str(k): str(v) for k, v in df.dtypes.items()}
            
            if show_stats:
                debug_info["memory_mb"] = df.memory_usage(deep=True).sum() / 1024 / 1024
                debug_info["missing_count"] = int(df.isnull().sum().sum())
                debug_info["missing_pct"] = float((df.isnull().sum().sum() / df.size) * 100) if df.size > 0 else 0
            
            if show_sample:
                debug_info["sample"] = df.head(sample_rows).to_dict()
            
        except Exception as e:
            debug_info["conversion_error"] = str(e)
            debug_info["raw_data_repr"] = repr(data)[:500]
        
        # Breakpoint (print and pause)
        if breakpoint_enabled:
            print("=" * 50)
            print("DEBUG INSPECTOR BREAKPOINT")
            print("=" * 50)
            for key, value in debug_info.items():
                print(f"{key}: {value}")
            print("=" * 50)
        
        return NodeResult(
            outputs={"Passthrough": data},
            metadata=debug_info
        )


class DataLoggerNode(NodeRuntime):
    """Log data to file or console."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Data")
        
        output_target = self.get_option("Output", "Console")
        log_file = self.get_option("Log File", "pipeline.log")
        level = self.get_option("Level", "INFO")
        timestamp = self.get_option("Timestamp", True)
        
        log_entry = {
            "level": level,
            "data_type": str(type(data)),
        }
        
        if timestamp:
            log_entry["timestamp"] = datetime.now().isoformat()
        
        if data is not None:
            try:
                df = ensure_dataframe(data)
                log_entry["shape"] = df.shape
                log_entry["columns"] = df.columns.tolist()
            except Exception:
                log_entry["data_repr"] = repr(data)[:200]
        
        log_message = json.dumps(log_entry, default=str)
        
        if output_target in ["Console", "Both"]:
            print(f"[{level}] {log_message}")
        
        if output_target in ["File", "Both"]:
            try:
                logs_dir = Path("logs")
                logs_dir.mkdir(exist_ok=True)
                
                with open(logs_dir / log_file, "a") as f:
                    f.write(log_message + "\n")
            except Exception as e:
                log_entry["file_error"] = str(e)
        
        return NodeResult(
            outputs={"Passthrough": data},
            metadata=log_entry
        )


class CheckpointNode(NodeRuntime):
    """Save/load pipeline state checkpoints."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        state = inputs.get("State")
        
        action = self.get_option("Action", "Save")
        checkpoint_name = self.get_option("Checkpoint Name", "checkpoint")
        include_data = self.get_option("Include Data", False)
        include_model = self.get_option("Include Model", True)
        
        checkpoints_dir = Path("data/cache")
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoints_dir / f"{checkpoint_name}.pkl"
        
        try:
            import pickle
            
            if action == "Save":
                save_state = {}
                
                if include_model and "model" in (state or {}):
                    save_state["model"] = state["model"]
                
                if include_data and "data" in (state or {}):
                    save_state["data"] = state["data"]
                
                save_state["metadata"] = {
                    "timestamp": datetime.now().isoformat(),
                    "checkpoint_name": checkpoint_name,
                }
                
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(save_state, f)
                
                return NodeResult(
                    outputs={"State": state},
                    metadata={"action": "saved", "path": str(checkpoint_path)}
                )
            
            elif action == "Load":
                if checkpoint_path.exists():
                    with open(checkpoint_path, "rb") as f:
                        loaded_state = pickle.load(f)
                    
                    return NodeResult(
                        outputs={"State": loaded_state},
                        metadata={"action": "loaded", "path": str(checkpoint_path)}
                    )
                else:
                    return NodeResult(
                        outputs={"State": state},
                        metadata={"action": "load_failed", "error": "Checkpoint not found"}
                    )
            
            else:  # Auto
                if checkpoint_path.exists():
                    with open(checkpoint_path, "rb") as f:
                        loaded_state = pickle.load(f)
                    return NodeResult(
                        outputs={"State": loaded_state},
                        metadata={"action": "auto_loaded"}
                    )
                else:
                    return NodeResult(
                        outputs={"State": state},
                        metadata={"action": "auto_no_checkpoint"}
                    )
        
        except Exception as e:
            return NodeResult(outputs={"State": state}, success=False, error_message=str(e))


class TimerNode(NodeRuntime):
    """Measure execution time."""
    
    _start_times: dict[str, float] = {}
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        _ = inputs.get("Start Signal")
        
        unit = self.get_option("Unit", "Seconds")
        log_time = self.get_option("Log Time", True)
        cumulative = self.get_option("Cumulative", False)
        
        current_time = time.time()
        timer_id = self.node_id
        
        if timer_id not in self._start_times:
            # First call - start timer
            self._start_times[timer_id] = current_time
            elapsed = 0
        else:
            # Subsequent call - calculate elapsed
            elapsed = current_time - self._start_times[timer_id]
            
            if not cumulative:
                self._start_times[timer_id] = current_time
        
        # Convert to requested unit
        if unit == "Milliseconds":
            elapsed = elapsed * 1000
        elif unit == "Minutes":
            elapsed = elapsed / 60
        
        if log_time:
            print(f"[TIMER] Elapsed: {elapsed:.3f} {unit.lower()}")
        
        return NodeResult(
            outputs={
                "Elapsed Time": elapsed,
                "End Signal": True,
            },
            metadata={
                "unit": unit,
                "elapsed": elapsed,
            }
        )


class LoopControllerNode(NodeRuntime):
    """Control iteration over data or parameters."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        iterator = inputs.get("Iterator")
        
        max_iterations = self.get_option("Max Iterations", 100)
        break_on_error = self.get_option("Break on Error", True)
        progress_bar = self.get_option("Progress Bar", True)
        
        if iterator is None:
            return NodeResult(outputs={}, success=False, error_message="No iterator provided")
        
        try:
            # Convert to list if needed
            if hasattr(iterator, "__iter__"):
                items = list(iterator)[:max_iterations]
            else:
                items = [iterator]
            
            results = []
            errors = []
            
            for i, item in enumerate(items):
                try:
                    results.append({
                        "index": i,
                        "item": item,
                    })
                    
                    if progress_bar:
                        pct = (i + 1) / len(items) * 100
                        print(f"\rProgress: {pct:.1f}% ({i+1}/{len(items)})", end="")
                        
                except Exception as e:
                    errors.append({"index": i, "error": str(e)})
                    if break_on_error:
                        break
            
            if progress_bar:
                print()  # New line after progress
            
            return NodeResult(
                outputs={
                    "Current Item": results[-1]["item"] if results else None,
                    "Index": len(results) - 1,
                    "Done Signal": True,
                },
                metadata={
                    "total_iterations": len(results),
                    "errors": len(errors),
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))


class NoteCommentNode(NodeRuntime):
    """Documentation node - doesn't process data."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        title = self.get_option("Title", "Note")
        description = self.get_option("Description", "")
        color = self.get_option("Color", "Yellow")
        
        return NodeResult(
            outputs={},
            metadata={
                "title": title,
                "description": description,
                "color": color,
                "is_note": True,
            }
        )


class AIAdvisorNode(NodeRuntime):
    """AI-powered suggestions and recommendations."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Data")
        pipeline_state = inputs.get("Pipeline State", {})
        
        algo_suggestions = self.get_option("Algorithm Suggestions", True)
        preprocessing_tips = self.get_option("Preprocessing Tips", True)
        hp_hints = self.get_option("Hyperparameter Hints", True)
        quality_warnings = self.get_option("Data Quality Warnings", True)
        verbosity = self.get_option("Verbosity", "Normal")
        
        suggestions = []
        warnings = []
        
        if data is not None:
            try:
                df = ensure_dataframe(data)
                
                # Data quality analysis
                if quality_warnings:
                    missing_pct = (df.isnull().sum().sum() / df.size) * 100 if df.size > 0 else 0
                    
                    if missing_pct > 30:
                        warnings.append({
                            "type": "high_missing",
                            "severity": "high",
                            "message": f"⚠️ {missing_pct:.1f}% missing values detected. Consider imputation or dropping columns.",
                        })
                    elif missing_pct > 10:
                        warnings.append({
                            "type": "moderate_missing",
                            "severity": "medium",
                            "message": f"ℹ️ {missing_pct:.1f}% missing values. Simple imputation recommended.",
                        })
                    
                    # Check for high cardinality
                    for col in df.select_dtypes(include=["object", "category"]).columns:
                        if df[col].nunique() > 100:
                            warnings.append({
                                "type": "high_cardinality",
                                "severity": "medium",
                                "message": f"ℹ️ Column '{col}' has high cardinality ({df[col].nunique()} unique values). Consider encoding or grouping.",
                            })
                
                # Preprocessing suggestions
                if preprocessing_tips:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_cols) > 0:
                        suggestions.append({
                            "type": "scaling",
                            "message": "💡 Consider scaling numeric features for distance-based algorithms (KNN, SVM).",
                        })
                    
                    cat_cols = df.select_dtypes(include=["object", "category"]).columns
                    if len(cat_cols) > 0:
                        suggestions.append({
                            "type": "encoding",
                            "message": f"💡 {len(cat_cols)} categorical columns detected. Apply encoding before modeling.",
                        })
                
                # Algorithm suggestions
                if algo_suggestions:
                    n_samples = len(df)
                    n_features = len(df.columns)
                    
                    if n_samples < 1000:
                        suggestions.append({
                            "type": "algorithm",
                            "message": "💡 Small dataset - consider simpler models (Logistic Regression, Decision Trees) to avoid overfitting.",
                        })
                    elif n_samples > 100000:
                        suggestions.append({
                            "type": "algorithm",
                            "message": "💡 Large dataset - tree-based ensembles (XGBoost, LightGBM) often perform well.",
                        })
                    
                    if n_features > 100:
                        suggestions.append({
                            "type": "feature_selection",
                            "message": "💡 High-dimensional data - consider feature selection or dimensionality reduction.",
                        })
                
                # Hyperparameter hints
                if hp_hints:
                    suggestions.append({
                        "type": "hyperparameter",
                        "message": "💡 Use cross-validation with hyperparameter tuning for better generalization.",
                    })
                
            except Exception as e:
                warnings.append({
                    "type": "analysis_error",
                    "severity": "low",
                    "message": f"Could not analyze data: {str(e)}",
                })
        
        return NodeResult(
            outputs={
                "Suggestions": suggestions,
                "Warnings": warnings,
            },
            metadata={
                "n_suggestions": len(suggestions),
                "n_warnings": len(warnings),
                "verbosity": verbosity,
            }
        )
