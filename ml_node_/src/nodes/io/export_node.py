"""
Export and Inference Nodes - Save models and run predictions.
"""
from __future__ import annotations

from typing import Any
import os
from pathlib import Path

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe


class ExportModelNode(NodeRuntime):
    """
    Export trained models in various formats:
    - pickle
    - joblib
    - onnx
    - tensorflow SavedModel
    - pytorch
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        model = inputs.get("Trained Model")
        preprocessors = inputs.get("Preprocessors", {})
        
        if model is None:
            return NodeResult(outputs={}, success=False, error_message="No model to export")
        
        format_type = self.get_option("Format", "joblib")
        model_name = self.get_option("Model Name", "my_model")
        include_pipeline = self.get_option("Include Pipeline", True)
        compress = self.get_option("Compress", False)
        version = self.get_option("Version", "1.0.0")
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        file_path = models_dir / f"{model_name}_v{version}"
        
        try:
            if format_type == "joblib":
                import joblib
                full_path = str(file_path) + ".joblib"
                
                if include_pipeline and preprocessors:
                    save_obj = {"model": model, "preprocessors": preprocessors}
                else:
                    save_obj = model
                
                if compress:
                    joblib.dump(save_obj, full_path, compress=3)
                else:
                    joblib.dump(save_obj, full_path)
            
            elif format_type == "pickle":
                import pickle
                full_path = str(file_path) + ".pkl"
                
                if include_pipeline and preprocessors:
                    save_obj = {"model": model, "preprocessors": preprocessors}
                else:
                    save_obj = model
                
                with open(full_path, "wb") as f:
                    pickle.dump(save_obj, f)
            
            elif format_type == "onnx":
                # ONNX export requires sklearn-onnx
                try:
                    from skl2onnx import convert_sklearn
                    from skl2onnx.common.data_types import FloatTensorType
                    
                    # Get input shape
                    n_features = inputs.get("n_features", 10)
                    initial_type = [('float_input', FloatTensorType([None, n_features]))]
                    
                    onnx_model = convert_sklearn(model, initial_types=initial_type)
                    full_path = str(file_path) + ".onnx"
                    
                    with open(full_path, "wb") as f:
                        f.write(onnx_model.SerializeToString())
                        
                except ImportError:
                    return NodeResult(outputs={}, success=False, 
                                    error_message="skl2onnx not installed for ONNX export")
            
            elif format_type == "tensorflow":
                try:
                    full_path = str(file_path) + "_tf"
                    model.save(full_path)
                except Exception as e:
                    return NodeResult(outputs={}, success=False, 
                                    error_message=f"TensorFlow save failed: {e}")
            
            elif format_type == "pytorch":
                try:
                    import torch
                    full_path = str(file_path) + ".pt"
                    torch.save(model.state_dict(), full_path)
                except Exception as e:
                    return NodeResult(outputs={}, success=False, 
                                    error_message=f"PyTorch save failed: {e}")
            
            else:
                # Default to joblib
                import joblib
                full_path = str(file_path) + ".joblib"
                joblib.dump(model, full_path)
            
            return NodeResult(
                outputs={
                    "Export Path": full_path,
                },
                metadata={
                    "format": format_type,
                    "model_name": model_name,
                    "version": version,
                    "file_size_mb": os.path.getsize(full_path) / 1024 / 1024 if os.path.exists(full_path) else 0,
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))


class InferenceNode(NodeRuntime):
    """Run predictions on new data."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        model = inputs.get("Model")
        new_data = inputs.get("New Data")
        
        if model is None or new_data is None:
            return NodeResult(outputs={}, success=False, error_message="Missing model or data")
        
        new_data = ensure_dataframe(new_data)
        
        output_type = self.get_option("Output Type", "Both")
        apply_threshold = self.get_option("Apply Threshold", True)
        threshold = self.get_option("Threshold", 0.5)
        batch_mode = self.get_option("Batch Mode", True)
        
        try:
            # Make predictions
            predictions = model.predict(new_data.fillna(0))
            
            # Get probabilities if available
            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(new_data.fillna(0))
                
                # Apply threshold for binary classification
                if apply_threshold and probabilities.shape[1] == 2:
                    predictions = (probabilities[:, 1] >= threshold).astype(int)
            
            outputs = {}
            
            if output_type in ["Labels", "Both"]:
                outputs["Predictions"] = pd.Series(predictions, name="prediction")
            
            if output_type in ["Probabilities", "Both"] and probabilities is not None:
                prob_df = pd.DataFrame(
                    probabilities,
                    columns=[f"prob_class_{i}" for i in range(probabilities.shape[1])]
                )
                outputs["Probabilities"] = prob_df
            
            return NodeResult(
                outputs=outputs,
                metadata={
                    "n_predictions": len(predictions),
                    "output_type": output_type,
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
