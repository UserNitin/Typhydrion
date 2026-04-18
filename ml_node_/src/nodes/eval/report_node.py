"""
Report Generator Node - Generate comprehensive analysis reports.
"""
from __future__ import annotations

from typing import Any
import json
from datetime import datetime

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe


class ReportNode(NodeRuntime):
    """
    Generate analysis reports in various formats:
    - HTML
    - PDF (if libraries available)
    - Markdown
    - JSON
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        metrics = inputs.get("Metrics", {})
        visualizations = inputs.get("Visualizations", {})
        explanations = inputs.get("Explanations", {})
        
        format_type = self.get_option("Format", "HTML")
        include_graphs = self.get_option("Include Graphs", True)
        include_summary = self.get_option("Include Data Summary", True)
        include_code = self.get_option("Include Code", False)
        
        try:
            if format_type == "HTML":
                report = self._generate_html_report(metrics, visualizations, explanations, include_graphs, include_summary)
            elif format_type == "Markdown":
                report = self._generate_markdown_report(metrics, visualizations, explanations, include_summary)
            elif format_type == "JSON":
                report = self._generate_json_report(metrics, visualizations, explanations)
            else:
                report = self._generate_html_report(metrics, visualizations, explanations, include_graphs, include_summary)
            
            return NodeResult(
                outputs={
                    "Report File": report,
                },
                metadata={
                    "format": format_type,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _generate_html_report(self, metrics: dict, visualizations: dict, 
                              explanations: dict, include_graphs: bool, include_summary: bool) -> str:
        """Generate HTML report."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Pipeline Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e8e8e8;
            line-height: 1.6;
            padding: 40px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #4ecdc4, #44a08d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        h2 {
            font-size: 1.5em;
            margin: 30px 0 15px;
            color: #4ecdc4;
            border-bottom: 2px solid #4ecdc4;
            padding-bottom: 10px;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .metric {
            display: inline-block;
            padding: 15px 25px;
            margin: 10px;
            background: rgba(78, 205, 196, 0.1);
            border-radius: 8px;
            border-left: 4px solid #4ecdc4;
        }
        .metric-label { font-size: 0.9em; color: #888; }
        .metric-value { font-size: 1.8em; font-weight: bold; color: #4ecdc4; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        th { background: rgba(78, 205, 196, 0.2); color: #4ecdc4; }
        tr:hover { background: rgba(255, 255, 255, 0.03); }
        .timestamp { color: #666; font-size: 0.9em; margin-top: 40px; text-align: center; }
        img { max-width: 100%; border-radius: 8px; margin: 15px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 ML Pipeline Report</h1>
        <p>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
"""
        
        # Metrics section
        if metrics:
            html += """
        <h2>📊 Performance Metrics</h2>
        <div class="card">
"""
            for key, value in metrics.items():
                if key not in ['confusion_matrix', 'classification_report'] and not isinstance(value, (dict, list)):
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    html += f"""
            <div class="metric">
                <div class="metric-label">{key.replace('_', ' ').title()}</div>
                <div class="metric-value">{formatted_value}</div>
            </div>
"""
            html += """
        </div>
"""
            
            # Classification report table
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                if isinstance(report, dict):
                    html += """
        <h2>📋 Classification Report</h2>
        <div class="card">
            <table>
                <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
"""
                    for cls, vals in report.items():
                        if isinstance(vals, dict):
                            html += f"""
                <tr>
                    <td>{cls}</td>
                    <td>{vals.get('precision', 0):.4f}</td>
                    <td>{vals.get('recall', 0):.4f}</td>
                    <td>{vals.get('f1-score', 0):.4f}</td>
                    <td>{int(vals.get('support', 0))}</td>
                </tr>
"""
                    html += """
            </table>
        </div>
"""
        
        # Visualizations
        if include_graphs and visualizations:
            html += """
        <h2>📈 Visualizations</h2>
        <div class="card">
"""
            if isinstance(visualizations, dict) and 'image_base64' in visualizations:
                html += f"""
            <img src="data:image/png;base64,{visualizations['image_base64']}" alt="Visualization">
"""
            html += """
        </div>
"""
        
        # Explanations
        if explanations:
            html += """
        <h2>🔍 Model Explanations</h2>
        <div class="card">
"""
            if isinstance(explanations, dict):
                for key, value in explanations.items():
                    html += f"<p><strong>{key}:</strong> {value}</p>\n"
            html += """
        </div>
"""
        
        html += """
        <p class="timestamp">Report generated by Typhydrion ML Pipeline</p>
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_markdown_report(self, metrics: dict, visualizations: dict, 
                                   explanations: dict, include_summary: bool) -> str:
        """Generate Markdown report."""
        md = f"""# ML Pipeline Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Performance Metrics

"""
        
        if metrics:
            for key, value in metrics.items():
                if key not in ['confusion_matrix', 'classification_report'] and not isinstance(value, (dict, list)):
                    formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
                    md += f"- **{key.replace('_', ' ').title()}**: {formatted}\n"
        
        md += "\n---\n\nReport generated by Typhydrion ML Pipeline\n"
        
        return md
    
    def _generate_json_report(self, metrics: dict, visualizations: dict, explanations: dict) -> str:
        """Generate JSON report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self._make_json_serializable(metrics),
            "visualizations": {"has_data": bool(visualizations)},
            "explanations": self._make_json_serializable(explanations),
        }
        
        return json.dumps(report, indent=2)
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        else:
            return obj


class ModelExplainerNode(NodeRuntime):
    """Explain model predictions using SHAP, LIME, etc."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        model = inputs.get("Trained Model")
        X_test = inputs.get("X_test")
        
        if model is None or X_test is None:
            return NodeResult(outputs={}, success=False, error_message="Missing model or test data")
        
        X_test = ensure_dataframe(X_test)
        
        method = self.get_option("Method", "Permutation")
        n_samples = self.get_option("N Samples", 100)
        global_importance = self.get_option("Global Importance", True)
        local_explanations = self.get_option("Local Explanations", False)
        
        try:
            explanations = {}
            feature_importance = {}
            
            # Permutation importance (works with any model)
            if method == "Permutation":
                from sklearn.inspection import permutation_importance
                
                y_test = inputs.get("y_test")
                if y_test is not None:
                    result = permutation_importance(
                        model, X_test.fillna(0), y_test,
                        n_repeats=10, random_state=42, n_jobs=-1
                    )
                    feature_importance = dict(zip(X_test.columns, result.importances_mean))
                    explanations["method"] = "Permutation Importance"
            
            # Tree-based importance
            elif method == "Tree Explainer":
                if hasattr(model, "feature_importances_"):
                    feature_importance = dict(zip(X_test.columns, model.feature_importances_))
                    explanations["method"] = "Tree Feature Importance"
            
            # SHAP (if available)
            elif method == "SHAP":
                try:
                    import shap
                    
                    explainer = shap.Explainer(model, X_test.head(100).fillna(0))
                    shap_values = explainer(X_test.head(n_samples).fillna(0))
                    
                    # Mean absolute SHAP values
                    mean_shap = np.abs(shap_values.values).mean(axis=0)
                    feature_importance = dict(zip(X_test.columns, mean_shap))
                    explanations["method"] = "SHAP Values"
                    
                except ImportError:
                    explanations["error"] = "SHAP library not installed"
            
            # Partial Dependence
            elif method == "Partial Dependence":
                try:
                    from sklearn.inspection import partial_dependence
                    
                    # Get top features
                    if hasattr(model, "feature_importances_"):
                        top_features = np.argsort(model.feature_importances_)[-5:][::-1]
                        pd_results = {}
                        
                        for idx in top_features:
                            feature_name = X_test.columns[idx]
                            result = partial_dependence(
                                model, X_test.fillna(0), [idx],
                                kind="average"
                            )
                            pd_results[feature_name] = {
                                "values": result["values"][0].tolist(),
                                "average": result["average"][0].tolist(),
                            }
                        
                        explanations["partial_dependence"] = pd_results
                        explanations["method"] = "Partial Dependence"
                        
                except Exception:
                    pass
            
            return NodeResult(
                outputs={
                    "Explanations": explanations,
                    "Feature Importance": feature_importance,
                },
                metadata={
                    "method": method,
                    "n_features": len(feature_importance),
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
