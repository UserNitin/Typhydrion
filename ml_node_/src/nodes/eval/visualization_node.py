"""
Visualization Node - Generate visual reports and charts.
"""
from __future__ import annotations

from typing import Any
import io
import base64

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class VisualizationNode(NodeRuntime):
    """
    Generate visualizations:
    - Loss/Accuracy curves
    - Confusion matrix
    - ROC curve
    - PR curve
    - Feature importance
    - Learning curve
    - Residual plot
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        metrics = inputs.get("Metrics", {})
        training_history = inputs.get("Training History", {})
        
        chart_type = self.get_option("Chart Type", "Confusion Matrix")
        live_update = self.get_option("Live Update", True)
        save_figure = self.get_option("Save Figure", False)
        theme = self.get_option("Theme", "Dark")
        
        if not MATPLOTLIB_AVAILABLE:
            return NodeResult(
                outputs={"Graph Data": {"error": "matplotlib not available"}},
                success=False,
                error_message="matplotlib not installed"
            )
        
        try:
            # Set theme
            self._set_theme(theme)
            
            # Generate chart
            fig = None
            chart_data = {}
            
            if chart_type == "Loss Curve":
                fig, chart_data = self._plot_loss_curve(training_history)
            
            elif chart_type == "Accuracy Curve":
                fig, chart_data = self._plot_accuracy_curve(training_history)
            
            elif chart_type == "Confusion Matrix":
                fig, chart_data = self._plot_confusion_matrix(metrics)
            
            elif chart_type == "ROC Curve":
                fig, chart_data = self._plot_roc_curve(metrics, inputs)
            
            elif chart_type == "PR Curve":
                fig, chart_data = self._plot_pr_curve(metrics, inputs)
            
            elif chart_type == "Feature Importance":
                fig, chart_data = self._plot_feature_importance(inputs)
            
            elif chart_type == "Learning Curve":
                fig, chart_data = self._plot_learning_curve(training_history)
            
            elif chart_type == "Residual Plot":
                fig, chart_data = self._plot_residuals(inputs)
            
            # Convert figure to base64 if available
            image_data = None
            if fig is not None:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                           facecolor='#1a1a2e', edgecolor='none')
                buf.seek(0)
                image_data = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
            
            return NodeResult(
                outputs={
                    "Graph Data": {
                        "chart_type": chart_type,
                        "image_base64": image_data,
                        "data": chart_data,
                    },
                },
                metadata={
                    "chart_type": chart_type,
                    "has_image": image_data is not None,
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _set_theme(self, theme: str) -> None:
        """Set matplotlib theme."""
        if theme == "Dark":
            plt.style.use('dark_background')
            plt.rcParams['axes.facecolor'] = '#1a1a2e'
            plt.rcParams['figure.facecolor'] = '#1a1a2e'
            plt.rcParams['text.color'] = 'white'
            plt.rcParams['axes.labelcolor'] = 'white'
            plt.rcParams['xtick.color'] = 'white'
            plt.rcParams['ytick.color'] = 'white'
        elif theme == "Light":
            plt.style.use('default')
        elif theme == "Seaborn":
            plt.style.use('seaborn-v0_8-whitegrid')
        else:
            plt.style.use('bmh')
    
    def _plot_loss_curve(self, history: dict) -> tuple:
        """Plot training loss curve."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'loss' in history:
            ax.plot(history['loss'], label='Training Loss', color='#4ecdc4')
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Validation Loss', color='#ff6b6b')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, {"type": "loss_curve"}
    
    def _plot_accuracy_curve(self, history: dict) -> tuple:
        """Plot training accuracy curve."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'accuracy' in history:
            ax.plot(history['accuracy'], label='Training Accuracy', color='#4ecdc4')
        if 'val_accuracy' in history:
            ax.plot(history['val_accuracy'], label='Validation Accuracy', color='#ff6b6b')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Accuracy Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, {"type": "accuracy_curve"}
    
    def _plot_confusion_matrix(self, metrics: dict) -> tuple:
        """Plot confusion matrix."""
        cm = metrics.get('confusion_matrix')
        if cm is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No confusion matrix data', ha='center', va='center')
            return fig, {"type": "confusion_matrix", "data": None}
        
        cm = np.array(cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Labels
        classes = range(len(cm))
        ax.set(xticks=np.arange(len(classes)),
               yticks=np.arange(len(classes)),
               xticklabels=classes,
               yticklabels=classes,
               ylabel='True label',
               xlabel='Predicted label',
               title='Confusion Matrix')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        return fig, {"type": "confusion_matrix", "data": cm.tolist()}
    
    def _plot_roc_curve(self, metrics: dict, inputs: dict) -> tuple:
        """Plot ROC curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            from sklearn.metrics import roc_curve, auc
            
            y_test = inputs.get("y_test")
            probabilities = inputs.get("Probabilities")
            
            if y_test is not None and probabilities is not None:
                y_score = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color='#4ecdc4', lw=2,
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                
                return fig, {"type": "roc_curve", "auc": roc_auc}
        except Exception:
            pass
        
        ax.text(0.5, 0.5, 'No ROC curve data', ha='center', va='center')
        return fig, {"type": "roc_curve", "data": None}
    
    def _plot_pr_curve(self, metrics: dict, inputs: dict) -> tuple:
        """Plot Precision-Recall curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            y_test = inputs.get("y_test")
            probabilities = inputs.get("Probabilities")
            
            if y_test is not None and probabilities is not None:
                y_score = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
                precision, recall, _ = precision_recall_curve(y_test, y_score)
                avg_precision = average_precision_score(y_test, y_score)
                
                ax.plot(recall, precision, color='#4ecdc4', lw=2,
                       label=f'PR curve (AP = {avg_precision:.2f})')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                ax.legend(loc="lower left")
                ax.grid(True, alpha=0.3)
                
                return fig, {"type": "pr_curve", "average_precision": avg_precision}
        except Exception:
            pass
        
        ax.text(0.5, 0.5, 'No PR curve data', ha='center', va='center')
        return fig, {"type": "pr_curve", "data": None}
    
    def _plot_feature_importance(self, inputs: dict) -> tuple:
        """Plot feature importance."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        importance = inputs.get("Feature Importance")
        if importance is None or not isinstance(importance, dict):
            ax.text(0.5, 0.5, 'No feature importance data', ha='center', va='center')
            return fig, {"type": "feature_importance", "data": None}
        
        # Sort by importance
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
        features, values = zip(*sorted_imp)
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color='#4ecdc4', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Top 20 Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        return fig, {"type": "feature_importance", "data": dict(sorted_imp)}
    
    def _plot_learning_curve(self, history: dict) -> tuple:
        """Plot learning curve."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        if 'loss' in history:
            ax1.plot(history.get('loss', []), label='Train', color='#4ecdc4')
            ax1.plot(history.get('val_loss', []), label='Val', color='#ff6b6b')
            ax1.set_title('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        if 'accuracy' in history:
            ax2.plot(history.get('accuracy', []), label='Train', color='#4ecdc4')
            ax2.plot(history.get('val_accuracy', []), label='Val', color='#ff6b6b')
            ax2.set_title('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, {"type": "learning_curve"}
    
    def _plot_residuals(self, inputs: dict) -> tuple:
        """Plot residuals for regression."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        predictions = inputs.get("Predictions")
        y_test = inputs.get("y_test")
        
        if predictions is None or y_test is None:
            ax.text(0.5, 0.5, 'No residual data', ha='center', va='center')
            return fig, {"type": "residual_plot", "data": None}
        
        predictions = np.array(predictions).ravel()
        y_test = np.array(y_test).ravel()
        residuals = y_test - predictions
        
        ax.scatter(predictions, residuals, alpha=0.5, color='#4ecdc4')
        ax.axhline(y=0, color='#ff6b6b', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        return fig, {"type": "residual_plot"}
