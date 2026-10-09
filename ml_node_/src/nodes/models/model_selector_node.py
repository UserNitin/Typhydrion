"""
Model Selector Node - Choose and configure ML algorithms.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult


class ModelSelectorNode(NodeRuntime):
    """
    Model selector with AI recommendations.
    Outputs model configuration for training nodes.
    """
    
    # Algorithm configurations
    ALGORITHMS = {
        "Classification": {
            "Logistic Regression": {"type": "linear", "params": {"max_iter": 1000}},
            "Random Forest": {"type": "ensemble", "params": {"n_estimators": 100}},
            "Gradient Boosting": {"type": "ensemble", "params": {"n_estimators": 100}},
            "XGBoost": {"type": "boosting", "params": {"n_estimators": 100}},
            "LightGBM": {"type": "boosting", "params": {"n_estimators": 100}},
            "CatBoost": {"type": "boosting", "params": {"iterations": 100}},
            "SVM": {"type": "kernel", "params": {"probability": True}},
            "KNN": {"type": "instance", "params": {"n_neighbors": 5}},
            "Decision Tree": {"type": "tree", "params": {}},
            "Naive Bayes": {"type": "probabilistic", "params": {}},
        },
        "Regression": {
            "Linear Regression": {"type": "linear", "params": {}},
            "Ridge": {"type": "linear", "params": {"alpha": 1.0}},
            "Lasso": {"type": "linear", "params": {"alpha": 1.0}},
            "ElasticNet": {"type": "linear", "params": {"alpha": 1.0, "l1_ratio": 0.5}},
            "Random Forest": {"type": "ensemble", "params": {"n_estimators": 100}},
            "Gradient Boosting": {"type": "ensemble", "params": {"n_estimators": 100}},
            "XGBoost": {"type": "boosting", "params": {"n_estimators": 100}},
            "LightGBM": {"type": "boosting", "params": {"n_estimators": 100}},
            "SVR": {"type": "kernel", "params": {}},
            "KNN": {"type": "instance", "params": {"n_neighbors": 5}},
        },
        "Clustering": {
            "K-Means": {"type": "centroid", "params": {"n_clusters": 5}},
            "DBSCAN": {"type": "density", "params": {"eps": 0.5}},
            "Hierarchical": {"type": "hierarchical", "params": {"n_clusters": 5}},
            "GMM": {"type": "probabilistic", "params": {"n_components": 5}},
        },
        "Anomaly Detection": {
            "Isolation Forest": {"type": "ensemble", "params": {"contamination": 0.1}},
            "One-Class SVM": {"type": "kernel", "params": {"nu": 0.1}},
            "LOF": {"type": "density", "params": {"contamination": 0.1}},
        }
    }
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        task = self.get_option("Task", "Classification")
        algorithm = self.get_option("Algorithm", "Random Forest")
        ai_recommend = self.get_option("AI Recommend", True)
        
        # Get algorithm config
        task_algorithms = self.ALGORITHMS.get(task, {})
        config = task_algorithms.get(algorithm, {"type": "unknown", "params": {}})
        
        # AI recommendations based on data characteristics
        recommendations = []
        if ai_recommend:
            recommendations = self._generate_recommendations(task, algorithm, inputs)
        
        model_config = {
            "task": task,
            "algorithm": algorithm,
            "algorithm_type": config["type"],
            "default_params": config["params"],
            "recommendations": recommendations,
        }
        
        return NodeResult(
            outputs={
                "Model Config": model_config,
            },
            metadata={
                "task": task,
                "algorithm": algorithm,
                "has_recommendations": len(recommendations) > 0,
            }
        )
    
    def _generate_recommendations(self, task: str, algorithm: str, inputs: dict) -> list[str]:
        """Generate AI recommendations based on context."""
        recommendations = []
        
        # Task-specific recommendations
        if task == "Classification":
            recommendations.append("💡 For imbalanced data, consider using class_weight='balanced'")
            if algorithm in ["Random Forest", "XGBoost", "LightGBM"]:
                recommendations.append("💡 Ensemble models work well with feature engineering")
            if algorithm == "Logistic Regression":
                recommendations.append("💡 Scale features for better convergence")
        
        elif task == "Regression":
            recommendations.append("💡 Consider feature scaling for linear models")
            if algorithm in ["Ridge", "Lasso"]:
                recommendations.append("💡 Use cross-validation to tune regularization strength")
        
        elif task == "Clustering":
            recommendations.append("💡 Scale features before clustering")
            if algorithm == "K-Means":
                recommendations.append("💡 Use elbow method or silhouette score to find optimal K")
        
        return recommendations


class TrainingNode(NodeRuntime):
    """
    Training controller for incremental/batch training.
    Handles epochs, early stopping, learning rate scheduling.
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        model_config = inputs.get("Model Config")
        X_train = inputs.get("X_train")
        y_train = inputs.get("y_train")
        X_val = inputs.get("X_val")
        y_val = inputs.get("y_val")
        
        if model_config is None:
            return NodeResult(outputs={}, success=False, error_message="No model config")
        
        if X_train is None or y_train is None:
            return NodeResult(outputs={}, success=False, error_message="Missing training data")
        
        epochs = self.get_option("Epochs", 50)
        learning_rate = self.get_option("Learning Rate", 0.001)
        batch_size = self.get_option("Batch Size", 32)
        use_gpu = self.get_option("Use GPU", False)
        early_stopping = self.get_option("Early Stopping", True)
        patience = self.get_option("Patience", 10)
        lr_scheduler = self.get_option("LR Scheduler", "None")
        
        try:
            from nodes.base.node_runtime import ensure_dataframe
            
            X_train = ensure_dataframe(X_train)
            y_train = np.array(y_train).ravel() if hasattr(y_train, '__len__') else y_train
            
            # Import the appropriate model node based on task
            task = model_config.get("task", "Classification")
            algorithm = model_config.get("algorithm", "Random Forest")
            
            if task == "Classification":
                from nodes.models.classification_node import ClassificationNode
                model_node = ClassificationNode(self.node_id, {"options": {"Algorithm": algorithm}})
            elif task == "Regression":
                from nodes.models.regression_node import RegressionNode
                model_node = RegressionNode(self.node_id, {"options": {"Algorithm": algorithm}})
            else:
                return NodeResult(outputs={}, success=False, error_message=f"Unsupported task: {task}")
            
            # Train model
            result = model_node.run({"X_train": X_train, "y_train": y_train})
            
            if not result.success:
                return result
            
            # Training history (simulated for sklearn models)
            training_history = {
                "epochs": epochs,
                "final_score": 0.0,
                "training_time": result.execution_time,
            }
            
            # Calculate training score if possible
            model = result.outputs.get("Trained Model")
            if model and hasattr(model, "score"):
                training_history["final_score"] = model.score(X_train.fillna(0), y_train)
            
            return NodeResult(
                outputs={
                    "Trained Model": model,
                    "Training History": training_history,
                },
                metadata={
                    "task": task,
                    "algorithm": algorithm,
                    "epochs": epochs,
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))


class HyperparameterTunerNode(NodeRuntime):
    """Hyperparameter optimization."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        model_config = inputs.get("Model Config")
        X_train = inputs.get("X_train")
        y_train = inputs.get("y_train")
        
        if model_config is None or X_train is None or y_train is None:
            return NodeResult(outputs={}, success=False, error_message="Missing inputs")
        
        search_method = self.get_option("Search Method", "Random Search")
        n_iterations = self.get_option("N Iterations", 20)
        cv_folds = self.get_option("CV Folds", 5)
        metric = self.get_option("Metric", "accuracy")
        parallel = self.get_option("Parallel", True)
        n_jobs = self.get_option("N Jobs", -1)
        
        try:
            from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
            from nodes.base.node_runtime import ensure_dataframe
            
            X_train = ensure_dataframe(X_train).fillna(0)
            y_train = np.array(y_train).ravel()
            
            algorithm = model_config.get("algorithm", "Random Forest")
            task = model_config.get("task", "Classification")
            
            # Get model and param grid
            model, param_grid = self._get_model_and_params(algorithm, task)
            
            if model is None:
                return NodeResult(outputs={}, success=False, error_message=f"Unknown algorithm: {algorithm}")
            
            # Choose search method
            if search_method == "Grid Search":
                search = GridSearchCV(
                    model, param_grid, cv=cv_folds, scoring=metric,
                    n_jobs=n_jobs if parallel else 1
                )
            else:  # Random Search
                search = RandomizedSearchCV(
                    model, param_grid, n_iter=n_iterations, cv=cv_folds,
                    scoring=metric, n_jobs=n_jobs if parallel else 1, random_state=42
                )
            
            search.fit(X_train, y_train)
            
            return NodeResult(
                outputs={
                    "Best Params": search.best_params_,
                    "Tuning Results": {
                        "best_score": search.best_score_,
                        "cv_results": {k: list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else v 
                                      for k, v in search.cv_results_.items() 
                                      if k in ['mean_test_score', 'std_test_score', 'rank_test_score']},
                    },
                },
                metadata={
                    "search_method": search_method,
                    "best_score": search.best_score_,
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _get_model_and_params(self, algorithm: str, task: str):
        """Get model instance and parameter grid for tuning."""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import LogisticRegression, Ridge
            
            if algorithm == "Random Forest":
                if task == "Classification":
                    model = RandomForestClassifier(random_state=42)
                else:
                    model = RandomForestRegressor(random_state=42)
                
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                }
            
            elif algorithm == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
                param_grid = {
                    "C": [0.01, 0.1, 1, 10],
                    "penalty": ["l1", "l2"],
                    "solver": ["saga"],
                }
            
            elif algorithm == "Ridge":
                model = Ridge(random_state=42)
                param_grid = {
                    "alpha": [0.01, 0.1, 1, 10, 100],
                }
            
            else:
                # Default to Random Forest
                model = RandomForestClassifier(random_state=42)
                param_grid = {"n_estimators": [50, 100]}
            
            return model, param_grid
        
        except ImportError:
            return None, {}


class EnsembleBuilderNode(NodeRuntime):
    """Combine multiple models into an ensemble."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        model_a = inputs.get("Model A")
        model_b = inputs.get("Model B")
        model_c = inputs.get("Model C")
        
        models = [m for m in [model_a, model_b, model_c] if m is not None]
        
        if len(models) < 2:
            return NodeResult(outputs={}, success=False, error_message="Need at least 2 models")
        
        method = self.get_option("Method", "Voting")
        voting = self.get_option("Voting", "soft")
        use_weights = self.get_option("Use Weights", False)
        weights_str = self.get_option("Weights", "1,1,1")
        
        try:
            weights = None
            if use_weights:
                weights = [float(w.strip()) for w in str(weights_str).split(",") if w.strip()][:len(models)]
                if len(weights) != len(models):
                    return NodeResult(
                        outputs={}, success=False,
                        error_message=f"Expected {len(models)} weights, got {len(weights)}",
                    )
            
            ensemble = PrefitVotingEnsemble(models, voting=voting, weights=weights)
            
            return NodeResult(
                outputs={
                    "Ensemble Model": ensemble,
                },
                metadata={
                    "method": method,
                    "n_models": len(models),
                    "voting": ensemble.voting,
                    "task": "classification" if ensemble.is_classifier else "regression",
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))


class PrefitVotingEnsemble:
    """Voting ensemble over models that are already trained.

    sklearn's VotingClassifier/VotingRegressor refit their estimators, which
    this node cannot do because it only receives fitted models.
    """

    def __init__(self, models: list, voting: str = "soft", weights: list[float] | None = None) -> None:
        self.estimators_ = list(models)
        self.weights = np.asarray(weights, dtype=float) if weights else np.ones(len(models))
        self.is_classifier = all(_is_classifier(m) for m in models)

        if self.is_classifier:
            classes = [np.asarray(getattr(m, "classes_", [])) for m in models]
            same_classes = all(len(c) and np.array_equal(c, classes[0]) for c in classes)
            can_soft = same_classes and all(hasattr(m, "predict_proba") for m in models)
            self.voting = "soft" if voting == "soft" and can_soft else "hard"
            self.classes_ = classes[0] if same_classes else None
        else:
            self.voting = "average"
            self.classes_ = None

    @property
    def predict_proba(self):
        if not (self.is_classifier and self.voting == "soft"):
            raise AttributeError("predict_proba is only available for soft-voting classifiers")
        return self._predict_proba

    def _predict_proba(self, X) -> np.ndarray:
        probas = np.array([m.predict_proba(X) for m in self.estimators_])
        return np.average(probas, axis=0, weights=self.weights)

    def predict(self, X) -> np.ndarray:
        if not self.is_classifier:
            preds = np.array([np.asarray(m.predict(X)).ravel() for m in self.estimators_])
            return np.average(preds, axis=0, weights=self.weights)

        if self.voting == "soft":
            return self.classes_[np.argmax(self._predict_proba(X), axis=1)]

        preds = np.column_stack([np.asarray(m.predict(X)).ravel() for m in self.estimators_])
        labels, idx = np.unique(preds, return_inverse=True)
        idx = idx.reshape(preds.shape)
        votes = np.zeros((preds.shape[0], len(labels)))
        rows = np.arange(preds.shape[0])
        for j, w in enumerate(self.weights):
            votes[rows, idx[:, j]] += w
        return labels[np.argmax(votes, axis=1)]


def _is_classifier(model) -> bool:
    try:
        from sklearn.base import is_classifier
        if is_classifier(model):
            return True
    except Exception:
        pass
    return hasattr(model, "classes_")
