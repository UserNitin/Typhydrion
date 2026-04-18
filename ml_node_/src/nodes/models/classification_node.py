"""
Classification Model Node - Train classification models.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe

# Try to import sklearn
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        AdaBoostClassifier
    )
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import LightGBM
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Try to import CatBoost
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class ClassificationNode(NodeRuntime):
    """
    Classification model with support for:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost, LightGBM, CatBoost
    - SVM, KNN, Decision Tree
    - Naive Bayes
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        X_train = inputs.get("X_train")
        y_train = inputs.get("y_train")
        X_test = inputs.get("X_test")
        
        if X_train is None or y_train is None:
            return NodeResult(outputs={}, success=False, error_message="Missing training data")
        
        X_train = ensure_dataframe(X_train)
        y_train = np.array(y_train).ravel() if hasattr(y_train, '__len__') else y_train
        
        algorithm = self.get_option("Algorithm", "Random Forest")
        
        try:
            model = self._create_model(algorithm)
            
            if model is None:
                return NodeResult(outputs={}, success=False, 
                                error_message=f"Algorithm '{algorithm}' not available")
            
            # Train the model
            model.fit(X_train.fillna(0), y_train)
            
            # Store fitted model
            self.set_fitted_state("model", model)
            self.set_fitted_state("feature_names", X_train.columns.tolist())
            
            # Make predictions if test data provided
            predictions = None
            probabilities = None
            
            if X_test is not None:
                X_test = ensure_dataframe(X_test)
                predictions = model.predict(X_test.fillna(0))
                
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_test.fillna(0))
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(model, "feature_importances_"):
                feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            elif hasattr(model, "coef_"):
                coef = model.coef_.ravel() if model.coef_.ndim > 1 else model.coef_
                feature_importance = dict(zip(X_train.columns, np.abs(coef)))
            
            return NodeResult(
                outputs={
                    "Trained Model": model,
                    "Predictions": predictions,
                    "Probabilities": probabilities,
                    "Feature Importance": feature_importance,
                },
                metadata={
                    "algorithm": algorithm,
                    "n_samples": len(X_train),
                    "n_features": X_train.shape[1],
                    "n_classes": len(np.unique(y_train)),
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _create_model(self, algorithm: str):
        """Create the appropriate model based on algorithm name."""
        if not SKLEARN_AVAILABLE:
            return None
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(probability=True, random_state=42),
            "Naive Bayes": GaussianNB(),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        }
        
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'
            )
        
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        
        if CATBOOST_AVAILABLE:
            models["CatBoost"] = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
        
        return models.get(algorithm)
    
    def predict(self, X) -> np.ndarray:
        """Make predictions using fitted model."""
        model = self.get_fitted_state("model")
        if model is None:
            raise ValueError("Model not fitted")
        
        X = ensure_dataframe(X)
        return model.predict(X.fillna(0))
    
    def predict_proba(self, X) -> np.ndarray:
        """Get prediction probabilities."""
        model = self.get_fitted_state("model")
        if model is None or not hasattr(model, "predict_proba"):
            raise ValueError("Model not fitted or doesn't support probabilities")
        
        X = ensure_dataframe(X)
        return model.predict_proba(X.fillna(0))
