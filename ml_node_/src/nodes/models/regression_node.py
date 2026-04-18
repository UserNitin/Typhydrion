"""
Regression Model Node - Train regression models.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe

# Try to import sklearn
try:
    from sklearn.linear_model import (
        LinearRegression, Ridge, Lasso, ElasticNet
    )
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor,
        AdaBoostRegressor
    )
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import LightGBM
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Try to import CatBoost
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class RegressionNode(NodeRuntime):
    """
    Regression model with support for:
    - Linear Regression
    - Ridge, Lasso, ElasticNet
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - XGBoost, LightGBM, CatBoost
    - SVR, KNN Regressor, Decision Tree
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
            
            if X_test is not None:
                X_test = ensure_dataframe(X_test)
                predictions = model.predict(X_test.fillna(0))
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(model, "feature_importances_"):
                feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            elif hasattr(model, "coef_"):
                coef = model.coef_.ravel() if hasattr(model.coef_, 'ravel') else [model.coef_]
                if len(coef) == len(X_train.columns):
                    feature_importance = dict(zip(X_train.columns, np.abs(coef)))
            
            return NodeResult(
                outputs={
                    "Trained Model": model,
                    "Predictions": predictions,
                    "Feature Importance": feature_importance,
                },
                metadata={
                    "algorithm": algorithm,
                    "n_samples": len(X_train),
                    "n_features": X_train.shape[1],
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _create_model(self, algorithm: str):
        """Create the appropriate model based on algorithm name."""
        if not SKLEARN_AVAILABLE:
            return None
        
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0, random_state=42),
            "Lasso": Lasso(alpha=1.0, random_state=42),
            "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "SVR": SVR(),
            "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        }
        
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42)
        
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        
        if CATBOOST_AVAILABLE:
            models["CatBoost"] = CatBoostRegressor(iterations=100, random_state=42, verbose=0)
        
        return models.get(algorithm)
    
    def predict(self, X) -> np.ndarray:
        """Make predictions using fitted model."""
        model = self.get_fitted_state("model")
        if model is None:
            raise ValueError("Model not fitted")
        
        X = ensure_dataframe(X)
        return model.predict(X.fillna(0))
