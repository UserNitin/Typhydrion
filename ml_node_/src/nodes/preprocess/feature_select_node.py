"""
Feature Selector Node - Select best features for model training.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

from nodes.base.node_runtime import NodeRuntime, NodeResult, ensure_dataframe

# Try to import sklearn
try:
    from sklearn.feature_selection import (
        VarianceThreshold, SelectKBest, RFE,
        f_classif, f_regression, mutual_info_classif, mutual_info_regression
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import Lasso, LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class FeatureSelectNode(NodeRuntime):
    """
    Select best features using various methods:
    - Variance Threshold
    - Correlation analysis
    - SelectKBest (statistical tests)
    - RFE (Recursive Feature Elimination)
    - L1 Regularization
    - Tree-based importance
    - Mutual Information
    """
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        features = inputs.get("Features")
        target = inputs.get("Target")
        
        if features is None:
            return NodeResult(outputs={}, success=False, error_message="No features data")
        
        X = ensure_dataframe(features).copy()
        y = target if target is not None else None
        
        method = self.get_option("Method", "Variance Threshold")
        k_features = self.get_option("K Features", 10)
        threshold = self.get_option("Threshold", 0.1)
        ai_suggestion = self.get_option("AI Suggestion", True)
        
        # Get numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols]
        
        try:
            if method == "Variance Threshold":
                selected_cols, scores = self._variance_threshold(X_numeric, threshold)
            
            elif method == "Correlation":
                selected_cols, scores = self._correlation_filter(X_numeric, y, threshold)
            
            elif method == "SelectKBest":
                if y is None:
                    return NodeResult(outputs={}, success=False, error_message="Target required for SelectKBest")
                selected_cols, scores = self._select_k_best(X_numeric, y, k_features)
            
            elif method == "RFE":
                if y is None:
                    return NodeResult(outputs={}, success=False, error_message="Target required for RFE")
                selected_cols, scores = self._rfe(X_numeric, y, k_features)
            
            elif method == "L1 Regularization":
                if y is None:
                    return NodeResult(outputs={}, success=False, error_message="Target required for L1")
                selected_cols, scores = self._l1_selection(X_numeric, y, k_features)
            
            elif method == "Tree Importance":
                if y is None:
                    return NodeResult(outputs={}, success=False, error_message="Target required for Tree Importance")
                selected_cols, scores = self._tree_importance(X_numeric, y, k_features)
            
            elif method == "Mutual Information":
                if y is None:
                    return NodeResult(outputs={}, success=False, error_message="Target required for MI")
                selected_cols, scores = self._mutual_information(X_numeric, y, k_features)
            
            else:
                selected_cols = numeric_cols[:k_features]
                scores = {col: 1.0 for col in selected_cols}
            
            # Create selected features DataFrame
            selected_features = X[selected_cols] if selected_cols else X
            
            # Add AI suggestion
            if ai_suggestion and scores:
                top_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k_features]
                scores["_ai_suggestion"] = f"Top features: {[f[0] for f in top_features[:5]]}"
            
            return NodeResult(
                outputs={
                    "Selected Features": selected_features,
                    "Feature Scores": scores,
                },
                metadata={
                    "method": method,
                    "selected_count": len(selected_cols),
                    "original_count": len(numeric_cols),
                }
            )
        
        except Exception as e:
            return NodeResult(outputs={}, success=False, error_message=str(e))
    
    def _variance_threshold(self, X: pd.DataFrame, threshold: float) -> tuple[list[str], dict]:
        """Filter features by variance."""
        variances = X.var()
        mask = variances > threshold
        selected = variances[mask].index.tolist()
        scores = variances.to_dict()
        return selected, scores
    
    def _correlation_filter(self, X: pd.DataFrame, y, threshold: float) -> tuple[list[str], dict]:
        """Filter by correlation with target or remove highly correlated features."""
        if y is not None:
            # Correlation with target
            y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
            correlations = X.apply(lambda col: col.corr(y_series))
            mask = correlations.abs() > threshold
            selected = correlations[mask].index.tolist()
            scores = correlations.abs().to_dict()
        else:
            # Remove highly correlated features among themselves
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > 1 - threshold)]
            selected = [col for col in X.columns if col not in to_drop]
            scores = {col: 1.0 for col in selected}
        return selected, scores
    
    def _select_k_best(self, X: pd.DataFrame, y, k: int) -> tuple[list[str], dict]:
        """Select K best features using statistical tests."""
        if not SKLEARN_AVAILABLE:
            return X.columns[:k].tolist(), {}
        
        X_clean = X.fillna(0)
        y_clean = pd.Series(y).fillna(0) if hasattr(y, 'fillna') else np.nan_to_num(y)
        
        # Determine if classification or regression
        is_classification = len(np.unique(y_clean)) < 20
        score_func = f_classif if is_classification else f_regression
        
        selector = SelectKBest(score_func=score_func, k=min(k, X_clean.shape[1]))
        selector.fit(X_clean, y_clean)
        
        scores = dict(zip(X.columns, selector.scores_))
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
        
        return selected, scores
    
    def _rfe(self, X: pd.DataFrame, y, k: int) -> tuple[list[str], dict]:
        """Recursive Feature Elimination."""
        if not SKLEARN_AVAILABLE:
            return X.columns[:k].tolist(), {}
        
        X_clean = X.fillna(0)
        y_clean = pd.Series(y).fillna(0) if hasattr(y, 'fillna') else np.nan_to_num(y)
        
        is_classification = len(np.unique(y_clean)) < 20
        estimator = RandomForestClassifier(n_estimators=50, random_state=42) if is_classification else RandomForestRegressor(n_estimators=50, random_state=42)
        
        selector = RFE(estimator, n_features_to_select=min(k, X_clean.shape[1]), step=1)
        selector.fit(X_clean, y_clean)
        
        scores = dict(zip(X.columns, 1.0 / selector.ranking_))
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
        
        return selected, scores
    
    def _l1_selection(self, X: pd.DataFrame, y, k: int) -> tuple[list[str], dict]:
        """L1 regularization based selection."""
        if not SKLEARN_AVAILABLE:
            return X.columns[:k].tolist(), {}
        
        X_clean = X.fillna(0)
        y_clean = pd.Series(y).fillna(0) if hasattr(y, 'fillna') else np.nan_to_num(y)
        
        is_classification = len(np.unique(y_clean)) < 20
        
        if is_classification:
            model = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42)
        else:
            model = Lasso(alpha=0.01, random_state=42)
        
        model.fit(X_clean, y_clean)
        
        coef = model.coef_.ravel() if hasattr(model.coef_, 'ravel') else model.coef_
        scores = dict(zip(X.columns, np.abs(coef)))
        
        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        selected = [f[0] for f in top_k if f[1] > 0]
        
        return selected, scores
    
    def _tree_importance(self, X: pd.DataFrame, y, k: int) -> tuple[list[str], dict]:
        """Tree-based feature importance."""
        if not SKLEARN_AVAILABLE:
            return X.columns[:k].tolist(), {}
        
        X_clean = X.fillna(0)
        y_clean = pd.Series(y).fillna(0) if hasattr(y, 'fillna') else np.nan_to_num(y)
        
        is_classification = len(np.unique(y_clean)) < 20
        model = RandomForestClassifier(n_estimators=100, random_state=42) if is_classification else RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_clean, y_clean)
        
        scores = dict(zip(X.columns, model.feature_importances_))
        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        selected = [f[0] for f in top_k]
        
        return selected, scores
    
    def _mutual_information(self, X: pd.DataFrame, y, k: int) -> tuple[list[str], dict]:
        """Mutual information based selection."""
        if not SKLEARN_AVAILABLE:
            return X.columns[:k].tolist(), {}
        
        X_clean = X.fillna(0)
        y_clean = pd.Series(y).fillna(0) if hasattr(y, 'fillna') else np.nan_to_num(y)
        
        is_classification = len(np.unique(y_clean)) < 20
        mi_func = mutual_info_classif if is_classification else mutual_info_regression
        
        mi_scores = mi_func(X_clean, y_clean, random_state=42)
        scores = dict(zip(X.columns, mi_scores))
        
        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        selected = [f[0] for f in top_k]
        
        return selected, scores


class TextPreprocessorNode(NodeRuntime):
    """Preprocess text columns."""
    
    def run(self, inputs: dict[str, Any]) -> NodeResult:
        data = inputs.get("Chunk")
        if data is None:
            return NodeResult(outputs={}, success=False, error_message="No data")
        
        df = ensure_dataframe(data).copy()
        
        column = self.get_option("Column", "text")
        lowercase = self.get_option("Lowercase", True)
        remove_punctuation = self.get_option("Remove Punctuation", True)
        remove_stopwords = self.get_option("Remove Stopwords", True)
        stemming = self.get_option("Stemming", False)
        lemmatization = self.get_option("Lemmatization", True)
        
        if column not in df.columns:
            return NodeResult(outputs={"Processed Text": df}, metadata={"message": f"Column '{column}' not found"})
        
        import re
        import string
        
        # Basic stopwords
        STOPWORDS = {"the", "a", "an", "is", "it", "to", "of", "and", "in", "for", "on", "with", "at", "by", "from"}
        
        def process_text(text):
            if pd.isna(text):
                return ""
            text = str(text)
            
            if lowercase:
                text = text.lower()
            
            if remove_punctuation:
                text = text.translate(str.maketrans("", "", string.punctuation))
            
            if remove_stopwords:
                words = text.split()
                words = [w for w in words if w.lower() not in STOPWORDS]
                text = " ".join(words)
            
            # Stemming and lemmatization would require nltk/spacy
            # Basic implementation here
            
            return text.strip()
        
        df[column] = df[column].apply(process_text)
        
        return NodeResult(outputs={"Processed Text": df})
