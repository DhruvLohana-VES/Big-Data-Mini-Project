"""
Classical ML Baseline Models:
1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
Evaluates Sentiment and Toxicity across Gold-Only, LLM-Only, and Mixed Regimes.
"""

import os
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class ClassicalBaselines:
    """
    Implements TF-IDF + Logistic Regression and TF-IDF + SVM baselines.
    """

    def __init__(self):
        self.vectorizer_params = {
            "ngram_range": (1, 3),
            "analyzer": "word",
            "max_features": 2500,
            "sublinear_tf": True
        }

    def get_pipeline(self, model_type: str = "logistic_regression") -> Pipeline:
        """Construct TF-IDF + Classifier pipeline."""
        if model_type == "logistic_regression":
            clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        elif model_type == "svm":
            clf = LinearSVC(C=1.0, random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return Pipeline([
            ("tfidf", TfidfVectorizer(**self.vectorizer_params)),
            ("clf", clf)
        ])

    def train_and_evaluate_regimes(
        self,
        gold_df: pd.DataFrame,
        llm_df: pd.DataFrame,
        task: str = "sentiment",
        model_type: str = "logistic_regression"
    ) -> List[Dict[str, Any]]:
        """
        Train and evaluate the baseline across:
        1. Gold Only
        2. LLM Only
        3. Mixed (Gold + LLM)
        Evaluates on held-out Gold test set.
        """
        # Prepare evaluation Gold set (20% holdout for testing)
        n_gold = len(gold_df)
        split_idx = int(n_gold * 0.75)
        
        train_gold = gold_df.iloc[:split_idx]
        test_gold = gold_df.iloc[split_idx:]
        
        # Prepare LLM training set
        llm_train = llm_df.copy()
        llm_label_col = f"llm_{task}" if f"llm_{task}" in llm_train.columns else task

        regimes = {
            "gold_only": {
                "X": train_gold["text"].tolist(),
                "y": train_gold[task].astype(str).str.lower().tolist()
            },
            "llm_only": {
                "X": llm_train["text"].tolist(),
                "y": llm_train[llm_label_col].astype(str).str.lower().tolist()
            },
            "mixed": {
                "X": train_gold["text"].tolist() + llm_train["text"].tolist(),
                "y": train_gold[task].astype(str).str.lower().tolist() + llm_train[llm_label_col].astype(str).str.lower().tolist()
            }
        }

        X_test = test_gold["text"].tolist()
        y_test = test_gold[task].astype(str).str.lower().tolist()

        results = []
        for regime_name, regime_data in regimes.items():
            pipe = self.get_pipeline(model_type=model_type)
            pipe.fit(regime_data["X"], regime_data["y"])
            
            y_pred = pipe.predict(X_test)
            
            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

            results.append({
                "model": f"TF-IDF + {'Logistic Regression' if model_type == 'logistic_regression' else 'Linear SVM'}",
                "model_category": "Classical Baseline",
                "training_regime": regime_name,
                "task": task,
                "sentiment_macro_f1": round(f1, 4) if task == "sentiment" else None,
                "toxicity_macro_f1": round(f1, 4) if task == "toxicity" else None,
                "macro_f1": round(f1, 4),
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4)
            })

        return results


def train_and_eval_baselines(gold_path: str = "data/gold/gold_master.csv", llm_path: str = "data/llm_annotations/reasoning_aware.csv") -> pd.DataFrame:
    """Train all classical baselines across all tasks and regimes."""
    gold_df = pd.read_csv(gold_path)
    if os.path.exists(llm_path):
        llm_df = pd.read_csv(llm_path)
    else:
        llm_df = gold_df.copy()
        llm_df["llm_sentiment"] = llm_df["sentiment"]
        llm_df["llm_toxicity"] = llm_df["toxicity"]

    baselines = ClassicalBaselines()
    all_res = []

    for task in ["sentiment", "toxicity"]:
        for model_type in ["logistic_regression", "svm"]:
            res = baselines.train_and_evaluate_regimes(gold_df, llm_df, task=task, model_type=model_type)
            all_res.extend(res)

    return pd.DataFrame(all_res)


if __name__ == "__main__":
    df_res = train_and_eval_baselines()
    print("=== Classical ML Baselines Benchmark ===")
    print(df_res[["model", "training_regime", "task", "macro_f1", "accuracy"]].to_string(index=False))
