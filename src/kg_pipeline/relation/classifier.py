from __future__ import annotations

from typing import Dict, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline


class RelationClassifier:
    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3))),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    @staticmethod
    def mark_entities(text: str, head: Dict, tail: Dict) -> str:
        marks = [
            (head["start"], "<H>"),
            (head["end"], "</H>"),
            (tail["start"], "<T>"),
            (tail["end"], "</T>"),
        ]
        marks.sort(key=lambda x: x[0], reverse=True)

        s = text
        for idx, token in marks:
            if 0 <= idx <= len(s):
                s = s[:idx] + token + s[idx:]
        return s

    def fit(self, samples: List[Dict]) -> None:
        X = [self.mark_entities(s["text"], s["head"], s["tail"]) for s in samples]
        y = [s["label"] for s in samples]
        self.pipeline.fit(X, y)

    def predict(self, text: str, head: Dict, tail: Dict) -> str:
        x = self.mark_entities(text, head, tail)
        return self.pipeline.predict([x])[0]

    def predict_with_score(self, text: str, head: Dict, tail: Dict) -> tuple[str, float]:
        x = self.mark_entities(text, head, tail)
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba([x])[0]
            idx = int(proba.argmax())
            label = self.pipeline.classes_[idx]
            return label, float(proba[idx])
        label = self.pipeline.predict([x])[0]
        return label, 1.0

    def evaluate(self, samples: List[Dict]) -> Dict[str, float]:
        X = [self.mark_entities(s["text"], s["head"], s["tail"]) for s in samples]
        y_true = [s["label"] for s in samples]
        y_pred = self.pipeline.predict(X)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        micro_f1 = f1_score(y_true, y_pred, average="micro")
        return {
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
            "report": classification_report(y_true, y_pred, zero_division=0),
        }

    def save(self, path: str) -> None:
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> "RelationClassifier":
        inst = cls()
        inst.pipeline = joblib.load(path)
        return inst
