from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class LinkResult:
    entity_id: str
    score: float
    name: str
    type: str


class EntityLinker:
    def __init__(self) -> None:
        self.kb: List[Dict] = []
        self.name_texts: List[str] = []
        self.desc_texts: List[str] = []
        self.name_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
        self.desc_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
        self.name_matrix = None
        self.desc_matrix = None

    def fit(self, kb_rows: List[Dict]) -> None:
        self.kb = kb_rows
        self.name_texts = []
        self.desc_texts = []

        for row in kb_rows:
            aliases = row.get("aliases", [])
            name_blob = " ".join([row.get("name", "")] + aliases)
            desc_blob = f"{row.get('name', '')} {row.get('description', '')}"
            self.name_texts.append(name_blob)
            self.desc_texts.append(desc_blob)

        self.name_matrix = self.name_vectorizer.fit_transform(self.name_texts)
        self.desc_matrix = self.desc_vectorizer.fit_transform(self.desc_texts)

    def _surface_score(self, mention: str, kb_row: Dict) -> float:
        candidates = [kb_row.get("name", "")] + kb_row.get("aliases", [])
        scores = [SequenceMatcher(None, mention, c).ratio() for c in candidates if c]
        return max(scores) if scores else 0.0

    def generate_candidates(self, mention: str, top_k: int = 5) -> List[int]:
        if self.name_matrix is None:
            raise RuntimeError("Linker is not trained.")

        q = self.name_vectorizer.transform([mention])
        sim = cosine_similarity(q, self.name_matrix).flatten()
        idx = sim.argsort()[::-1][:top_k]
        return idx.tolist()

    def disambiguate(self, mention: str, context: str, top_k: int = 5) -> LinkResult | None:
        if not self.kb:
            raise RuntimeError("Linker is not trained.")

        cand_ids = self.generate_candidates(mention, top_k=top_k)
        if not cand_ids:
            return None

        q_ctx = self.desc_vectorizer.transform([context])
        ctx_scores = cosine_similarity(q_ctx, self.desc_matrix).flatten()

        best = None
        for i in cand_ids:
            row = self.kb[i]
            surface = self._surface_score(mention, row)
            context_score = float(ctx_scores[i])
            score = 0.65 * surface + 0.35 * context_score
            if best is None or score > best.score:
                best = LinkResult(
                    entity_id=row["entity_id"],
                    score=score,
                    name=row.get("name", ""),
                    type=row.get("type", ""),
                )
        return best

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "kb": self.kb,
                "name_vectorizer": self.name_vectorizer,
                "desc_vectorizer": self.desc_vectorizer,
                "name_matrix": self.name_matrix,
                "desc_matrix": self.desc_matrix,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "EntityLinker":
        obj = joblib.load(path)
        inst = cls()
        inst.kb = obj["kb"]
        inst.name_vectorizer = obj["name_vectorizer"]
        inst.desc_vectorizer = obj["desc_vectorizer"]
        inst.name_matrix = obj["name_matrix"]
        inst.desc_matrix = obj["desc_matrix"]
        return inst
