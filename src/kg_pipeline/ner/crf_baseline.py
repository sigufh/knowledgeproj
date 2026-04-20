from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import joblib
import sklearn_crfsuite

from kg_pipeline.utils.spans import bio_to_entities, entities_to_bio, score_entities


@dataclass
class CRFConfig:
    c1: float = 0.1
    c2: float = 0.1
    max_iterations: int = 100
    all_possible_transitions: bool = True
    use_context: bool = True
    use_bigram: bool = True


class CRFNamedEntityRecognizer:
    def __init__(self, config: CRFConfig | None = None) -> None:
        self.config = config or CRFConfig()
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=self.config.c1,
            c2=self.config.c2,
            max_iterations=self.config.max_iterations,
            all_possible_transitions=self.config.all_possible_transitions,
        )

    def _char_features(self, chars: List[str], i: int) -> Dict[str, str]:
        ch = chars[i]
        features: Dict[str, str] = {
            "bias": "1.0",
            "ch": ch,
            "is_digit": str(ch.isdigit()),
            "is_alpha": str(ch.isalpha()),
            "is_upper": str(ch.isupper()),
            "is_lower": str(ch.islower()),
        }

        if i > 0:
            prev_ch = chars[i - 1]
            features.update(
                {
                    "-1:ch": prev_ch,
                    "-1:is_digit": str(prev_ch.isdigit()),
                    "-1:is_alpha": str(prev_ch.isalpha()),
                }
            )
            if self.config.use_bigram:
                features["bigram-1"] = prev_ch + ch
        else:
            features["BOS"] = "1"

        if i < len(chars) - 1:
            next_ch = chars[i + 1]
            features.update(
                {
                    "+1:ch": next_ch,
                    "+1:is_digit": str(next_ch.isdigit()),
                    "+1:is_alpha": str(next_ch.isalpha()),
                }
            )
            if self.config.use_bigram:
                features["bigram+1"] = ch + next_ch
        else:
            features["EOS"] = "1"

        if self.config.use_context:
            if i > 1:
                features["-2:ch"] = chars[i - 2]
            if i < len(chars) - 2:
                features["+2:ch"] = chars[i + 2]
        return features

    def _sent_features(self, text: str) -> List[Dict[str, str]]:
        chars = list(text)
        return [self._char_features(chars, i) for i in range(len(chars))]

    def fit(self, samples: List[Dict]) -> None:
        X = [self._sent_features(s["text"]) for s in samples]
        y = [entities_to_bio(s["text"], s.get("entities", [])) for s in samples]
        self.model.fit(X, y)

    def predict_tags(self, text: str) -> List[str]:
        X = [self._sent_features(text)]
        return self.model.predict(X)[0]

    def predict_entities(self, text: str) -> List[Dict]:
        tags = self.predict_tags(text)
        return bio_to_entities(text, tags)

    def save(self, path: str) -> None:
        joblib.dump({"config": self.config, "model": self.model}, path)

    @classmethod
    def load(cls, path: str) -> "CRFNamedEntityRecognizer":
        obj = joblib.load(path)
        inst = cls(config=obj["config"])
        inst.model = obj["model"]
        return inst


def evaluate_ner(model: CRFNamedEntityRecognizer, samples: List[Dict]) -> Dict[str, float]:
    gold = [s.get("entities", []) for s in samples]
    pred = [model.predict_entities(s["text"]) for s in samples]
    return score_entities(gold, pred)
