from __future__ import annotations

from typing import List, Dict, Tuple


def entities_to_bio(text: str, entities: List[Dict]) -> List[str]:
    tags = ["O"] * len(text)
    for ent in entities:
        start = int(ent["start"])
        end = int(ent["end"])
        label = ent["label"]
        if start < 0 or end > len(text) or start >= end:
            continue
        tags[start] = f"B-{label}"
        for idx in range(start + 1, end):
            tags[idx] = f"I-{label}"
    return tags


def bio_to_entities(text: str, tags: List[str]) -> List[Dict]:
    entities: List[Dict] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if not tag.startswith("B-"):
            i += 1
            continue
        label = tag[2:]
        start = i
        i += 1
        while i < len(tags) and tags[i] == f"I-{label}":
            i += 1
        end = i
        entities.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "text": text[start:end],
            }
        )
    return entities


def score_entities(gold: List[List[Dict]], pred: List[List[Dict]]) -> Dict[str, float]:
    gold_set = set()
    pred_set = set()

    for idx, row in enumerate(gold):
        for ent in row:
            gold_set.add((idx, ent["start"], ent["end"], ent["label"]))

    for idx, row in enumerate(pred):
        for ent in row:
            pred_set.add((idx, ent["start"], ent["end"], ent["label"]))

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def split_sentences(text: str) -> List[Tuple[int, int, str]]:
    boundaries = "。！？!?;；\n"
    spans: List[Tuple[int, int, str]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in boundaries:
            end = i + 1
            sent = text[start:end].strip()
            if sent:
                spans.append((start, end, sent))
            start = end
    if start < len(text):
        sent = text[start:].strip()
        if sent:
            spans.append((start, len(text), sent))
    return spans
