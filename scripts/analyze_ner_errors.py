#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_pipeline.data.io import load_jsonl
from kg_pipeline.ner.crf_baseline import CRFNamedEntityRecognizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze NER errors")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", default="artifacts/ner_errors.jsonl")
    return p.parse_args()


def _to_key_set(entities):
    return {(e["start"], e["end"], e["label"]) for e in entities}


def _from_key(text, key):
    start, end, label = key
    return {
        "start": start,
        "end": end,
        "label": label,
        "text": text[start:end],
    }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.data)
    model = CRFNamedEntityRecognizer.load(args.model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    errors = []
    for row in rows:
        text = row["text"]
        gold = row.get("entities", [])
        pred = model.predict_entities(text)
        gold_keys = _to_key_set(gold)
        pred_keys = _to_key_set(pred)
        if gold_keys == pred_keys:
            continue
        missed = sorted(gold_keys - pred_keys)
        spurious = sorted(pred_keys - gold_keys)
        errors.append(
            {
                "text": text,
                "gold": gold,
                "pred": pred,
                "missed": [_from_key(text, k) for k in missed],
                "spurious": [_from_key(text, k) for k in spurious],
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Error cases: {len(errors)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
