#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_pipeline.data.io import load_jsonl
from kg_pipeline.ner.crf_baseline import CRFConfig, CRFNamedEntityRecognizer, evaluate_ner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CRF NER ablation experiments")
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--output", default="artifacts/ner_ablation.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)

    settings = {
        "full": CRFConfig(use_context=True, use_bigram=True),
        "no_context": CRFConfig(use_context=False, use_bigram=True),
        "no_bigram": CRFConfig(use_context=True, use_bigram=False),
        "no_context_no_bigram": CRFConfig(use_context=False, use_bigram=False),
    }

    results = {}
    for name, cfg in settings.items():
        model = CRFNamedEntityRecognizer(cfg)
        model.fit(train_rows)
        metrics = evaluate_ner(model, dev_rows)
        results[name] = metrics
        print(f"[{name}] {metrics}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved ablation results to {out}")


if __name__ == "__main__":
    main()
