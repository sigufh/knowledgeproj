#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kg_pipeline.data.io import load_jsonl
from kg_pipeline.ner.crf_baseline import CRFConfig, CRFNamedEntityRecognizer, evaluate_ner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CRF NER baseline")
    p.add_argument("--train", required=True, help="Path to NER train jsonl")
    p.add_argument("--dev", required=True, help="Path to NER dev jsonl")
    p.add_argument("--model-out", required=True, help="Output model path")
    p.add_argument("--no-context", action="store_true", help="Disable +-2 context features")
    p.add_argument("--no-bigram", action="store_true", help="Disable bigram features")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)

    cfg = CRFConfig(
        use_context=not args.no_context,
        use_bigram=not args.no_bigram,
    )
    model = CRFNamedEntityRecognizer(cfg)
    model.fit(train_rows)

    metrics = evaluate_ner(model, dev_rows)
    print("[CRF NER] dev metrics:", metrics)

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
