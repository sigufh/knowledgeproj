#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kg_pipeline.data.io import load_jsonl
from kg_pipeline.relation.classifier import RelationClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train relation classifier")
    p.add_argument("--train", required=True, help="Path to relation train jsonl")
    p.add_argument("--dev", required=True, help="Path to relation dev jsonl")
    p.add_argument("--model-out", required=True, help="Output model path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)

    model = RelationClassifier()
    model.fit(train_rows)
    metrics = model.evaluate(dev_rows)

    print("[RE] dev macro_f1:", metrics["macro_f1"])
    print("[RE] dev micro_f1:", metrics["micro_f1"])
    print("[RE] report:\n", metrics["report"])

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Saved relation model to {out}")


if __name__ == "__main__":
    main()
