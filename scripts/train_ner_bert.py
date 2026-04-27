#!/usr/bin/env python3
from __future__ import annotations

import argparse

from kg_pipeline.data.io import load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BERT-CRF NER")
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="bert-base-chinese")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from kg_pipeline.ner.bert_crf import BERTCRFConfig, BERTCRFNamedEntityRecognizer
    except Exception as exc:
        raise SystemExit(
            "BERT-CRF依赖未就绪（通常是torch环境问题）。"
            f"请先修复环境后再运行。原始错误: {exc!r}"
        ) from exc

    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)

    cfg = BERTCRFConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
    )

    model = BERTCRFNamedEntityRecognizer(cfg)
    metrics = model.fit(train_rows, dev_rows)
    print("[BERT-CRF NER] dev metrics:", metrics)

    model.save(args.output_dir)
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
