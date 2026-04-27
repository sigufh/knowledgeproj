#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download HuggingFace model locally")
    p.add_argument("--model-name", default="bert-base-chinese")
    p.add_argument("--out-dir", default="artifacts/models/bert-base-chinese")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    tok.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print(f"Saved model to {out_dir}")


if __name__ == "__main__":
    main()
