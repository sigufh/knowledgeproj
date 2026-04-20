#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kg_pipeline.data.io import load_jsonl
from kg_pipeline.el.linker import EntityLinker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train entity linker")
    p.add_argument("--kb", required=True, help="KB jsonl path")
    p.add_argument("--model-out", required=True, help="Output model path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    kb_rows = load_jsonl(args.kb)

    linker = EntityLinker()
    linker.fit(kb_rows)

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    linker.save(str(out))
    print(f"Saved linker to {out}")


if __name__ == "__main__":
    main()
