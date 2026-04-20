#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import itertools

from kg_pipeline.el.linker import EntityLinker
from kg_pipeline.ner.crf_baseline import CRFNamedEntityRecognizer
from kg_pipeline.relation.classifier import RelationClassifier
from kg_pipeline.utils.spans import split_sentences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run end-to-end extraction pipeline")
    p.add_argument("--text", required=True, help="Input raw text")
    p.add_argument("--ner-type", default="crf", choices=["crf", "bert_crf"])
    p.add_argument("--ner-model", required=True, help="Path to NER model")
    p.add_argument("--linker-model", required=True, help="Path to linker model")
    p.add_argument("--re-model", required=True, help="Path to relation model")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.ner_type == "crf":
        ner = CRFNamedEntityRecognizer.load(args.ner_model)
    else:
        from kg_pipeline.ner.bert_crf import BERTCRFNamedEntityRecognizer

        ner = BERTCRFNamedEntityRecognizer.load(args.ner_model)

    linker = EntityLinker.load(args.linker_model)
    rel = RelationClassifier.load(args.re_model)

    entities = ner.predict_entities(args.text)

    linked_entities = []
    for ent in entities:
        mention = ent["text"]
        link = linker.disambiguate(mention=mention, context=args.text, top_k=5)
        linked_entities.append(
            {
                **ent,
                "entity_id": link.entity_id if link else None,
                "kb_name": link.name if link else None,
                "link_score": link.score if link else 0.0,
            }
        )

    relations = []
    sent_spans = split_sentences(args.text)
    for s_start, s_end, sent_text in sent_spans:
        sent_ents = [
            x for x in linked_entities if x["start"] >= s_start and x["end"] <= s_end
        ]
        for e1, e2 in itertools.combinations(sent_ents, 2):
            h1 = {
                "start": e1["start"] - s_start,
                "end": e1["end"] - s_start,
                "label": e1["label"],
            }
            t1 = {
                "start": e2["start"] - s_start,
                "end": e2["end"] - s_start,
                "label": e2["label"],
            }
            l1, s1 = rel.predict_with_score(sent_text, h1, t1)

            h2 = t1
            t2 = h1
            l2, s2 = rel.predict_with_score(sent_text, h2, t2)

            cand = []
            if l1 != "NO_REL":
                cand.append((l1, s1, e1, e2))
            if l2 != "NO_REL":
                cand.append((l2, s2, e2, e1))
            if not cand:
                continue

            label, _, head_ent, tail_ent = max(cand, key=lambda x: x[1])
            relations.append(
                {
                    "head": {"text": head_ent["text"], "entity_id": head_ent.get("entity_id")},
                    "tail": {"text": tail_ent["text"], "entity_id": tail_ent.get("entity_id")},
                    "label": label,
                    "sentence": sent_text,
                }
            )

    output = {
        "text": args.text,
        "entities": linked_entities,
        "relations": relations,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
