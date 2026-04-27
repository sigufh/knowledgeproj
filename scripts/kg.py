#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kg_pipeline.el.linker import EntityLinker
from kg_pipeline.ner.crf_baseline import CRFNamedEntityRecognizer
from kg_pipeline.pipeline import (
    aggregate_disambiguated_entities,
    build_pipeline_payload,
    build_runtime_lexicon,
    export_kg_artifacts,
    extract_candidate_entities,
    link_entities,
    write_csv_rows,
)
from kg_pipeline.relation.classifier import RelationClassifier


EXTRACT_FIELDS = ["entity", "entity_type", "start", "end", "source_file", "source"]
MENTION_FIELDS = [
    "mention",
    "entity_type",
    "start",
    "end",
    "source_file",
    "entity_id",
    "kb_name",
    "kb_type",
    "link_score",
    "normalized_entity",
    "disambiguation_confidence",
    "disambiguation_method",
    "source",
]
ENTITY_FIELDS = [
    "main_entity",
    "entity_type",
    "mention_count",
    "mentions",
    "source_file",
    "normalized_entity",
    "aliases",
    "section_titles",
    "merged_entities",
    "disambiguation_confidence",
    "disambiguation_method",
    "disambiguation_basis",
]
SUBMISSION_PRIORITY = {
    "R_BORN_IN": 10,
    "R_FATHER": 9,
    "R_MOTHER": 9,
    "R_STUDIED_AT": 8,
    "R_GRADUATED_FROM": 8,
    "R_WORKED_FOR": 7,
    "R_WORKED_ON": 7,
    "R_CRACKED": 7,
    "R_WROTE": 6,
    "R_PUBLISHED": 6,
    "R_PROPOSED": 6,
    "R_AWARDED": 5,
    "R_REPUTED_AS": 5,
    "R_PARDONED": 5,
    "R_APOLOGIZED_TO": 4,
    "R_PERSECUTED": 4,
    "R_CONVICTED": 4,
}


def _read_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_paths(text_path: Path, out_dir: Path) -> Dict[str, Path]:
    stem = text_path.stem
    extract_dir = out_dir / "entity_extraction"
    disamb_dir = out_dir / "entity_disambiguation"
    kg_dir = out_dir / "knowledge_graph"
    fig_dir = out_dir / "figures"
    return {
        "lexicon_jsonl": extract_dir / f"{stem}_lexicon.jsonl",
        "extract_csv": extract_dir / f"{stem}.csv",
        "mentions_csv": disamb_dir / f"{stem}_mentions.csv",
        "entities_csv": disamb_dir / f"{stem}.csv",
        "pipeline_json": kg_dir / f"{stem}_pipeline.json",
        "relation_lexicon_json": kg_dir / f"{stem}_relation_lexicon.json",
        "core_json": kg_dir / f"{stem}_core.json",
        "triples_csv": kg_dir / f"{stem}_triples.csv",
        "nodes_csv": kg_dir / f"{stem}_nodes.csv",
        "figure_png": fig_dir / "pipeline_relation_graph.png",
        "submission_json": kg_dir / f"{stem}_submission_core.json",
        "submission_csv": kg_dir / f"{stem}_submission_core.csv",
    }


def run_extract(text_path: Path, ner_model: str, linker_model: str, output_csv: Path, lexicon_jsonl: Path | None = None) -> Dict[str, object]:
    text = text_path.read_text(encoding="utf-8")
    ner = CRFNamedEntityRecognizer.load(ner_model)
    linker = EntityLinker.load(linker_model)
    lexicon = build_runtime_lexicon(text=text, linker=linker)
    if lexicon_jsonl is not None:
        lexicon_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with lexicon_jsonl.open("w", encoding="utf-8") as f:
            for row in lexicon:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = extract_candidate_entities(text=text, ner=ner, linker=linker, source_file=text_path.name, lexicon=lexicon)
    write_csv_rows(path=output_csv, rows=rows, fieldnames=EXTRACT_FIELDS)
    return {"rows": rows, "text": text, "lexicon": lexicon}


def run_disambiguate(
    text_path: Path,
    linker_model: str,
    extract_csv: Path,
    mentions_csv: Path,
    entities_csv: Path,
    min_link_score: float,
) -> Dict[str, object]:
    text = text_path.read_text(encoding="utf-8")
    extracted = _read_csv(extract_csv)
    linker = EntityLinker.load(linker_model)
    mentions = link_entities(
        text=text,
        extracted_rows=extracted,
        linker=linker,
        min_link_score=min_link_score,
        source_file=text_path.name,
    )
    write_csv_rows(path=mentions_csv, rows=mentions, fieldnames=MENTION_FIELDS)
    entities = aggregate_disambiguated_entities(text=text, linked_rows=mentions, source_file=text_path.name)
    write_csv_rows(path=entities_csv, rows=entities, fieldnames=ENTITY_FIELDS)
    return {"mentions": mentions, "entities": entities, "text": text}


def run_build(
    text_path: Path,
    mentions_csv: Path,
    pipeline_json: Path,
    relation_lexicon_json: Path,
    core_json: Path,
    triples_csv: Path,
    nodes_csv: Path,
    re_model: str,
    relation_strategy: str,
    relation_label_lang: str,
    require_linked_entity: bool,
    max_pair_gap: int,
    pair_neighbors: int,
) -> Dict:
    text = text_path.read_text(encoding="utf-8")
    mentions = _read_csv(mentions_csv)
    rel_model = RelationClassifier.load(re_model) if re_model else None
    pipeline = build_pipeline_payload(
        text=text,
        linked_mentions=mentions,
        re_model=rel_model,
        relation_strategy=relation_strategy,
        relation_label_lang=relation_label_lang,
        require_linked_entity=require_linked_entity,
        max_pair_gap=max_pair_gap,
        pair_neighbors=pair_neighbors,
    )
    pipeline_json.parent.mkdir(parents=True, exist_ok=True)
    pipeline_json.write_text(json.dumps(pipeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    relation_lexicon_json.write_text(
        json.dumps(pipeline.get("relation_lexicon", []), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    export_kg_artifacts(pipeline=pipeline, core_json=core_json, triples_csv=triples_csv, nodes_csv=nodes_csv)
    return pipeline


def _build_final_report(pipeline: Dict, pipeline_path: Path, core_json: Path, triples_csv: Path, nodes_csv: Path) -> str:
    entities = pipeline.get("entities", [])
    rels = pipeline.get("relations", [])
    figure_path = pipeline_path.parents[1] / "figures" / "pipeline_relation_graph.png"
    core = _load_json(core_json) if core_json.exists() else {"nodes": []}
    core_nodes = core.get("nodes", [])
    rel_dist = Counter(r.get("label", r.get("relation_id", "")) for r in rels)
    preferred_order = [
        "R_BORN_IN",
        "R_STUDIED_AT",
        "R_GRADUATED_FROM",
        "R_WORKED_FOR",
        "R_CRACKED",
        "R_PUBLISHED",
        "R_PARDONED",
    ]
    top_triples = []
    seen = set()
    for rel_id in preferred_order:
        for row in rels:
            key = (row["head"].get("text"), row.get("relation_id"), row["tail"].get("text"))
            if row.get("relation_id") == rel_id and key not in seen:
                top_triples.append(row)
                seen.add(key)
                break
    for row in rels:
        if len(top_triples) >= 6:
            break
        key = (row["head"].get("text"), row.get("relation_id"), row["tail"].get("text"))
        if key not in seen:
            top_triples.append(row)
            seen.add(key)

    lines = [
        "# 图灵文本知识图谱实验报告（最终版）",
        "",
        "## 1. 数据与任务",
        "- 输入文本：`data/sample/input.txt`（图灵传记长文本）",
        "- 任务通路：实体抽取 -> 实体链接 -> 关系抽取 -> 图谱构建",
        "- 关键原则：训练域与抽取域对齐（同源文本构建训练集）",
        "",
        "## 2. 本次模型与产物",
        "- NER模型：`artifacts/ner_crf_turing.joblib`",
        "- 链接模型：`artifacts/linker_turing.joblib`",
        "- 关系模型：`artifacts/re_clf_turing.joblib`（本次主用 open 规则抽取）",
        f"- 图谱JSON：`{pipeline_path.as_posix()}`",
        f"- 核心图谱JSON：`{core_json.as_posix()}`",
        f"- 三元组CSV：`{triples_csv.as_posix()}`",
        f"- 节点CSV：`{nodes_csv.as_posix()}`",
        f"- 可视化：`{figure_path.as_posix()}`",
        "",
        "## 3. 结果统计",
        f"- 抽取实体数：{len(entities)}",
        f"- 抽取关系数：{len(rels)}",
        f"- 核心节点数（参与关系）：{len(core_nodes)}",
        f"- 关系类型分布：{dict(rel_dist)}",
        "",
        "## 4. 代表性三元组",
    ]
    for row in top_triples:
        lines.append(f"- ({row['head'].get('text','')}, {row.get('label', row.get('relation_id',''))}, {row['tail'].get('text','')})")
    lines.extend(
        [
            "",
            "## 5. 质量说明",
            "- 已抑制自环关系与明显误触发（如“要求…赦免”导致的伪关系）。",
            "- 当前关系以事件触发词高精规则为主，已补上出生、就读、毕业等关键事实链，并覆盖普林斯顿求学阶段。",
            "- 仍有可继续补强的点：部分英文头衔实体类型、长句事件关系覆盖、少量未链接尾实体。",
            "- 若用于最终提交，建议补充少量人工校验三元组。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def export_final_artifacts(pipeline_path: Path, core_json: Path, triples_csv: Path, nodes_csv: Path, report_md: Path) -> None:
    pipeline = _load_json(pipeline_path)
    export_kg_artifacts(pipeline=pipeline, core_json=core_json, triples_csv=triples_csv, nodes_csv=nodes_csv)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(_build_final_report(pipeline, pipeline_path, core_json, triples_csv, nodes_csv), encoding="utf-8-sig")


def _relation_rank(row: Dict) -> tuple:
    rel_id = row.get("relation_id", "")
    head = row.get("head", {}).get("text", "")
    tail = row.get("tail", {}).get("text", "")
    return (
        SUBMISSION_PRIORITY.get(rel_id, 0),
        float(row.get("confidence", 0.0)),
        1 if row.get("head", {}).get("entity_id") else 0,
        1 if row.get("tail", {}).get("entity_id") else 0,
        -len(head),
        -len(tail),
    )


def export_submission_view(pipeline_path: Path, out_json: Path, out_csv: Path, out_md: Path, max_relations: int) -> None:
    pipe = _load_json(pipeline_path)
    rels = sorted(pipe.get("relations", []), key=_relation_rank, reverse=True)

    kept = []
    seen = set()
    rel_type_count: Dict[str, int] = {}
    for row in rels:
        rel_id = row.get("relation_id", "")
        key = (
            row.get("head", {}).get("entity_id") or row.get("head", {}).get("text"),
            rel_id,
            row.get("tail", {}).get("entity_id") or row.get("tail", {}).get("text"),
        )
        if key in seen or rel_type_count.get(rel_id, 0) >= 3:
            continue
        kept.append(row)
        seen.add(key)
        rel_type_count[rel_id] = rel_type_count.get(rel_id, 0) + 1
        if len(kept) >= max_relations:
            break

    used = set()
    for row in kept:
        used.add(row["head"].get("entity_id") or f"MENTION::{row['head'].get('text','')}")
        used.add(row["tail"].get("entity_id") or f"MENTION::{row['tail'].get('text','')}")
    nodes = [n for n in pipe.get("entity_nodes", []) if n.get("node_id") in used]
    out_obj = {
        "text": pipe.get("text", ""),
        "node_count": len(nodes),
        "edge_count": len(kept),
        "nodes": nodes,
        "relations": kept,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["head", "relation", "tail", "confidence", "source"])
        writer.writeheader()
        for row in kept:
            writer.writerow(
                {
                    "head": row["head"].get("text", ""),
                    "relation": row.get("label", row.get("relation_id", "")),
                    "tail": row["tail"].get("text", ""),
                    "confidence": row.get("confidence", ""),
                    "source": row.get("source", ""),
                }
            )
    lines = ["# 提交版核心图谱", "", f"- 关系数：{len(kept)}", f"- 节点数：{len(nodes)}", ""]
    for row in kept:
        lines.append(f"- ({row['head'].get('text','')}, {row.get('label', row.get('relation_id',''))}, {row['tail'].get('text','')})")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def visualize_pipeline_graph(pipeline_path: Path, out_dir: Path) -> None:
    import visualize_results as viz

    viz._configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _load_json(pipeline_path)
    viz.plot_relation_graph(pipeline, out_dir / "pipeline_relation_graph.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified staged KG pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Stage 1: extract candidate entities to CSV")
    p_extract.add_argument("--text-file", required=True)
    p_extract.add_argument("--ner-model", required=True)
    p_extract.add_argument("--linker-model", required=True)
    p_extract.add_argument("--out-dir", default="outputs/reference_style_chain")
    p_extract.add_argument("--output-csv", default="")
    p_extract.add_argument("--lexicon-jsonl", default="")

    p_disamb = sub.add_parser("disambiguate", help="Stage 2: link and aggregate extracted entities")
    p_disamb.add_argument("--text-file", required=True)
    p_disamb.add_argument("--linker-model", required=True)
    p_disamb.add_argument("--out-dir", default="outputs/reference_style_chain")
    p_disamb.add_argument("--extract-csv", default="")
    p_disamb.add_argument("--mentions-csv", default="")
    p_disamb.add_argument("--entities-csv", default="")
    p_disamb.add_argument("--min-link-score", type=float, default=0.45)

    p_build = sub.add_parser("build", help="Stage 3: build KG from linked mentions")
    p_build.add_argument("--text-file", required=True)
    p_build.add_argument("--out-dir", default="outputs/reference_style_chain")
    p_build.add_argument("--mentions-csv", default="")
    p_build.add_argument("--pipeline-json", default="")
    p_build.add_argument("--relation-lexicon-json", default="")
    p_build.add_argument("--core-json", default="")
    p_build.add_argument("--triples-csv", default="")
    p_build.add_argument("--nodes-csv", default="")
    p_build.add_argument("--re-model", default="")
    p_build.add_argument("--relation-strategy", choices=["open", "hybrid", "classifier"], default="open")
    p_build.add_argument("--relation-label-lang", choices=["en", "zh", "both"], default="zh")
    p_build.add_argument("--require-linked-entity", action="store_true")
    p_build.add_argument("--max-pair-gap", type=int, default=64)
    p_build.add_argument("--pair-neighbors", type=int, default=6)

    p_export = sub.add_parser("export", help="Stage 4: export report and submission artifacts")
    p_export.add_argument("--text-file", required=True)
    p_export.add_argument("--out-dir", default="outputs/reference_style_chain")
    p_export.add_argument("--pipeline-json", default="")
    p_export.add_argument("--core-json", default="")
    p_export.add_argument("--triples-csv", default="")
    p_export.add_argument("--nodes-csv", default="")
    p_export.add_argument("--report-md", default="reports/final_turing_kg_report.md")
    p_export.add_argument("--submission-json", default="")
    p_export.add_argument("--submission-csv", default="")
    p_export.add_argument("--submission-md", default="reports/submission_core_kg.md")
    p_export.add_argument("--max-relations", type=int, default=20)

    p_vis = sub.add_parser("visualize", help="Stage 5: visualize relation graph")
    p_vis.add_argument("--text-file", required=True)
    p_vis.add_argument("--out-dir", default="outputs/reference_style_chain")
    p_vis.add_argument("--pipeline-json", default="")
    p_vis.add_argument("--figure-dir", default="")

    p_all = sub.add_parser("all", help="Run the full staged pipeline end to end")
    p_all.add_argument("--text-file", required=True)
    p_all.add_argument("--ner-model", required=True)
    p_all.add_argument("--linker-model", required=True)
    p_all.add_argument("--re-model", default="")
    p_all.add_argument("--out-dir", default="outputs/reference_style_chain")
    p_all.add_argument("--min-link-score", type=float, default=0.45)
    p_all.add_argument("--relation-strategy", choices=["open", "hybrid", "classifier"], default="open")
    p_all.add_argument("--relation-label-lang", choices=["en", "zh", "both"], default="zh")
    p_all.add_argument("--require-linked-entity", action="store_true")
    p_all.add_argument("--max-pair-gap", type=int, default=64)
    p_all.add_argument("--pair-neighbors", type=int, default=6)
    p_all.add_argument("--report-md", default="reports/final_turing_kg_report.md")
    p_all.add_argument("--submission-md", default="reports/submission_core_kg.md")
    p_all.add_argument("--max-relations", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    text_path = Path(args.text_file)
    stage_paths = _stage_paths(text_path, Path(args.out_dir))

    if args.command == "extract":
        output_csv = Path(args.output_csv) if args.output_csv else stage_paths["extract_csv"]
        lexicon_jsonl = Path(args.lexicon_jsonl) if args.lexicon_jsonl else stage_paths["lexicon_jsonl"]
        res = run_extract(text_path, args.ner_model, args.linker_model, output_csv, lexicon_jsonl)
        print(f"extract_csv={output_csv}")
        print(f"lexicon_jsonl={lexicon_jsonl}")
        print(f"entity_rows={len(res['rows'])}")
        return

    if args.command == "disambiguate":
        extract_csv = Path(args.extract_csv) if args.extract_csv else stage_paths["extract_csv"]
        mentions_csv = Path(args.mentions_csv) if args.mentions_csv else stage_paths["mentions_csv"]
        entities_csv = Path(args.entities_csv) if args.entities_csv else stage_paths["entities_csv"]
        res = run_disambiguate(text_path, args.linker_model, extract_csv, mentions_csv, entities_csv, args.min_link_score)
        print(f"mentions_csv={mentions_csv}")
        print(f"entities_csv={entities_csv}")
        print(f"mention_rows={len(res['mentions'])}")
        print(f"entity_groups={len(res['entities'])}")
        return

    if args.command == "build":
        mentions_csv = Path(args.mentions_csv) if args.mentions_csv else stage_paths["mentions_csv"]
        pipeline_json = Path(args.pipeline_json) if args.pipeline_json else stage_paths["pipeline_json"]
        relation_lexicon_json = Path(args.relation_lexicon_json) if args.relation_lexicon_json else stage_paths["relation_lexicon_json"]
        core_json = Path(args.core_json) if args.core_json else stage_paths["core_json"]
        triples_csv = Path(args.triples_csv) if args.triples_csv else stage_paths["triples_csv"]
        nodes_csv = Path(args.nodes_csv) if args.nodes_csv else stage_paths["nodes_csv"]
        pipeline = run_build(
            text_path,
            mentions_csv,
            pipeline_json,
            relation_lexicon_json,
            core_json,
            triples_csv,
            nodes_csv,
            args.re_model,
            args.relation_strategy,
            args.relation_label_lang,
            args.require_linked_entity,
            args.max_pair_gap,
            args.pair_neighbors,
        )
        print(f"pipeline_json={pipeline_json}")
        print(f"relation_lexicon_json={relation_lexicon_json}")
        print(f"core_json={core_json}")
        print(f"triples_csv={triples_csv}")
        print(f"nodes_csv={nodes_csv}")
        print(f"relation_rows={len(pipeline.get('relations', []))}")
        return

    if args.command == "export":
        pipeline_json = Path(args.pipeline_json) if args.pipeline_json else stage_paths["pipeline_json"]
        core_json = Path(args.core_json) if args.core_json else stage_paths["core_json"]
        triples_csv = Path(args.triples_csv) if args.triples_csv else stage_paths["triples_csv"]
        nodes_csv = Path(args.nodes_csv) if args.nodes_csv else stage_paths["nodes_csv"]
        submission_json = Path(args.submission_json) if args.submission_json else stage_paths["submission_json"]
        submission_csv = Path(args.submission_csv) if args.submission_csv else stage_paths["submission_csv"]
        export_final_artifacts(pipeline_json, core_json, triples_csv, nodes_csv, Path(args.report_md))
        export_submission_view(pipeline_json, submission_json, submission_csv, Path(args.submission_md), args.max_relations)
        print(f"report_md={args.report_md}")
        print(f"submission_json={submission_json}")
        print(f"submission_csv={submission_csv}")
        print(f"submission_md={args.submission_md}")
        return

    if args.command == "visualize":
        pipeline_json = Path(args.pipeline_json) if args.pipeline_json else stage_paths["pipeline_json"]
        figure_dir = Path(args.figure_dir) if args.figure_dir else Path(args.out_dir) / "figures"
        visualize_pipeline_graph(pipeline_json, figure_dir)
        print(f"figure_png={figure_dir / 'pipeline_relation_graph.png'}")
        return

    if args.command == "all":
        extract_res = run_extract(text_path, args.ner_model, args.linker_model, stage_paths["extract_csv"], stage_paths["lexicon_jsonl"])
        disamb_res = run_disambiguate(
            text_path,
            args.linker_model,
            stage_paths["extract_csv"],
            stage_paths["mentions_csv"],
            stage_paths["entities_csv"],
            args.min_link_score,
        )
        pipeline = run_build(
            text_path,
            stage_paths["mentions_csv"],
            stage_paths["pipeline_json"],
            stage_paths["relation_lexicon_json"],
            stage_paths["core_json"],
            stage_paths["triples_csv"],
            stage_paths["nodes_csv"],
            args.re_model,
            args.relation_strategy,
            args.relation_label_lang,
            args.require_linked_entity,
            args.max_pair_gap,
            args.pair_neighbors,
        )
        visualize_pipeline_graph(stage_paths["pipeline_json"], Path(args.out_dir) / "figures")
        export_final_artifacts(stage_paths["pipeline_json"], stage_paths["core_json"], stage_paths["triples_csv"], stage_paths["nodes_csv"], Path(args.report_md))
        export_submission_view(
            stage_paths["pipeline_json"],
            stage_paths["submission_json"],
            stage_paths["submission_csv"],
            Path(args.submission_md),
            args.max_relations,
        )
        print(f"extract_csv={stage_paths['extract_csv']}")
        print(f"lexicon_jsonl={stage_paths['lexicon_jsonl']}")
        print(f"mentions_csv={stage_paths['mentions_csv']}")
        print(f"entities_csv={stage_paths['entities_csv']}")
        print(f"pipeline_json={stage_paths['pipeline_json']}")
        print(f"relation_lexicon_json={stage_paths['relation_lexicon_json']}")
        print(f"core_json={stage_paths['core_json']}")
        print(f"triples_csv={stage_paths['triples_csv']}")
        print(f"nodes_csv={stage_paths['nodes_csv']}")
        print(f"figure_png={(Path(args.out_dir) / 'figures' / 'pipeline_relation_graph.png')}")
        print(f"report_md={args.report_md}")
        print(f"submission_md={args.submission_md}")
        print(f"entity_rows={len(extract_res['rows'])} disambiguated_groups={len(disamb_res['entities'])} relation_rows={len(pipeline.get('relations', []))}")


if __name__ == "__main__":
    main()
