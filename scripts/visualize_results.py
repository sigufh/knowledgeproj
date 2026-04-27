#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from textwrap import wrap
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize NER/EL/RE outputs")
    p.add_argument("--ablation", help="Path to ner_ablation.json")
    p.add_argument("--errors", help="Path to ner_errors.jsonl")
    p.add_argument("--pipeline", help="Path to pipeline output json")
    p.add_argument("--out-dir", default="artifacts/figures", help="Output figure directory")
    return p.parse_args()


def _load_json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> List[Dict]:
    rows: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _configure_matplotlib() -> None:
    # Fallback font list for Chinese text on common platforms.
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_ablation(ablation: Dict[str, Dict[str, float]], output_path: Path) -> None:
    names = list(ablation.keys())
    f1 = [ablation[k].get("f1", 0.0) for k in names]
    precision = [ablation[k].get("precision", 0.0) for k in names]
    recall = [ablation[k].get("recall", 0.0) for k in names]

    x = list(range(len(names)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width for i in x], precision, width=width, label="Precision")
    ax.bar(x, recall, width=width, label="Recall")
    ax.bar([i + width for i in x], f1, width=width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("NER Ablation Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _overlap(a: Dict, b: Dict) -> bool:
    return not (a["end"] <= b["start"] or b["end"] <= a["start"])


def plot_error_analysis(error_rows: List[Dict], output_path: Path) -> None:
    missed_counter = Counter()
    spurious_counter = Counter()
    boundary_error = 0
    type_error = 0

    for row in error_rows:
        gold = row.get("gold", [])
        pred = row.get("pred", [])
        for g in gold:
            same_span = [p for p in pred if p["start"] == g["start"] and p["end"] == g["end"]]
            if any(p["label"] == g["label"] for p in same_span):
                continue
            if same_span:
                type_error += 1
            elif any(_overlap(g, p) and p["label"] == g["label"] for p in pred):
                boundary_error += 1
            missed_counter[g["label"]] += 1
        for p in row.get("spurious", []):
            spurious_counter[p["label"]] += 1

    labels = sorted(set(missed_counter) | set(spurious_counter))
    missed_vals = [missed_counter[l] for l in labels]
    spurious_vals = [spurious_counter[l] for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = list(range(len(labels)))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], missed_vals, width=width, label="Missed")
    ax1.bar([i + width / 2 for i in x], spurious_vals, width=width, label="Spurious")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels if labels else ["N/A"])
    ax1.set_title("NER Error by Label")
    ax1.legend()

    ax2.bar(["Boundary", "Type"], [boundary_error, type_error], color=["#1f77b4", "#ff7f0e"])
    ax2.set_title("NER Error Type Summary")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_relation_graph(pipeline: Dict, output_path: Path) -> None:
    entities = pipeline.get("entity_nodes", []) or pipeline.get("entities", [])
    relations = pipeline.get("relations", [])
    if not entities or not relations:
        return

    # Use only nodes that appear in at least one relation to avoid huge clutter.
    active_node_ids = set()
    for rel in relations:
        h = rel.get("head", {})
        t = rel.get("tail", {})
        hid = h.get("entity_id") or h.get("text")
        tid = t.get("entity_id") or t.get("text")
        if hid and tid and hid != tid:
            active_node_ids.add(hid)
            active_node_ids.add(tid)

    if not active_node_ids:
        return

    nodes = []
    node_id_to_idx = {}
    for e in entities:
        node_id = e.get("node_id") or e.get("entity_id") or e.get("text")
        if not node_id or node_id not in active_node_ids:
            continue
        node_id_to_idx[node_id] = len(nodes)
        nodes.append(
            {
                "node_id": node_id,
                "label": e.get("name") or e.get("text", ""),
                "type": e.get("type") or e.get("label", ""),
            }
        )

    if not nodes:
        return

    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n["node_id"], label=n["label"], node_type=n["type"])

    edge_labels = {}
    for rel in relations:
        h = rel.get("head", {})
        t = rel.get("tail", {})
        hid = h.get("entity_id") or h.get("text")
        tid = t.get("entity_id") or t.get("text")
        if hid not in node_id_to_idx or tid not in node_id_to_idx or hid == tid:
            continue
        g.add_edge(hid, tid)
        edge_labels[(hid, tid)] = rel.get("label", rel.get("relation_id", "REL"))

    if g.number_of_edges() == 0:
        return

    def chunked(items: List[str], size: int) -> List[List[str]]:
        return [items[i : i + size] for i in range(0, len(items), size)]

    def radial_layout(graph: nx.DiGraph, center: str) -> Dict[str, Tuple[float, float]]:
        undirected = graph.to_undirected()
        dist_map = nx.single_source_shortest_path_length(undirected, center)
        pos: Dict[str, Tuple[float, float]] = {center: (0.0, 0.0)}
        base_radius = 2.8
        ring_gap = 2.25

        for dist in sorted(d for d in set(dist_map.values()) if d > 0):
            shell_nodes = [n for n, d in dist_map.items() if d == dist]
            shell_nodes.sort(key=lambda n: (-graph.degree(n), graph.nodes[n]["label"]))
            per_ring = 6 if dist == 1 else 8
            groups = chunked(shell_nodes, per_ring)
            for group_idx, group in enumerate(groups):
                radius = base_radius + (dist - 1) * ring_gap * 2.1 + group_idx * 1.55
                angle_offset = dist * 0.46 + group_idx * 0.31
                for idx, node in enumerate(group):
                    angle = angle_offset + (2 * math.pi * idx / max(len(group), 1))
                    pos[node] = (radius * math.cos(angle), radius * math.sin(angle))

        remaining = sorted([n for n in graph.nodes() if n not in pos], key=lambda n: graph.nodes[n]["label"])
        extra_groups = chunked(remaining, 10)
        outer_start = base_radius + max(dist_map.values(), default=0) * ring_gap * 2.2 + 1.8
        for group_idx, group in enumerate(extra_groups):
            radius = outer_start + group_idx * 1.7
            angle_offset = 0.35 * (group_idx + 1)
            for idx, node in enumerate(group):
                angle = angle_offset + (2 * math.pi * idx / max(len(group), 1))
                pos[node] = (radius * math.cos(angle), radius * math.sin(angle))
        return pos

    hub = max(g.degree(), key=lambda item: item[1])[0]
    pos = radial_layout(g, hub)

    def wrapped_label(text: str, width: int = 10) -> str:
        return "\n".join(wrap(text, width=width, break_long_words=True, break_on_hyphens=False) or [""])

    label_pos = {}
    for node, (x, y) in pos.items():
        radius = math.sqrt(float(x) ** 2 + float(y) ** 2)
        if radius == 0:
            label_pos[node] = (x, y - 0.42)
            continue
        radial_offset = 0.28 + 0.025 * min(radius, 8.0)
        tangent_shift = 0.18 if y >= 0 else -0.18
        label_pos[node] = (
            float(x) * (1 + radial_offset / radius) - float(y) / radius * tangent_shift,
            float(y) * (1 + radial_offset / radius) + float(x) / radius * tangent_shift,
        )

    fig, ax = plt.subplots(figsize=(24, 19))
    ax.set_title("Pipeline Relation Graph")
    ax.axis("off")

    degree = dict(g.degree())
    node_sizes = [820 + 140 * degree[n] for n in g.nodes()]
    type_colors = {
        "PER": "#dbeafe",
        "ORG": "#dcfce7",
        "LOC": "#fde68a",
        "PROD": "#f5d0fe",
        "CONCEPT": "#fde2e4",
    }
    node_colors = [type_colors.get(g.nodes[n].get("node_type", ""), "#e5e7eb") for n in g.nodes()]
    nx.draw_networkx_nodes(g, pos, node_size=node_sizes, node_color=node_colors, edgecolors="#475569", linewidths=1.2, ax=ax)
    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=16,
        width=1.0,
        alpha=0.65,
        edge_color="#94a3b8",
        connectionstyle="arc3,rad=0.06",
        min_source_margin=20,
        min_target_margin=20,
    )
    nx.draw_networkx_labels(
        g,
        label_pos,
        labels={n: wrapped_label(g.nodes[n]["label"], width=9) for n in g.nodes()},
        font_size=8,
        ax=ax,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92, "pad": 1.8},
    )
    def draw_edge_labels() -> None:
        incident_count: Dict[str, int] = Counter()
        for u, v in g.edges():
            incident_count[u] += 1
            incident_count[v] += 1

        for idx, ((u, v), label) in enumerate(edge_labels.items()):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0
            dx = x2 - x1
            dy = y2 - y1
            norm = math.hypot(dx, dy) or 1.0
            nx_off = -dy / norm
            ny_off = dx / norm

            crowded = max(incident_count.get(u, 0), incident_count.get(v, 0)) >= 5
            base_offset = 0.34 if crowded else 0.24
            offset_sign = 1 if (idx % 2 == 0) else -1
            if u == hub or v == hub:
                offset_sign = 1 if y1 <= y2 else -1
                base_offset += 0.12

            lx = mx + nx_off * base_offset * offset_sign
            ly = my + ny_off * base_offset * offset_sign
            angle = math.degrees(math.atan2(dy, dx))
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180

            ax.text(
                lx,
                ly,
                label,
                fontsize=8,
                rotation=angle,
                rotation_mode="anchor",
                ha="center",
                va="center",
                color="#0f172a",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#cbd5e1",
                    "alpha": 0.95,
                    "boxstyle": "round,pad=0.22",
                },
                zorder=5,
            )

    draw_edge_labels()

    legend_handles = []
    for node_type, color in type_colors.items():
        if any(g.nodes[n].get("node_type") == node_type for n in g.nodes()):
            legend_handles.append(Line2D([0], [0], marker="o", color="w", label=node_type, markerfacecolor=color, markeredgecolor="#475569", markersize=9))
    if legend_handles:
        ax.legend(handles=legend_handles, title="Node Type", loc="upper left", frameon=True)
    ax.margins(0.36)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _configure_matplotlib()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ablation:
        ablation = _load_json(args.ablation)
        plot_ablation(ablation, out_dir / "ner_ablation.png")
        print(f"Saved: {out_dir / 'ner_ablation.png'}")

    if args.errors:
        errors = _load_jsonl(args.errors)
        plot_error_analysis(errors, out_dir / "ner_error_analysis.png")
        print(f"Saved: {out_dir / 'ner_error_analysis.png'}")

    if args.pipeline:
        pipeline = _load_json(args.pipeline)
        plot_relation_graph(pipeline, out_dir / "pipeline_relation_graph.png")
        print(f"Saved: {out_dir / 'pipeline_relation_graph.png'}")


if __name__ == "__main__":
    main()
