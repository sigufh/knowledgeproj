#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXTRA_ROWS = [
    {
        "entity_id": "E_PER_ETHEL",
        "name": "Ethel Sara Stoney",
        "aliases": ["图灵的母亲Ethel Sara Stoney"],
        "type": "PER",
        "description": "艾伦·图灵的母亲。",
    },
    {
        "entity_id": "E_LOC_PADDINGTON",
        "name": "帕丁顿",
        "aliases": [],
        "type": "LOC",
        "description": "伦敦地区地名，图灵出生相关地点。",
    },
    {
        "entity_id": "E_ORG_SHERBORNE",
        "name": "舍伯恩学校",
        "aliases": ["舍伯恩", "Sherborne School"],
        "type": "ORG",
        "description": "图灵少年时期就读学校。",
    },
    {
        "entity_id": "E_ORG_UOM_CL",
        "name": "曼彻斯特大学计算机实验室",
        "aliases": ["曼彻斯特计算机实验室", "Manchester Computing Laboratory"],
        "type": "ORG",
        "description": "图灵任副主任的计算机实验室。",
    },
    {
        "entity_id": "E_PROD_CMI",
        "name": "计算机器和智能",
        "aliases": ["计算机械和智能", "Computing Machinery and Intelligence"],
        "type": "PROD",
        "description": "图灵关于机器智能的重要论文。",
    },
    {
        "entity_id": "E_PROD_CMT",
        "name": "机器会思考吗",
        "aliases": ["Can Machines Think", "Can Machines Think?"],
        "type": "PROD",
        "description": "图灵在机器智能讨论中的经典提问。",
    },
    {
        "entity_id": "E_PROD_APC",
        "name": "The Applications of Probability to Cryptography",
        "aliases": [],
        "type": "PROD",
        "description": "图灵关于密码分析概率方法的论文。",
    },
    {
        "entity_id": "E_PROD_PSR",
        "name": "Paper on Statistics of Repetitions",
        "aliases": [],
        "type": "PROD",
        "description": "图灵关于重复统计的密码分析论文。",
    },
    {
        "entity_id": "E_ORG_HUT8",
        "name": "Hut 8",
        "aliases": ["小屋8"],
        "type": "ORG",
        "description": "图灵二战时期参与的布莱切利密码分析小组。",
    },
    {
        "entity_id": "E_ORG_GCHQ",
        "name": "GCHQ",
        "aliases": ["英国政府通信总部"],
        "type": "ORG",
        "description": "GC&CS 的继任机构，英国政府通信总部。",
    },
    {
        "entity_id": "E_PROD_OCN",
        "name": "On Computable Numbers",
        "aliases": ["On Computable Numbers, with an Application to the Entscheidungsproblem"],
        "type": "PROD",
        "description": "图灵关于可计算性的经典论文。",
    },
    {
        "entity_id": "E_ORG_ST_MICHAEL",
        "name": "St. Michael's",
        "aliases": ["St. Michael", "圣迈克尔"],
        "type": "ORG",
        "description": "图灵幼年注册的日间学校。",
    },
    {
        "entity_id": "E_PER_LEWIN",
        "name": "Ronald Lewin",
        "aliases": [],
        "type": "PER",
        "description": "研究布莱切利庄园历史的学者。",
    },
    {
        "entity_id": "E_PER_VON_NEUMANN",
        "name": "冯·诺依曼",
        "aliases": ["John von Neumann", "约翰·冯·诺依曼"],
        "type": "PER",
        "description": "数学家，曾考虑聘请图灵做博士后助理。",
    },
    {
        "entity_id": "E_PER_CHURCH",
        "name": "阿隆佐·邱奇",
        "aliases": ["比阿隆佐·邱奇", "Alonzo Church"],
        "type": "PER",
        "description": "逻辑学家，与图灵工作相关。",
    },
    {
        "entity_id": "E_PER_EINSTEIN",
        "name": "阿尔伯特·爱因斯坦",
        "aliases": ["Albert Einstein", "爱因斯坦"],
        "type": "PER",
        "description": "物理学家，图灵少年时期阅读其著作。",
    },
    {
        "entity_id": "E_ORG_LANL",
        "name": "新墨西哥州洛斯阿拉莫斯国家实验室",
        "aliases": ["洛斯阿拉莫斯国家实验室", "Los Alamos National Laboratory"],
        "type": "ORG",
        "description": "后续依据图灵理论实现国际象棋程序的研究机构。",
    },
    {
        "entity_id": "E_PROD_ENIAC",
        "name": "ENIAC",
        "aliases": [],
        "type": "PROD",
        "description": "早期电子计算机平台。",
    },
    {
        "entity_id": "E_PROD_LA_CHESS",
        "name": "洛斯阿拉莫斯象棋",
        "aliases": [],
        "type": "PROD",
        "description": "在 ENIAC 上实现的早期国际象棋程序。",
    },
    {
        "entity_id": "E_ORG_WAC",
        "name": "沃尔顿竞技俱乐部",
        "aliases": [],
        "type": "ORG",
        "description": "图灵跑步相关俱乐部。",
    },
    {
        "entity_id": "E_PROD_DECODE_NAZI",
        "name": "解码纳粹秘密",
        "aliases": ["Nova PBS纪录片《解码纳粹秘密》"],
        "type": "PROD",
        "description": "与图灵相关的纪录片作品。",
    },
    {
        "entity_id": "E_PER_RICHARDS",
        "name": "Thomas Richards",
        "aliases": ["托马斯·理查兹", "汤姆·理查兹"],
        "type": "PER",
        "description": "与图灵长跑成绩对比的人物。",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment Turing KB with missing entities for current input text")
    p.add_argument("--kb", default="data/sample/kb.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    kb_path = Path(args.kb)
    rows = []
    if kb_path.exists():
        with kb_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    by_id = {row["entity_id"]: row for row in rows}
    for row in EXTRA_ROWS:
        by_id[row["entity_id"]] = row

    merged = list(by_id.values())
    merged.sort(key=lambda row: row["entity_id"])
    with kb_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"KB updated: {kb_path}")
    print(f"KB size: {len(merged)}")


if __name__ == "__main__":
    main()
