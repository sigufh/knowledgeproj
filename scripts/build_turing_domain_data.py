from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kg_pipeline.data.io import dump_jsonl


def split_sentences(text: str) -> List[str]:
    seps = "。！？!?;；\n"
    out: List[str] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in seps:
            sent = text[start : i + 1].strip()
            if sent:
                out.append(sent)
            start = i + 1
    if start < len(text):
        tail = text[start:].strip()
        if tail:
            out.append(tail)
    return out


def span(text: str, mention: str, nth: int = 1) -> Tuple[int, int]:
    pos = -1
    s = 0
    for _ in range(nth):
        pos = text.find(mention, s)
        if pos < 0:
            raise ValueError(f"mention not found: {mention}")
        s = pos + 1
    return pos, pos + len(mention)


def ent(text: str, mention: str, label: str, nth: int = 1) -> Dict:
    s, e = span(text, mention, nth=nth)
    return {"start": s, "end": e, "label": label}


def rel(text: str, h_m: str, h_l: str, t_m: str, t_l: str, label: str, h_n: int = 1, t_n: int = 1) -> Dict:
    hs, he = span(text, h_m, h_n)
    ts, te = span(text, t_m, t_n)
    return {
        "text": text,
        "head": {"start": hs, "end": he, "label": h_l},
        "tail": {"start": ts, "end": te, "label": t_l},
        "label": label,
    }


def main() -> None:
    input_path = Path("data/sample/input.txt")
    text = input_path.read_text(encoding="utf-8")
    sents = split_sentences(text)

    def find_sent(*must_contain: str, nth: int = 1) -> str:
        hit = 0
        for s in sents:
            if all(k in s for k in must_contain):
                hit += 1
                if hit == nth:
                    return s
        raise ValueError(f"cannot find sentence with keys={must_contain}, nth={nth}")

    # Pick domain-consistent sentences from current input text by keywords.
    s0 = find_sent("艾伦·麦席森·图灵", "Alan Mathison Turing")
    s_bombe = find_sent("Bombe", "恩尼格玛密码机")
    s_test = find_sent("图灵测试")
    s_tm = find_sent("图灵机模型")
    s_father = find_sent("朱利斯·麦席森·图灵", "Julius Mathison Turing")
    s_school1 = find_sent("圣迈克尔", "St. Michael")
    s_school2 = find_sent("马尔伯勒学院")
    s_school3 = find_sent("舍伯恩学校")
    s_school4 = find_sent("南安普顿")
    s_cambridge = find_sent("图灵考入剑桥大学国王学院")
    s_princeton = find_sent("普林斯顿大学")
    s_bletchley = find_sent("布莱切利庄园破解德国密码")
    s_gccs = find_sent("兼职工作", "GC&CS")
    s_knox = find_sent("第利温·诺克斯", "恩尼格玛密码机")
    s_obe = find_sent("乔治六世国王任命", "大英帝国勋章")
    s_npl = find_sent("国家物理实验室")
    s_uom = find_sent("曼彻斯特大学")
    s_morph = find_sent("形态发生的化学基础")
    s_apology = find_sent("戈登·布朗", "公开道歉")
    s_pardon = find_sent("英国司法部", "赦免")
    s_petition = find_sent("英国首相府邸", "英国政府")
    s_act = find_sent("艾伦·图灵法案")

    ner_rows = [
        {"text": s0, "entities": [ent(s0, "艾伦·麦席森·图灵", "PER"), ent(s0, "Alan Mathison Turing", "PER"), ent(s0, "阿兰·图灵", "PER"), ent(s0, "英国", "LOC")]},
        {"text": s_bombe, "entities": [ent(s_bombe, "Bombe", "PROD"), ent(s_bombe, "恩尼格玛密码机", "PROD")]},
        {"text": s_test, "entities": [ent(s_test, "图灵测试", "PROD"), ent(s_test, "图灵", "PER")]},
        {"text": s_tm, "entities": [ent(s_tm, "图灵", "PER"), ent(s_tm, "图灵机", "PROD")]},
        {"text": s_father, "entities": [ent(s_father, "朱利斯·麦席森·图灵", "PER"), ent(s_father, "Julius Mathison Turing", "PER"), ent(s_father, "英属印度", "LOC")]},
        {"text": s_school1, "entities": [ent(s_school1, "圣迈克尔", "ORG"), ent(s_school1, "St. Michael", "ORG")]},
        {"text": s_school2, "entities": [ent(s_school2, "马尔伯勒学院", "ORG")]},
        {"text": s_school3, "entities": [ent(s_school3, "舍伯恩学校", "ORG")]},
        {"text": s_school4, "entities": [ent(s_school4, "图灵", "PER"), ent(s_school4, "南安普顿", "LOC")]},
        {"text": s_cambridge, "entities": [ent(s_cambridge, "图灵", "PER"), ent(s_cambridge, "剑桥大学国王学院", "ORG")]},
        {"text": s_princeton, "entities": [ent(s_princeton, "图灵", "PER"), ent(s_princeton, "普林斯顿大学", "ORG"), ent(s_princeton, "Jane Eliza Procter Visiting Fellow", "PROD")]},
        {"text": s_bletchley, "entities": [ent(s_bletchley, "图灵", "PER"), ent(s_bletchley, "布莱切利庄园", "LOC")]},
        {"text": s_gccs, "entities": [ent(s_gccs, "图灵", "PER"), ent(s_gccs, "GC&CS", "ORG")]},
        {"text": s_knox, "entities": [ent(s_knox, "GC&CS", "ORG"), ent(s_knox, "第利温·诺克斯", "PER"), ent(s_knox, "恩尼格玛密码机", "PROD")]},
        {"text": s_obe, "entities": [ent(s_obe, "图灵", "PER"), ent(s_obe, "乔治六世国王", "PER"), ent(s_obe, "大英帝国勋章", "PROD")]},
        {"text": s_npl, "entities": [ent(s_npl, "图灵", "PER"), ent(s_npl, "国家物理实验室", "ORG"), ent(s_npl, "ACE", "PROD")]},
        {"text": s_uom, "entities": [ent(s_uom, "曼彻斯特大学", "ORG"), ent(s_uom, "曼彻斯特一号", "PROD")]},
        {"text": s_morph, "entities": [ent(s_morph, "形态发生的化学基础", "PROD"), ent(s_morph, "The Chemical Basis of Morphogenesis", "PROD")]},
        {"text": s_apology, "entities": [ent(s_apology, "英国首相戈登·布朗", "PER"), ent(s_apology, "每日电讯报", "ORG"), ent(s_apology, "英国政府", "ORG"), ent(s_apology, "艾伦·图灵", "PER")]},
        {"text": s_pardon, "entities": [ent(s_pardon, "英国司法部", "ORG"), ent(s_pardon, "英国女王伊丽莎白二世", "PER"), ent(s_pardon, "艾伦·图灵", "PER"), ent(s_pardon, "图灵", "PER")]},
        {"text": s_petition, "entities": [ent(s_petition, "图灵", "PER"), ent(s_petition, "英国首相府邸", "ORG"), ent(s_petition, "英国政府", "ORG")]},
        {"text": s_act, "entities": [ent(s_act, "艾伦·图灵法案", "PROD")]},
    ]

    ner_train = ner_rows[:12]
    ner_dev = ner_rows[12:]

    re_rows = [
        rel(s_cambridge, "图灵", "PER", "剑桥大学国王学院", "ORG", "STUDIED_AT"),
        rel(s_princeton, "图灵", "PER", "普林斯顿大学", "ORG", "STUDIED_AT"),
        rel(s_gccs, "图灵", "PER", "GC&CS", "ORG", "WORKED_FOR"),
        rel(s_knox, "第利温·诺克斯", "PER", "恩尼格玛密码机", "PROD", "CRACKED"),
        rel(s_bombe, "Bombe", "PROD", "恩尼格玛密码机", "PROD", "NO_REL"),
        rel(s_test, "图灵", "PER", "图灵测试", "PROD", "PROPOSED"),
        rel(s_tm, "图灵", "PER", "图灵机", "PROD", "PROPOSED"),
        rel(s_npl, "图灵", "PER", "国家物理实验室", "ORG", "WORKED_FOR"),
        rel(s_obe, "图灵", "PER", "大英帝国勋章", "PROD", "AWARDED"),
        rel(s_apology, "英国政府", "ORG", "艾伦·图灵", "PER", "APOLOGIZED_TO"),
        rel(s_pardon, "英国司法部", "ORG", "艾伦·图灵", "PER", "PARDONED"),
        rel(s0, "艾伦·麦席森·图灵", "PER", "英国", "LOC", "NO_REL"),
    ]

    re_train = re_rows[:10]
    re_dev = re_rows[10:]

    kb = [
        {"entity_id": "E_PER_TURING", "name": "艾伦·图灵", "aliases": ["艾伦·麦席森·图灵", "阿兰·图灵", "图灵", "Alan Mathison Turing"], "type": "PER", "description": "英国数学家、计算机科学先驱。"},
        {"entity_id": "E_PER_JULIUS", "name": "朱利斯·麦席森·图灵", "aliases": ["Julius Mathison Turing"], "type": "PER", "description": "图灵的父亲。"},
        {"entity_id": "E_PER_NOX", "name": "第利温·诺克斯", "aliases": ["诺克斯"], "type": "PER", "description": "英国密码破译员。"},
        {"entity_id": "E_PER_ASA", "name": "Asa Briggs", "aliases": [], "type": "PER", "description": "历史学家。"},
        {"entity_id": "E_PER_GEORGE6", "name": "乔治六世国王", "aliases": ["乔治六世"], "type": "PER", "description": "英国国王。"},
        {"entity_id": "E_PER_BROWN", "name": "戈登·布朗", "aliases": ["英国首相戈登·布朗"], "type": "PER", "description": "英国前首相。"},
        {"entity_id": "E_PER_ELIZABETH2", "name": "伊丽莎白二世", "aliases": ["英国女王伊丽莎白二世"], "type": "PER", "description": "英国女王。"},
        {"entity_id": "E_ORG_CAMBRIDGE_KINGS", "name": "剑桥大学国王学院", "aliases": ["国王学院"], "type": "ORG", "description": "剑桥大学学院。"},
        {"entity_id": "E_ORG_PRINCETON", "name": "普林斯顿大学", "aliases": [], "type": "ORG", "description": "美国研究型大学。"},
        {"entity_id": "E_ORG_GCCS", "name": "GC&CS", "aliases": ["政府密码和密码学校"], "type": "ORG", "description": "英国密码破译组织。"},
        {"entity_id": "E_ORG_NAVY_UK", "name": "英国皇家海军", "aliases": [], "type": "ORG", "description": "英国海军。"},
        {"entity_id": "E_ORG_MI6", "name": "军情六处", "aliases": [], "type": "ORG", "description": "英国情报机构。"},
        {"entity_id": "E_ORG_NPL", "name": "国家物理实验室", "aliases": [], "type": "ORG", "description": "英国国家研究机构。"},
        {"entity_id": "E_ORG_UOM", "name": "曼彻斯特大学", "aliases": [], "type": "ORG", "description": "英国大学。"},
        {"entity_id": "E_ORG_DT", "name": "每日电讯报", "aliases": [], "type": "ORG", "description": "英国媒体。"},
        {"entity_id": "E_ORG_MOJ_UK", "name": "英国司法部", "aliases": [], "type": "ORG", "description": "英国司法部门。"},
        {"entity_id": "E_ORG_GOV_UK", "name": "英国政府", "aliases": [], "type": "ORG", "description": "英国政府机构。"},
        {"entity_id": "E_ORG_PM_UK", "name": "英国首相府邸", "aliases": [], "type": "ORG", "description": "英国首相官邸。"},
        {"entity_id": "E_LOC_UK", "name": "英国", "aliases": [], "type": "LOC", "description": "欧洲国家。"},
        {"entity_id": "E_LOC_LONDON", "name": "伦敦", "aliases": [], "type": "LOC", "description": "英国首都。"},
        {"entity_id": "E_LOC_INDIA", "name": "英属印度", "aliases": ["印度"], "type": "LOC", "description": "历史地理区域。"},
        {"entity_id": "E_LOC_BLETCHLEY", "name": "布莱切利庄园", "aliases": ["布莱切利", "Bletchley Park"], "type": "LOC", "description": "二战英国密码破译中心。"},
        {"entity_id": "E_PROD_ENIGMA", "name": "恩尼格玛密码机", "aliases": ["恩尼格玛"], "type": "PROD", "description": "德军密码系统。"},
        {"entity_id": "E_PROD_BOMBE", "name": "Bombe", "aliases": ["炸弹机"], "type": "PROD", "description": "用于破译恩尼格玛的机器。"},
        {"entity_id": "E_PROD_TM", "name": "图灵机", "aliases": [], "type": "PROD", "description": "可计算理论中的抽象模型。"},
        {"entity_id": "E_PROD_TT", "name": "图灵测试", "aliases": [], "type": "PROD", "description": "机器智能判定测试。"},
        {"entity_id": "E_PROD_ACE", "name": "ACE", "aliases": ["自动计算引擎"], "type": "PROD", "description": "早期计算机设计项目。"},
        {"entity_id": "E_PROD_M1", "name": "曼彻斯特一号", "aliases": [], "type": "PROD", "description": "早期电子计算机。"},
        {"entity_id": "E_PROD_OBE", "name": "大英帝国勋章", "aliases": ["OBE"], "type": "PROD", "description": "英国荣誉勋章。"},
        {"entity_id": "E_PROD_MORPH", "name": "形态发生的化学基础", "aliases": ["The Chemical Basis of Morphogenesis"], "type": "PROD", "description": "图灵在数理生物学的重要论文。"},
        {"entity_id": "E_PROD_TURING_ACT", "name": "艾伦·图灵法案", "aliases": ["图灵法案"], "type": "PROD", "description": "英国相关赦免法案。"},
    ]

    relation_ontology = [
        {"relation_id": "R_STUDIED_AT", "name_en": "STUDIED_AT", "name_zh": "就读于", "aliases": ["考入", "就读于", "攻读"]},
        {"relation_id": "R_WORKED_FOR", "name_en": "WORKED_FOR", "name_zh": "任职于", "aliases": ["任职", "兼职工作", "成为", "负责"]},
        {"relation_id": "R_CRACKED", "name_en": "CRACKED", "name_zh": "破译", "aliases": ["破解", "破译", "分析"]},
        {"relation_id": "R_PROPOSED", "name_en": "PROPOSED", "name_zh": "提出", "aliases": ["提出"]},
        {"relation_id": "R_AWARDED", "name_en": "AWARDED", "name_zh": "获授", "aliases": ["任命为", "授予", "获授"]},
        {"relation_id": "R_APOLOGIZED_TO", "name_en": "APOLOGIZED_TO", "name_zh": "公开道歉", "aliases": ["公开道歉", "道歉"]},
        {"relation_id": "R_PARDONED", "name_en": "PARDONED", "name_zh": "赦免", "aliases": ["赦免"]},
        {"relation_id": "R_REPUTED_AS", "name_en": "REPUTED_AS", "name_zh": "被誉为", "aliases": ["被誉为"]},
    ]

    base = Path("data/sample")
    dump_jsonl(ner_train, base / "ner_train.jsonl")
    dump_jsonl(ner_dev, base / "ner_dev.jsonl")
    dump_jsonl(re_train, base / "re_train.jsonl")
    dump_jsonl(re_dev, base / "re_dev.jsonl")
    dump_jsonl(kb, base / "kb.jsonl")
    (base / "relation_ontology.json").write_text(json.dumps(relation_ontology, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Built Turing-domain dataset from input.txt")
    print(f"NER train/dev: {len(ner_train)}/{len(ner_dev)}")
    print(f"RE train/dev: {len(re_train)}/{len(re_dev)}")
    print(f"KB size: {len(kb)}")


if __name__ == "__main__":
    main()
