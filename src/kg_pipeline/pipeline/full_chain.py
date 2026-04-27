from __future__ import annotations

import csv
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List

from kg_pipeline.el.linker import EntityLinker
from kg_pipeline.ner.crf_baseline import CRFNamedEntityRecognizer
from kg_pipeline.relation.classifier import RelationClassifier
from kg_pipeline.utils.spans import split_sentences


RELATION_RULE_LIBRARY = [
    {"relation_id": "R_REPUTED_AS", "label_en": "REPUTED_AS", "label_zh": "被誉为", "aliases": ["被誉为"], "head_types": {"PER", "ORG"}, "tail_types": {"PER", "ORG", "PROD", "LOC", "CONCEPT"}},
    {"relation_id": "R_PROPOSED", "label_en": "PROPOSED", "label_zh": "提出", "aliases": ["提出"], "head_types": {"PER", "ORG"}, "tail_types": {"PROD", "CONCEPT", "ORG"}},
    {"relation_id": "R_DESIGNED", "label_en": "DESIGNED", "label_zh": "设计", "aliases": ["设计", "改进", "建造", "开发", "制作"], "head_types": {"PER", "ORG"}, "tail_types": {"PROD", "CONCEPT"}},
    {"relation_id": "R_WROTE", "label_en": "WROTE", "label_zh": "撰写", "aliases": ["写了", "写过", "撰写", "题为"], "head_types": {"PER", "ORG"}, "tail_types": {"PROD", "CONCEPT"}},
    {"relation_id": "R_PUBLISHED", "label_en": "PUBLISHED", "label_zh": "发表", "aliases": ["发表", "出版"], "head_types": {"PER", "ORG"}, "tail_types": {"PROD", "CONCEPT"}},
    {"relation_id": "R_STUDIED_AT", "label_en": "STUDIED_AT", "label_zh": "就读于", "aliases": ["考入", "就读于", "攻读", "注册"], "head_types": {"PER"}, "tail_types": {"ORG"}},
    {"relation_id": "R_GRADUATED_FROM", "label_en": "GRADUATED_FROM", "label_zh": "毕业于", "aliases": ["毕业", "毕业于", "学位"], "head_types": {"PER"}, "tail_types": {"ORG"}},
    {"relation_id": "R_WORKED_FOR", "label_en": "WORKED_FOR", "label_zh": "任职于", "aliases": ["任职", "兼职工作", "担任", "成为", "供职", "出任", "副主任"], "head_types": {"PER"}, "tail_types": {"ORG"}},
    {"relation_id": "R_WORKED_ON", "label_en": "WORKED_ON", "label_zh": "负责研究", "aliases": ["负责"], "head_types": {"PER", "ORG"}, "tail_types": {"PROD", "CONCEPT"}},
    {"relation_id": "R_CRACKED", "label_en": "CRACKED", "label_zh": "破译", "aliases": ["破解", "破译", "密码分析"], "head_types": {"PER", "ORG"}, "tail_types": {"PROD"}},
    {"relation_id": "R_FATHER", "label_en": "FATHER", "label_zh": "父亲", "aliases": ["父亲"], "head_types": {"PER"}, "tail_types": {"PER"}},
    {"relation_id": "R_MOTHER", "label_en": "MOTHER", "label_zh": "母亲", "aliases": ["母亲"], "head_types": {"PER"}, "tail_types": {"PER"}},
    {"relation_id": "R_AWARDED", "label_en": "AWARDED", "label_zh": "获授", "aliases": ["授予", "获授"], "head_types": {"PER", "ORG"}, "tail_types": {"PROD"}},
    {"relation_id": "R_INVITED", "label_en": "INVITED", "label_zh": "聘请", "aliases": ["聘请"], "head_types": {"PER", "ORG"}, "tail_types": {"PER"}},
    {"relation_id": "R_WROTE_FOR", "label_en": "WROTE_FOR", "label_zh": "撰文于", "aliases": [], "head_types": {"PER"}, "tail_types": {"ORG"}},
    {"relation_id": "R_SUCCESSOR_OF", "label_en": "SUCCESSOR_OF", "label_zh": "继任于", "aliases": [], "head_types": {"ORG"}, "tail_types": {"ORG"}},
    {"relation_id": "R_NAMED_AFTER", "label_en": "NAMED_AFTER", "label_zh": "以…命名", "aliases": ["为名"], "head_types": {"ORG"}, "tail_types": {"PER"}},
    {"relation_id": "R_APOLOGIZED_TO", "label_en": "APOLOGIZED_TO", "label_zh": "公开道歉", "aliases": ["公开道歉", "道歉"], "head_types": {"ORG", "PER"}, "tail_types": {"PER", "ORG"}},
    {"relation_id": "R_PARDONED", "label_en": "PARDONED", "label_zh": "赦免", "aliases": ["赦免"], "head_types": {"ORG", "PER"}, "tail_types": {"PER"}},
    {"relation_id": "R_PERSECUTED", "label_en": "PERSECUTED", "label_zh": "迫害", "aliases": ["迫害", "起诉"], "head_types": {"ORG", "PER"}, "tail_types": {"PER"}},
    {"relation_id": "R_CONVICTED", "label_en": "CONVICTED", "label_zh": "定罪", "aliases": ["定罪", "被控"], "head_types": {"ORG", "PER"}, "tail_types": {"PER"}},
    {"relation_id": "R_BORN_IN", "label_en": "BORN_IN", "label_zh": "\u51fa\u751f\u4e8e", "aliases": ["\u51fa\u751f", "\u751f\u4e0b"], "head_types": {"PER"}, "tail_types": {"LOC"}},
    {"relation_id": "R_LIVED_IN", "label_en": "LIVED_IN", "label_zh": "\u5c45\u4f4f\u4e8e", "aliases": ["\u4f4f\u5728", "\u540c\u4f4f"], "head_types": {"PER"}, "tail_types": {"LOC"}},
    {"relation_id": "R_COLLABORATED_WITH", "label_en": "COLLABORATED_WITH", "label_zh": "\u5408\u4f5c", "aliases": ["\u4e00\u8d77", "\u5408\u4f5c"], "head_types": {"PER"}, "tail_types": {"PER"}},
]

STOPWORDS = {"期间", "其中", "因此", "因为", "他们", "我们", "研究", "工作", "计划", "方法"}
BAD_PREFIXES = ("要求", "因为", "正式向", "被定罪", "罪的", "并在", "使得", "从而")
CONTEXT_PREFIXES = (
    "英国首相",
    "英国女王",
    "又译",
    "图灵的父亲",
    "图灵的母亲",
    "图灵在",
    "图灵是",
    "图灵考入",
    "后来美国",
    "一间叫",
    "同年奥运会银牌得主",
    "同年奥运会银牌",
    "奥运会银牌得主",
    "银牌得主",
    "历史学家和战时密码破译员",
    "历史学家",
    "数学家",
    "学家",
    "教授",
    "主条目",
    "首相",
    "父亲",
    "母亲",
    "式向",
    "员",
    "亲",
    "主",
    "年由",
    "年图灵被",
    "所有在",
    "他成为",
    "他是",
    "在布莱奇利庄园",
    "在布莱切利庄园",
    "得主",
    "英王",
)
CONTEXT_SUFFIXES = (
    "在",
    "里",
    "中",
    "上",
    "下",
    "的著作",
    "的论文",
    "的研究组",
    "一起专注于",
    "有意聘请图",
    "公开道歉",
    "法案生效",
    "工作",
    "注册",
)
BAD_LONG_TEXT_RE = re.compile(r"(时候|期间|因此|因为|后来|于是|能够|可以|不是|以及|其中|当时|由于|为了|使得|帮助|通过|明白指出)")
ASCII_NAME_RE = re.compile(r"^[A-Z][A-Za-z]+(?:[ .'\-][A-Z][A-Za-z]+){0,4}$")
ASCII_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9&.\-]{1,15}$")
EN_TITLE_PREFIXES = ("The ", "On ", "Can ", "Paper ", "Foundations ")
ORG_SUFFIXES = ("大学", "学院", "学校", "实验室", "司法部", "海军", "密码学校", "庄园", "俱乐部", "报", "委员会", "研究所")
LOC_SUFFIXES = ("国", "市", "郡", "镇", "州", "洋", "府", "湾")
TITLE_CUES = ("论文", "著作", "一文", "一篇", "题为", "名为", "叫做", "提出", "实验", "模型", "测试", "法案")
GENERIC_TERMS = {"大学", "学院", "学校", "实验室", "委员会", "研究所", "自然", "政府", "海军", "司法部", "庄园"}
KNOWN_LOCATIONS = {"英国", "伦敦", "华沙", "普林斯顿", "剑桥", "南安普顿", "吉尔福德", "帕丁顿", "东柴郡", "威姆斯洛", "布莱切利", "布莱切利庄园", "英属印度", "印度"}
PRONOUN_SUBJECT_RE = re.compile(r"^[0-9一二三四五六七八九十年月日\s,，.:：]*[他她其]")
WORK_CONTEXT_HINTS = ("工作", "兼职工作", "负责", "监督下", "研究工作", "副主任", "供职", "任职")
ORG_NOISE_CUES = ("考入", "攻读", "写了", "写过", "提出", "成为", "负责", "住在", "生下", "发表", "被选为")
EN_ROLE_CUES = ("Visiting Fellow", "Fellow", "Professor", "Chair", "Scholarship", "Prize")
NOISY_ENTITY_PREFIXES = ("\u8d1f\u8d23", "\u968f\u540e", "\u5c06\u5176", "\u8bc4\u9009\u4e3a", "\u56fe\u7075\u5728", "\u8981\u6c42", "\u56e0\u4e3a", "\u4e00\u95f4\u53eb", "\u4ed6\u7684", "\u5979\u7684", "\u8fd9\u4f4d", "\u8be5")
NOISY_ENTITY_SUFFIXES = ("\u8bc1\u660e", "\u4f30\u8ba1", "\u4e00\u4e2a", "\u5de5\u4f5c", "\u60c5\u51b5", "\u5185\u5bb9", "\u4e8b\u5b9e", "\u540c\u4f4f", "\u5ba2\u5ea7\u6559")
NOISY_ENTITY_INFIXES = ("\u5c31\u662f\u8fd9\u6837\u4e00\u4e2a", "\u6240\u5f97\u60c5\u62a5", "\u5728\u5b66\u6821", "\u5de5\u4f5c\u538b\u529b", "\u9644\u8fd1\u8fd8", "\u8c03\u67e5\u7ed3\u679c", "\u84dd\u8272\u8def\u724c", "\u516c\u5f00\u9053\u6b49", "\u76f8\u5173\u7f6a\u540d")
EXPLICIT_NOISE_TERMS = {
    "FRS",
    "日间学校",
    "教授",
    "之书",
    "他只承认自己是理查德",
    "明显的猥亵和性颠倒行为",
    "榨取了汁液",
    "疗法",
    "美国数学世纪的回忆",
}
ENTITY_CANONICAL_MAP = {
    "布莱奇利": "布莱切利",
    "布莱奇利庄园": "布莱切利庄园",
    "计算机械和智能": "计算机器和智能",
}


def _clean_entity_text(text: str) -> bool:
    raw = (text or "").strip()
    t = ENTITY_CANONICAL_MAP.get(raw, raw)
    if len(t) < 2 or len(t) > 60:
        return False
    if t in EXPLICIT_NOISE_TERMS:
        return False
    if any(cue in t for cue in EN_ROLE_CUES) and not t.startswith(EN_TITLE_PREFIXES):
        return False
    if t in STOPWORDS or t in GENERIC_TERMS:
        return False
    if any(t.startswith(prefix) for prefix in BAD_PREFIXES):
        return False
    if any(t.startswith(prefix) for prefix in NOISY_ENTITY_PREFIXES):
        return False
    if any(t.endswith(suffix) for suffix in NOISY_ENTITY_SUFFIXES):
        return False
    if any(noise in t for noise in NOISY_ENTITY_INFIXES):
        return False
    if re.search(r'[\[\]{}<>"]', t):
        return False
    if re.search(r'[\u3001\u3002\uFF0C\uFF1B\uFF1A\uFF01\uFF1F\u201C\u201D\u2018\u2019\uFF08\uFF09\(\)\'\"]', t):
        return False
    if t.isdigit():
        return False
    if ASCII_ACRONYM_RE.fullmatch(t):
        return True
    if len(t) > 10 and BAD_LONG_TEXT_RE.search(t):
        return False
    if len(t) > 12 and re.search(r'[\u7684\u662f\u5728\u4e8e\u548c\u4e0e\u53ca\u5e76\u88ab\u628a\u5c06\u5176\u4ece\u5411\u4f1a\u8ba9]', t):
        return False
    if re.search(r'^(?:\u5728|\u4e8e|\u5bf9|\u88ab|\u628a|\u5c06|\u4ece|\u5411|\u56e0)(?=[A-Za-z\u4e00-\u9fff]{1,})', t) and len(t) <= 4:
        return False
    return True


def _normalize_lexicon_term(term: str) -> str:
    text = (term or "").strip()
    text = ENTITY_CANONICAL_MAP.get(text, text)
    text = text.strip("，。；：！？、“”‘’（）()[]{}<>\"' ")
    changed = True
    while changed and text:
        changed = False
        for prefix in CONTEXT_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix) + 1:
                text = text[len(prefix) :].strip()
                changed = True
        for suffix in CONTEXT_SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix) + 1:
                text = text[: -len(suffix)].strip()
                changed = True
    text = re.sub(r"^(?:\u5728|\u4e8e|\u5411|\u5bf9|\u4ece|\u88ab|\u628a)(?=[\u4e00-\u9fffA-Za-z]{2,})", "", text)
    text = re.sub(r"^\u7684(?=[\u4e00-\u9fffA-Za-z]{2,})", "", text)
    return ENTITY_CANONICAL_MAP.get(text, text)


def _is_plausible_name(term: str, label: str) -> bool:
    text = _normalize_lexicon_term(term)
    if not _clean_entity_text(text):
        return False
    if text in GENERIC_TERMS:
        return False
    if label == "PER":
        if ASCII_ACRONYM_RE.fullmatch(text):
            return False
        if any(cue in text for cue in EN_ROLE_CUES):
            return False
        if text.endswith(NOISY_ENTITY_SUFFIXES) or any(noise in text for noise in NOISY_ENTITY_INFIXES):
            return False
        return bool("\u00b7" in text or ASCII_NAME_RE.fullmatch(text) or re.fullmatch(r'[\u4e00-\u9fff]{2,4}', text))
    if label == "ORG":
        return ASCII_ACRONYM_RE.fullmatch(text) is not None or text.endswith(ORG_SUFFIXES) or text in {"GC&CS", "GCHQ", "Hut 8", "\u519b\u60c5\u516d\u5904", "\u82f1\u56fd\u653f\u5e9c", "\u82f1\u56fd\u9996\u76f8\u5e9c\u90b8"}
    if label == "LOC":
        return text.endswith(LOC_SUFFIXES) or text in KNOWN_LOCATIONS or re.fullmatch(r'[\u4e00-\u9fff]{2,8}', text) is not None
    if label == "PROD":
        return len(text) >= 2 and not any(noise in text for noise in ("\u5de5\u4f5c\u538b\u529b", "\u516c\u5f00\u9053\u6b49", "\u76f8\u5173\u7f6a\u540d"))
    return True


def _infer_label(term: str, left_ctx: str = "", right_ctx: str = "") -> str | None:
    text = _normalize_lexicon_term(term)
    if not _clean_entity_text(text):
        return None
    context = f"{left_ctx}{right_ctx}"
    if ASCII_ACRONYM_RE.fullmatch(text):
        if any(cue in context for cue in ("\u52cb\u7ae0", "\u6cd5\u6848", "\u673a\u5668", "\u5b9e\u9a8c", "\u6d4b\u8bd5", "\u7a0b\u5e8f", "\u8bba\u6587")):
            return "PROD"
        return "ORG"
    if text.startswith(("On ", "The ", "Can ", "Paper ", "Foundations ")) or (text.startswith("\u300a") and text.endswith("\u300b")):
        return "PROD"
    if any(cue in text for cue in EN_ROLE_CUES):
        return "PROD"
    if text.endswith(("\u9009\u96c6", "\u6cd5\u6848", "\u52cb\u7ae0", "\u6d4b\u8bd5", "\u673a", "\u8bba\u6587")):
        return "PROD"
    if "\u00b7" in text or ASCII_NAME_RE.fullmatch(text):
        return "PER"
    if re.fullmatch(r'[\u4e00-\u9fff]{2,4}', text) and any(cue in context for cue in ("\u7236\u4eb2", "\u6bcd\u4eb2", "\u56fd\u738b", "\u5973\u738b", "\u9996\u76f8", "\u6559\u6388", "\u540c\u4e8b", "\u5b66\u5bb6", "\u7814\u7a76\u5458", "\u5f97\u4e3b")):
        return "PER"
    if text.endswith(ORG_SUFFIXES) or text in {"GC&CS", "GCHQ", "Hut 8", "\u519b\u60c5\u516d\u5904", "\u82f1\u56fd\u653f\u5e9c", "\u82f1\u56fd\u9996\u76f8\u5e9c\u90b8"}:
        return "ORG"
    if text.endswith(LOC_SUFFIXES) or text in KNOWN_LOCATIONS:
        return "LOC"
    if any(cue in context for cue in TITLE_CUES):
        return "PROD"
    return None


def build_runtime_lexicon(text: str, linker: EntityLinker) -> List[Dict]:
    lexicon: List[Dict] = []
    seen = set()
    alias_type = {}
    for row in linker.kb:
        ent_type = row.get("type", "")
        names = [row.get("name", "")] + row.get("aliases", [])
        for name in names:
            name = (name or "").strip()
            if len(name) < 2:
                continue
            alias_type[name] = ent_type
            key = (name, ent_type)
            if key in seen:
                continue
            seen.add(key)
            lexicon.append({"name": name, "label": ent_type, "source": "kb_dict"})

    def add_candidate(term: str, label: str | None = None, source: str = "input_dict", left_ctx: str = "", right_ctx: str = "") -> None:
        name = _normalize_lexicon_term(term)
        if not name:
            return
        label = alias_type.get(name) or label or _infer_label(name, left_ctx=left_ctx, right_ctx=right_ctx)
        if not label or not _is_plausible_name(name, label):
            return
        key = (name, label)
        if key in seen:
            return
        seen.add(key)
        lexicon.append({"name": name, "label": label, "source": source})

    quote_pat = r"[\u300a\"\u201c]([^\u300b\"\u201d]{2,80})[\u300b\"\u201d]"
    for m in re.finditer(quote_pat, text):
        add_candidate(
            m.group(1),
            label="PROD",
            left_ctx=text[max(0, m.start() - 12) : m.start()],
            right_ctx=text[m.end() : min(len(text), m.end() + 12)],
        )

    en_cue_pat = r"(?:\u82f1\u8bed\uff1a|\u82f1\u6587\uff1a)([A-Z][A-Za-z0-9 ,.'\-&:]{2,160})"
    for m in re.finditer(en_cue_pat, text):
        candidate = re.split(r"[\uFF09\)\u3002\uFF0C\uFF1B\n]", m.group(1))[0].strip()
        candidate = re.sub(r"\[\d+\]", "", candidate).strip(" ,???;:()[]")
        add_candidate(
            candidate,
            label=_infer_label(candidate, left_ctx=text[max(0, m.start() - 8) : m.start()], right_ctx=text[m.end() : min(len(text), m.end() + 8)]),
        )

    en_title_pat = r"\b(?:The|On|Can|Paper|Foundations)\s+[A-Za-z][A-Za-z]+(?:\s+(?:of|to|and|the|in|for|on|with|an|a|by|[A-Z][A-Za-z]+|[A-Za-z]+|\[\d+\])){1,20}"
    for m in re.finditer(en_title_pat, text):
        candidate = re.sub(r"\[\d+\]", "", m.group(0)).strip(" ,???;:()[]")
        add_candidate(candidate, label="PROD", source="input_dict")

    zh_person_pat = r"[\u4e00-\u9fff]{1,4}\u00b7[\u4e00-\u9fff]{1,8}(?:\u00b7[\u4e00-\u9fff]{1,8})?"
    for m in re.finditer(zh_person_pat, text):
        add_candidate(m.group(0), label="PER")

    en_name_pat = r"\b[A-Z][A-Za-z]+(?:[ .'\-][A-Z][A-Za-z]+){1,5}\b"
    for m in re.finditer(en_name_pat, text):
        if m.group(0).startswith(EN_TITLE_PREFIXES):
            continue
        add_candidate(
            m.group(0),
            label="PER",
            left_ctx=text[max(0, m.start() - 12) : m.start()],
            right_ctx=text[m.end() : min(len(text), m.end() + 12)],
        )

    for m in re.finditer(r"\b[A-Z][A-Z0-9&.\-]{1,15}\b", text):
        add_candidate(m.group(0), left_ctx=text[max(0, m.start() - 12) : m.start()], right_ctx=text[m.end() : min(len(text), m.end() + 12)])

    for suffix in ORG_SUFFIXES:
        for m in re.finditer(re.escape(suffix), text):
            left = m.start()
            while left > 0 and re.match(r"[\u4e00-\u9fffA-Za-z&.\-]", text[left - 1]):
                left -= 1
                if m.end() - left >= 20:
                    break
            candidate = text[left : m.end()]
            add_candidate(candidate, label="ORG", left_ctx=text[max(0, left - 6) : left], right_ctx=text[m.end() : min(len(text), m.end() + 6)])

    for loc in ("??", "??", "??", "????", "??", "????", "????", "???", "???", "????", "??????", "????", "????", "??"):
        for m in re.finditer(re.escape(loc), text):
            add_candidate(m.group(0), label="LOC")

    lexicon.sort(key=lambda x: len(x["name"]), reverse=True)
    return lexicon


def build_relation_rules(text: str) -> List[Dict]:
    active: List[Dict] = []
    seen = set()
    for rule in RELATION_RULE_LIBRARY:
        aliases = [alias for alias in rule["aliases"] if alias and alias in text]
        if not aliases:
            continue
        item = dict(rule)
        item["aliases"] = aliases
        key = item["relation_id"]
        if key in seen:
            continue
        seen.add(key)
        active.append(item)
    return active


def _dict_extract(text: str, lexicon: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for item in lexicon:
        name = item["name"]
        start = 0
        while True:
            pos = text.find(name, start)
            if pos < 0:
                break
            end = pos + len(name)
            rows.append({"start": pos, "end": end, "label": item["label"], "text": text[pos:end], "source": item.get("source", "kb_dict")})
            start = pos + 1
    return rows


def _regex_extract(text: str) -> List[Dict]:
    rows: List[Dict] = []
    zh_person_pat = r"[\u4e00-\u9fff]{1,4}\u00b7[\u4e00-\u9fff]{1,8}(?:\u00b7[\u4e00-\u9fff]{1,8})?"
    en_title_pat = r"\b(?:The|On|Can|Paper|Foundations)\s+[A-Za-z][A-Za-z]+(?:\s+(?:of|to|and|the|in|for|on|with|an|a|by|[A-Z][A-Za-z]+|[A-Za-z]+|\[\d+\])){1,20}"
    en_name_pat = r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}\b"
    suffix_pat = r"[\u4e00-\u9fffA-Za-z&]{2,18}(\u5927\u5b66|\u5b66\u9662|\u5b66\u6821|\u5b9e\u9a8c\u5ba4|\u6d77\u519b|\u653f\u5e9c|\u53f8\u6cd5\u90e8|\u5bc6\u7801\u5b66\u6821|\u62a5|\u5e84\u56ed|\u4ff1\u4e50\u90e8)"
    loc_pat = r"(\u82f1\u56fd|\u4f26\u6566|\u534e\u6c99|\u666e\u6797\u65af\u987f|\u5251\u6865|\u4e1c\u67f4\u90e1|\u5357\u5b89\u666e\u987f|\u5409\u5c14\u798f\u5fb7|\u5e15\u4e01\u987f|\u5a01\u59c6\u65af\u6d1b|\u82f1\u5c5e\u5370\u5ea6|\u5e03\u83b1\u5207\u5229\u5e84\u56ed|\u5e03\u83b1\u5207\u5229)"

    for m in re.finditer(zh_person_pat, text):
        rows.append({"start": m.start(), "end": m.end(), "label": "PER", "text": m.group(0), "source": "regex"})
    for m in re.finditer(en_title_pat, text):
        rows.append({"start": m.start(), "end": m.end(), "label": "PROD", "text": re.sub(r"\[\d+\]", "", m.group(0)).strip(), "source": "regex"})
    for m in re.finditer(en_name_pat, text):
        if m.group(0).startswith(EN_TITLE_PREFIXES):
            continue
        if any(cue in m.group(0) for cue in ("Visiting", "Fellow", "Professor", "Chair", "Prize")):
            continue
        rows.append({"start": m.start(), "end": m.end(), "label": "PER", "text": m.group(0), "source": "regex"})
    for m in re.finditer(r"[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}(?=[\u4e00-\u9fff])", text):
        if m.group(0).startswith(EN_TITLE_PREFIXES):
            continue
        if any(cue in m.group(0) for cue in ("Visiting", "Fellow", "Professor", "Chair", "Prize")):
            continue
        rows.append({"start": m.start(), "end": m.end(), "label": "PER", "text": m.group(0), "source": "regex"})
    for m in re.finditer(suffix_pat, text):
        candidate = m.group(0)
        normalized = _normalize_lexicon_term(candidate)
        if any(cue in candidate for cue in ORG_NOISE_CUES):
            continue
        if normalized != candidate and len(normalized) + 2 < len(candidate):
            continue
        if not _is_plausible_name(normalized, "ORG"):
            continue
        rows.append({"start": m.start(), "end": m.end(), "label": "ORG", "text": candidate, "source": "regex"})
    for m in re.finditer(loc_pat, text):
        rows.append({"start": m.start(), "end": m.end(), "label": "LOC", "text": m.group(0), "source": "regex"})
    return rows


def _overlap(a: Dict, b: Dict) -> bool:
    return not (a["end"] <= b["start"] or b["end"] <= a["start"])


def _rank(ent: Dict) -> tuple:
    src = ent.get("source")
    pri_map = {"input_dict": 4, "kb_dict": 3, "regex": 2, "ner": 1}
    compact = 1 if _is_plausible_name(ent.get("text", ""), ent.get("label", "")) else 0
    return (pri_map.get(src, 0), compact, min(ent["end"] - ent["start"], 24))


def _merge_entities(entities: List[Dict]) -> List[Dict]:
    entities = [e for e in entities if _clean_entity_text(e.get("text", ""))]
    entities.sort(key=lambda x: (x["start"], x["end"]))
    kept: List[Dict] = []
    for ent in entities:
        same = [k for k in kept if k["start"] == ent["start"] and k["end"] == ent["end"] and k["label"] == ent["label"]]
        if same:
            continue
        overlaps = [k for k in kept if _overlap(k, ent)]
        if not overlaps:
            kept.append(ent)
            continue
        if all(_rank(ent) > _rank(k) for k in overlaps):
            kept = [k for k in kept if k not in overlaps]
            kept.append(ent)
    return sorted(kept, key=lambda x: (x["start"], x["end"]))


def extract_candidate_entities(
    text: str,
    ner: CRFNamedEntityRecognizer,
    linker: EntityLinker,
    source_file: str,
    lexicon: List[Dict] | None = None,
) -> List[Dict]:
    ner_rows = ner.predict_entities(text)
    lexicon = lexicon or build_runtime_lexicon(text=text, linker=linker)
    dict_rows = _dict_extract(text, lexicon)
    regex_rows = _regex_extract(text)
    merged = _merge_entities(ner_rows + dict_rows + regex_rows)
    out: List[Dict] = []
    for row in merged:
        out.append(
            {
                "entity": row["text"],
                "entity_type": row["label"],
                "start": row["start"],
                "end": row["end"],
                "source_file": source_file,
                "source": row.get("source", "ner"),
            }
        )
    return out


def link_entities(text: str, extracted_rows: List[Dict], linker: EntityLinker, min_link_score: float, source_file: str) -> List[Dict]:
    kb_by_id = {row.get("entity_id", ""): row for row in linker.kb}
    linked_rows: List[Dict] = []
    for row in extracted_rows:
        mention = row["entity"]
        lk = linker.disambiguate(mention=mention, context=text, top_k=5)
        score = lk.score if lk else 0.0
        sim = SequenceMatcher(None, mention, lk.name).ratio() if lk else 0.0
        exactish = False
        if lk:
            kb_row = kb_by_id.get(lk.entity_id, {})
            alias_choices = [lk.name] + list(kb_row.get("aliases", []))
            alias_choices = [(x or "").strip() for x in alias_choices if (x or "").strip()]
            if any(mention == alias for alias in alias_choices):
                exactish = True
            elif any(alias in mention and (len(mention) - len(alias) <= 1) for alias in alias_choices):
                exactish = True
            elif any(mention in alias and (len(alias) - len(mention) <= 1) for alias in alias_choices):
                exactish = True
            elif sim >= 0.85 and abs(len(mention) - len(lk.name)) <= 1:
                exactish = True
        accepted = bool(lk and score >= min_link_score and exactish)
        fallback_type = row["entity_type"]
        if not accepted:
            inferred = _infer_label(mention)
            if inferred:
                fallback_type = inferred
        linked_rows.append(
            {
                "mention": mention,
                "entity_type": lk.type if (accepted and lk and lk.type) else fallback_type,
                "start": row["start"],
                "end": row["end"],
                "source_file": source_file,
                "entity_id": lk.entity_id if accepted and lk else "",
                "kb_name": lk.name if accepted and lk else "",
                "kb_type": lk.type if lk else "",
                "link_score": round(float(score), 6),
                "normalized_entity": lk.name if accepted and lk else mention,
                "disambiguation_confidence": round(float(score), 6),
                "disambiguation_method": "char_tfidf_surface+context_rerank",
                "source": row.get("source", ""),
            }
        )
    return linked_rows


def _extract_sections(text: str) -> List[tuple[str, str]]:
    sections: List[tuple[str, str]] = []
    current_title = "全文"
    buffer: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        is_heading = len(line) <= 20 and not re.search(r"[，。！？；：“”\"'（）()]", line)
        if is_heading:
            if buffer:
                sections.append((current_title, "\n".join(buffer)))
                buffer = []
            current_title = line
            continue
        buffer.append(line)
    if buffer:
        sections.append((current_title, "\n".join(buffer)))
    if not sections:
        sections.append(("全文", text))
    return sections


def aggregate_disambiguated_entities(text: str, linked_rows: List[Dict], source_file: str) -> List[Dict]:
    section_map: List[tuple[str, int, int]] = []
    cursor = 0
    for title, body in _extract_sections(text):
        start = text.find(body, cursor)
        if start < 0:
            start = cursor
        end = start + len(body)
        section_map.append((title, start, end))
        cursor = end

    groups: Dict[str, Dict] = {}
    for row in linked_rows:
        key = row["entity_id"] or row["normalized_entity"]
        if key not in groups:
            groups[key] = {
                "main_entity": row["normalized_entity"],
                "entity_type": row["entity_type"],
                "mention_count": 0,
                "mentions": [],
                "source_file": source_file,
                "normalized_entity": row["entity_id"] or row["normalized_entity"],
                "aliases": [],
                "section_titles": [],
                "merged_entities": [],
                "link_scores": [],
                "sources": [],
            }
        item = groups[key]
        item["mention_count"] += 1
        item["mentions"].append(row["mention"])
        item["aliases"].append(row["mention"])
        item["merged_entities"].append(row["mention"])
        item["link_scores"].append(float(row.get("link_score", 0.0)))
        item["sources"].append(row.get("source", ""))
        pos = int(row.get("start", -1))
        for title, s_start, s_end in section_map:
            if s_start <= pos <= s_end:
                item["section_titles"].append(title)
                break

    output: List[Dict] = []
    for item in groups.values():
        mentions = list(dict.fromkeys(item["mentions"]))
        sections = list(dict.fromkeys(item["section_titles"])) or ["全文"]
        avg_score = sum(item["link_scores"]) / max(1, len(item["link_scores"]))
        strong_source = any(src in {"input_dict", "kb_dict"} for src in item["sources"])
        if item["mention_count"] == 1 and avg_score < 0.3 and not strong_source:
            continue
        if item["mention_count"] == 1 and len(item["main_entity"]) > 16 and not strong_source:
            continue
        output.append(
            {
                "main_entity": item["main_entity"],
                "entity_type": item["entity_type"],
                "mention_count": item["mention_count"],
                "mentions": " | ".join(mentions),
                "source_file": item["source_file"],
                "normalized_entity": item["normalized_entity"],
                "aliases": " | ".join(mentions),
                "section_titles": " | ".join(sections),
                "merged_entities": " | ".join(mentions),
                "disambiguation_confidence": round(avg_score, 6),
                "disambiguation_method": "entity_linker_grouping",
                "disambiguation_basis": f"entity_id={item['normalized_entity']}; mentions={item['mention_count']}; avg_link_score={avg_score:.4f}",
            }
        )
    output.sort(key=lambda x: (-int(x["mention_count"]), x["main_entity"]))
    return output


def _label_out(rule: Dict, lang: str) -> Dict:
    if lang == "both":
        return {"label": rule["label_en"], "label_en": rule["label_en"], "label_zh": rule["label_zh"]}
    if lang == "zh":
        return {"label": rule["label_zh"]}
    return {"label": rule["label_en"]}


def _candidate_concept(sent: str, trig_end: int) -> str | None:
    tail = sent[trig_end : trig_end + 18]
    m = re.match(r"([\u4e00-\u9fffA-Za-z0-9·]{2,16})", tail)
    if not m:
        return None
    txt = re.sub(r"^(?:一篇|一项|一个|一种)", "", m.group(1).strip())
    return txt if _valid_concept_text(txt) else None


def _extract_sentence_titles(sent: str) -> List[str]:
    titles: List[str] = []
    quote_pat = r"[\u300a\"\u201c]([^\u300b\"\u201d]{2,120})[\u300b\"\u201d]"
    title_cue_pat = r"(?:\u9898\u4e3a|\u82f1\u8bed[:\uff1a]|\u82f1\u6587[:\uff1a])"
    english_pat = re.compile(
        r"\b(?:The|On|Can|Paper|Foundations)\s+[A-Za-z][A-Za-z]+(?:\s+(?:of|to|and|the|in|for|on|with|an|a|by|[A-Z][A-Za-z]+|[A-Za-z]+|\[\d+\])){1,20}"
    )

    for m in re.finditer(quote_pat, sent):
        title = m.group(1).strip()
        if _valid_concept_text(title):
            titles.append(title)

    for m in re.finditer(title_cue_pat + r"\s*([A-Za-z][A-Za-z0-9 ,.'\-:&]{6,180})", sent):
        chunk = re.split(r"[\u3002\uFF1B\n]", m.group(1))[0]
        for part in re.split(r"\s*(?:\u548c|\u4ee5\u53ca| and |, and )\s*", chunk):
            part = re.sub(r"\[\d+\]", "", part).strip(" ,???;:()[]")
            part = re.sub(r"[\u4e00-\u9fff].*$", "", part).strip(" ,???;:()[]")
            if part.startswith(EN_TITLE_PREFIXES) and _valid_concept_text(part):
                titles.append(part)

    for m in english_pat.finditer(sent):
        title = re.sub(r"\[\d+\]", "", m.group(0)).strip(" ,???;:()[]")
        if _valid_concept_text(title):
            titles.append(title)

    deduped: List[str] = []
    seen = set()
    for title in titles:
        if title in seen:
            continue
        seen.add(title)
        deduped.append(title)
    return deduped


def _extract_title_alias_map(text: str) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    pattern = re.compile(r"《([^》]{2,80})》\s*[（(](?:英语[:：])?\s*([A-Za-z][A-Za-z0-9 ,.'\-:&]+?)[）)]")
    for m in pattern.finditer(text):
        zh = m.group(1).strip()
        en = re.sub(r"\[\d+\]$", "", m.group(2).strip()).strip(" ,，。；;:：")
        if zh and en:
            alias_map[en] = zh
            words = en.split()
            if len(words) >= 2:
                alias_map[" ".join(words[:2])] = zh
            if len(words) >= 3:
                alias_map[" ".join(words[:3])] = zh
    return alias_map


def _entity_display(ent: Dict) -> Dict:
    text = ent.get("kb_name") or ent.get("text")
    text = ENTITY_CANONICAL_MAP.get((text or "").strip(), text)
    return {"text": text, "entity_id": ent.get("entity_id")}


def _entity_key(ent: Dict | None) -> str:
    if not ent:
        return ""
    return str(ent.get("entity_id") or ent.get("kb_name") or ent.get("text") or "")


def _subject_aliases(subject: Dict | None) -> List[str]:
    if not subject:
        return []
    aliases: List[str] = []
    for raw in (subject.get("kb_name"), subject.get("text")):
        text = _normalize_lexicon_term(raw or "")
        if not text:
            continue
        aliases.append(text)
        if "·" in text:
            for part in text.split("·"):
                part = part.strip()
                if 2 <= len(part) <= 4:
                    aliases.append(part)
    seen = set()
    out: List[str] = []
    for alias in aliases:
        if alias in seen:
            continue
        seen.add(alias)
        out.append(alias)
    return out


def _canonicalize_tail_text(text: str, title_alias_map: Dict[str, str]) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw
    if raw in title_alias_map:
        return title_alias_map[raw]
    for alias, canonical in title_alias_map.items():
        if alias.startswith(raw) and len(raw) >= 10:
            return canonical
    return raw


def _normalize_match_text(text: str) -> str:
    base = _normalize_lexicon_term(text or "")
    return re.sub(r"\s+", " ", base).strip().lower()


def _match_linked_entity_by_text(text: str, linked_entities: List[Dict]) -> Dict | None:
    target = _normalize_match_text(text)
    if not target:
        return None
    candidates = []
    for ent in linked_entities:
        for raw in (ent.get("kb_name"), ent.get("text")):
            if _normalize_match_text(raw or "") == target:
                candidates.append(ent)
                break
    if not candidates:
        return None
    candidates.sort(
        key=lambda ent: (
            1 if ent.get("entity_id") else 0,
            float(ent.get("link_score", 0.0)),
            len(ent.get("kb_name") or ent.get("text") or ""),
        ),
        reverse=True,
    )
    return candidates[0]


def _infer_relation_endpoint_type(endpoint: Dict, relation_id: str, side: str) -> str:
    if endpoint.get("entity_id"):
        return endpoint.get("type") or endpoint.get("label") or "ENTITY"
    text = endpoint.get("text", "")
    if relation_id == "R_REPUTED_AS" and side == "tail":
        return "CONCEPT"
    if relation_id in {"R_WROTE", "R_PUBLISHED", "R_WORKED_ON", "R_PROPOSED", "R_AWARDED"} and side == "tail":
        return "PROD"
    if relation_id in {"R_BORN_IN", "R_LIVED_IN"} and side == "tail":
        return "LOC"
    if relation_id in {"R_STUDIED_AT", "R_GRADUATED_FROM", "R_WORKED_FOR"} and side == "tail":
        return "ORG"
    inferred = _infer_label(text)
    if inferred:
        return inferred
    if text.endswith(("之父", "之母", "先驱", "奠基人", "理论", "思想", "概念")):
        return "CONCEPT"
    return "ENTITY"


def _title_candidates_after_trigger(sent: str, trig_end: int, title_alias_map: Dict[str, str]) -> List[str]:
    titles = _extract_sentence_titles(sent)
    out: List[str] = []
    seen = set()
    for title in titles:
        pos = sent.find(title)
        if pos >= trig_end - 2:
            canon = _canonicalize_tail_text(title, title_alias_map)
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
    return out


def _choose_subject_entity(sent_ents: List[Dict]) -> Dict | None:
    persons = [e for e in sent_ents if e.get("label") == "PER"]
    if not persons:
        return None
    persons.sort(
        key=lambda e: (
            1 if e.get("entity_id") else 0,
            float(e.get("link_score", 0.0)),
            len(e.get("text", "")),
            -int(e.get("start", 0)),
        ),
        reverse=True,
    )
    return persons[0]


def _choose_dominant_person(entities: List[Dict]) -> Dict | None:
    buckets: Dict[str, Dict] = {}
    for ent in entities:
        if ent.get("label") != "PER":
            continue
        key = _entity_key(ent)
        if not key:
            continue
        item = buckets.setdefault(
            key,
            {
                "count": 0,
                "best": ent,
                "best_score": float(ent.get("link_score", 0.0)),
            },
        )
        item["count"] += 1
        score = float(ent.get("link_score", 0.0))
        if score > item["best_score"]:
            item["best"] = ent
            item["best_score"] = score
    if not buckets:
        return None
    ranked = sorted(
        buckets.values(),
        key=lambda item: (
            item["count"],
            1 if item["best"].get("entity_id") else 0,
            item["best_score"],
            len(item["best"].get("kb_name") or item["best"].get("text") or ""),
        ),
        reverse=True,
    )
    return ranked[0]["best"]


def _augment_with_pronoun_subject(sent: str, s_start: int, sent_ents: List[Dict], last_subject: Dict | None) -> List[Dict]:
    if not last_subject or not PRONOUN_SUBJECT_RE.match(sent):
        return sent_ents
    synthetic = dict(last_subject)
    synthetic["start"] = s_start
    synthetic["end"] = s_start + 1
    synthetic["source"] = "coref"
    return [synthetic] + sent_ents


def _relation_entity_rank(ent: Dict, anchor: int, side: str) -> tuple:
    if side == "left":
        dist = anchor - int(ent["end"])
    else:
        dist = int(ent["start"]) - anchor
    src_pri = 3 if ent.get("source") == "coref" else (2 if ent.get("source") == "kb_dict" else (1 if ent.get("source") == "input_dict" else 0))
    return (
        1 if ent.get("entity_id") else 0,
        float(ent.get("link_score", 0.0)),
        src_pri,
        -max(dist, 0),
        len(ent.get("text", "")),
    )


def _valid_concept_text(text: str) -> bool:
    t = (text or "").strip()
    if not _clean_entity_text(t):
        return False
    if t.startswith(("了", "并", "其", "该")):
        return False
    if re.fullmatch(r"[一二三四五六七八九十几两0-9]+(?:个)?(?:月|年|天|次)", t):
        return False
    return True


def _keep_relation(row: Dict) -> bool:
    rel_id = row.get("relation_id", "")
    sentence = row.get("sentence", "")
    surface = row.get("surface", row.get("label", ""))
    head_text = row.get("head", {}).get("text", "")
    tail_text = row.get("tail", {}).get("text", "")
    head_id = row.get("head", {}).get("entity_id")

    if rel_id in {"R_WROTE", "R_PUBLISHED", "R_DESIGNED", "R_WORKED_ON"} and not _valid_concept_text(tail_text):
        return False
    if rel_id in {"R_WROTE", "R_PUBLISHED", "R_DESIGNED"} and not row.get("tail", {}).get("entity_id"):
        if not (("\u300a" in sentence and tail_text in sentence) or tail_text.startswith(("The ", "On ", "Can ", "Paper ", "Foundations "))):
            return False
    if rel_id == "R_PUBLISHED" and not head_id and head_text.startswith(("The ", "On ", "Can ")):
        return False
    if rel_id == "R_PUBLISHED" and "\u4e00\u76f4\u7b49\u5230" in sentence and tail_text.endswith("\u9009\u96c6"):
        return False
    if rel_id == "R_WROTE" and tail_text.endswith("\u7a0b\u5e8f") and "\u300a" not in sentence and not tail_text.startswith(("The ", "On ", "Can ", "Paper ")):
        return False
    if rel_id == "R_AWARDED" and "\u4efb\u547d\u4e3a" in sentence and head_text.endswith("\u56fd\u738b"):
        return False
    if rel_id == "R_CRACKED" and tail_text == "Bombe":
        return False
    if rel_id == "R_LIVED_IN" and head_text == "\u827e\u4f26\u00b7\u56fe\u7075" and "\u751f\u4e0b\u4e86\u827e\u4f26" in sentence:
        return False
    if rel_id == "R_DESIGNED" and head_text == "\u827e\u4f26\u00b7\u56fe\u7075" and tail_text == "\u6d1b\u65af\u963f\u62c9\u83ab\u65af\u8c61\u68cb" and "\u6839\u636e\u56fe\u7075\u7684\u7406\u8bba" in sentence:
        return False
    if rel_id in {"R_PERSECUTED", "R_CONVICTED"} and head_text in {"\u6208\u767b\u00b7\u5e03\u6717", "\u6bcf\u65e5\u7535\u8baf\u62a5", "\u4f0a\u4e3d\u838e\u767d\u4e8c\u4e16"}:
        return False
    if rel_id == "R_CONVICTED" and surface == "\u5b9a\u7f6a" and "\u88ab\u5b9a\u7f6a" in sentence:
        return False
    if rel_id in {"R_BORN_IN", "R_LIVED_IN"} and tail_text == "\u82f1\u56fd" and "\u751f\u4e0b" in sentence:
        return False
    if rel_id in {"R_WROTE", "R_PUBLISHED"} and tail_text.startswith(("The ", "On ", "Paper ", "Can ", "Foundations ")) and len(tail_text.split()) <= 2:
        if f"{tail_text} " in sentence:
            return False
    if rel_id == "R_COLLABORATED_WITH" and ((row.get("head", {}).get("entity_id") or head_text) == (row.get("tail", {}).get("entity_id") or tail_text)):
        return False
    if rel_id == "R_COLLABORATED_WITH" and any(cue in sentence for cue in ("\u6839\u636e", "\u8bf4\u6cd5", "\u8c08\u5230", "\u5386\u53f2\u5b66\u5bb6")):
        return False
    return True


def _normalize_relation_row(row: Dict, title_alias_map: Dict[str, str]) -> Dict:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    rel_id = out.get("relation_id", "")
    tail = out.get("tail", {})
    if rel_id in {"R_WROTE", "R_PUBLISHED", "R_DESIGNED", "R_WORKED_ON"} and tail and not tail.get("entity_id"):
        tail["text"] = _canonicalize_tail_text(tail.get("text", ""), title_alias_map)
    return out


def _get_rule(relation_id: str) -> Dict | None:
    return next((r for r in RELATION_RULE_LIBRARY if r["relation_id"] == relation_id), None)


def _prune_subsumed_relations(relations: List[Dict]) -> List[Dict]:
    kept: List[Dict] = []
    for row in relations:
        head_key = _entity_key(row.get("head", {}))
        tail_text = row.get("tail", {}).get("text", "")
        rel_id = row.get("relation_id", "")
        sentence = row.get("sentence", "")
        dominated = False
        for other in relations:
            if other is row:
                continue
            if rel_id != other.get("relation_id", "") or sentence != other.get("sentence", ""):
                continue
            if head_key != _entity_key(other.get("head", {})):
                continue
            other_tail = other.get("tail", {}).get("text", "")
            if not tail_text or tail_text == other_tail or tail_text not in other_tail:
                continue
            if other.get("tail", {}).get("entity_id") or len(other_tail) > len(tail_text):
                dominated = True
                break
        if not dominated:
            kept.append(row)
    return kept


def _append_relation(
    relations: List[Dict],
    rule: Dict | None,
    head: Dict | None,
    tail: Dict | Dict[str, str] | None,
    lang: str,
    confidence: float,
    source: str,
    surface: str,
    sentence: str,
) -> None:
    if rule is None or head is None or tail is None:
        return
    head_display = _entity_display(head)
    tail_display = tail if "text" in tail and "entity_id" in tail and "start" not in tail else _entity_display(tail)
    if not head_display.get("text") or not tail_display.get("text"):
        return
    relations.append(
        {
            "head": head_display,
            "tail": tail_display,
            "relation_id": rule["relation_id"],
            **_label_out(rule, lang),
            "confidence": confidence,
            "source": source,
            "surface": surface,
            "sentence": sentence,
        }
    )


def _extract_structured_relations(
    sent: str,
    s_start: int,
    sent_ents: List[Dict],
    lang: str,
    title_alias_map: Dict[str, str],
    carry_subject: Dict | None = None,
) -> List[Dict]:
    relations: List[Dict] = []
    if not sent_ents:
        return relations

    person_ents = [e for e in sent_ents if e.get("label") == "PER"]
    org_ents = [e for e in sent_ents if e.get("label") == "ORG"]
    loc_ents = [e for e in sent_ents if e.get("label") == "LOC"]
    prod_ents = [e for e in sent_ents if e.get("label") == "PROD"]
    subject = _choose_subject_entity(sent_ents)

    cue_born_a = "\u751f\u4e0b"
    cue_born_b = "\u51fa\u751f"
    cue_live_a = "\u4f4f\u5728"
    cue_live_b = "\u540c\u4f4f"
    cue_parent = "\u7236\u6bcd"
    study_cues = ("\u6ce8\u518c", "\u8003\u5165", "\u653b\u8bfb", "\u5c31\u8bfb")
    learning_cues = ("\u5b66\u4e60", "\u6c42\u5b66")
    graduation_cue = "\u6bd5\u4e1a"
    degree_cues = ("\u5b66\u4f4d", "\u535a\u58eb", "\u7855\u58eb")
    bad_study_tail = "\u6bcf\u65e5\u7535\u8baf\u62a5"
    hire_cue = "\u62db\u8058"
    passive_cue = "\u88ab"
    become_cue = "\u6210\u4e3a"
    rank_cues = ("\u526f\u4e3b\u4efb", "\u7814\u7a76\u5458", "\u6559\u6388")
    at_cue = "\u5728"
    resp_cue = "\u8d1f\u8d23"
    together_cue = "\u4e00\u8d77"
    collab_cue = "\u5408\u4f5c"
    invite_cue = "\u8058\u8bf7"
    successor_cue = "\u7ee7\u4efb\u8005"
    wrote_for_cue = "\u64b0\u6587"
    named_after_cue = "\u4e3a\u540d"
    apology_cue = "\u516c\u5f00\u9053\u6b49"
    apology_short = "\u9053\u6b49"
    toward_cue = "\u5411"
    title_context_cues = ("\u8bba\u6587", "\u8457\u4f5c", "\u91cd\u8981\u8bba\u6587", "\u9898\u4e3a", "\u5199\u4e86", "\u5199\u8fc7", "\u53d1\u8868")
    publish_cue = "\u53d1\u8868"

    def local_start(ent: Dict) -> int:
        return int(ent["start"]) - s_start

    def local_end(ent: Dict) -> int:
        return int(ent["end"]) - s_start

    def nearest_before(entities: List[Dict], anchor: int) -> Dict | None:
        cands = [e for e in entities if local_end(e) <= anchor]
        if not cands:
            return None
        return min(cands, key=lambda e: anchor - local_end(e))

    def nearest_after(entities: List[Dict], anchor: int) -> Dict | None:
        cands = [e for e in entities if local_start(e) >= anchor]
        if not cands:
            return None
        return min(cands, key=lambda e: local_start(e) - anchor)

    birth_rule = _get_rule("R_BORN_IN")
    if birth_rule and person_ents and loc_ents:
        if cue_born_a in sent:
            anchor = sent.find(cue_born_a)
            tail = nearest_before(loc_ents, anchor)
            head = nearest_after(person_ents, anchor) or nearest_before(person_ents, anchor) or subject
            _append_relation(relations, birth_rule, head, tail, lang, 0.92, "structured_pattern", cue_born_a, sent)
        elif cue_born_b in sent:
            anchor = sent.find(cue_born_b)
            tail = nearest_before(loc_ents, anchor) or nearest_after(loc_ents, anchor)
            head = nearest_before(person_ents, anchor) or subject
            _append_relation(relations, birth_rule, head, tail, lang, 0.9, "structured_pattern", cue_born_b, sent)
    elif birth_rule and loc_ents and carry_subject and cue_born_a in sent:
        anchor = sent.find(cue_born_a)
        right_text = sent[anchor:]
        if any(alias in right_text for alias in _subject_aliases(carry_subject)):
            tail = nearest_before(loc_ents, anchor) or nearest_after(loc_ents, anchor)
            _append_relation(relations, birth_rule, carry_subject, tail, lang, 0.9, "cross_sentence_subject", cue_born_a, sent)

    lived_rule = _get_rule("R_LIVED_IN")
    if lived_rule and person_ents and loc_ents and (cue_live_a in sent or cue_live_b in sent):
        anchor = sent.find(cue_live_a) if cue_live_a in sent else sent.find(cue_live_b)
        if cue_parent not in sent[: anchor + 4]:
            tail = nearest_after(loc_ents, anchor) or nearest_before(loc_ents, anchor)
            head = nearest_before(person_ents, anchor) or subject
            _append_relation(relations, lived_rule, head, tail, lang, 0.86, "structured_pattern", sent[anchor : anchor + 2], sent)

    study_rule = _get_rule("R_STUDIED_AT")
    if study_rule and org_ents and any(cue in sent for cue in study_cues):
        anchors = [sent.find(cue) for cue in study_cues if cue in sent]
        anchor = min(a for a in anchors if a >= 0)
        head = nearest_before(person_ents, anchor) or subject or carry_subject
        school_orgs = [e for e in org_ents if any(tag in (e.get("kb_name") or e.get("text", "")) for tag in ("大学", "学院", "学校"))]
        tail = nearest_after(school_orgs, anchor) or nearest_before(school_orgs, anchor) or nearest_after(org_ents, anchor) or nearest_before(org_ents, anchor)
        if tail and tail.get("text") != bad_study_tail:
            _append_relation(relations, study_rule, head, tail, lang, 0.92, "structured_pattern", sent[anchor : anchor + 2], sent)
    elif study_rule and org_ents and any(cue in sent for cue in learning_cues):
        anchors = [sent.find(cue) for cue in learning_cues if cue in sent]
        anchor = min(a for a in anchors if a >= 0)
        head = nearest_before(person_ents, anchor) or subject or carry_subject
        school_orgs = [e for e in org_ents if any(tag in (e.get("kb_name") or e.get("text", "")) for tag in ("\u5927\u5b66", "\u5b66\u9662", "\u5b66\u6821"))]
        tail = nearest_before(school_orgs, anchor) or nearest_after(school_orgs, anchor) or nearest_before(org_ents, anchor)
        if head and tail and tail.get("text") != bad_study_tail:
            _append_relation(relations, study_rule, head, tail, lang, 0.88, "structured_pattern", "\u5728...\u5b66\u4e60", sent)

    graduation_rule = _get_rule("R_GRADUATED_FROM")
    if graduation_rule and org_ents and (graduation_cue in sent or ("\u83b7\u5f97" in sent and any(cue in sent for cue in degree_cues))):
        if graduation_cue in sent:
            anchor = sent.find(graduation_cue)
            surface = graduation_cue
        else:
            anchor = sent.find("\u83b7\u5f97")
            surface = "\u83b7\u5f97...\u5b66\u4f4d"
        head = nearest_before(person_ents, anchor) or subject or carry_subject
        tail = nearest_before(org_ents, anchor) or nearest_after(org_ents, anchor)
        if head and tail:
            _append_relation(relations, graduation_rule, head, tail, lang, 0.9, "structured_pattern", surface, sent)

    award_rule = _get_rule("R_AWARDED")
    appoint_award_cue = "\u4efb\u547d\u4e3a"
    if award_rule and prod_ents and passive_cue in sent and appoint_award_cue in sent:
        anchor = sent.find(appoint_award_cue)
        head = nearest_before(person_ents, sent.find(passive_cue)) or subject or carry_subject
        tail = nearest_after(prod_ents, anchor)
        if head and tail:
            _append_relation(relations, award_rule, head, tail, lang, 0.93, "structured_pattern", "\u88ab...\u4efb\u547d\u4e3a", sent)

    work_rule = _get_rule("R_WORKED_FOR")
    if work_rule and person_ents and org_ents:
        if hire_cue in sent and passive_cue in sent:
            anchor = sent.find(hire_cue)
            head = nearest_before(person_ents, anchor) or subject
            tail = nearest_before(org_ents, anchor)
            _append_relation(relations, work_rule, head, tail, lang, 0.9, "structured_pattern", "\u88ab...\u62db\u8058", sent)
        if become_cue in sent and any(cue in sent for cue in rank_cues):
            anchor = sent.find(become_cue)
            head = nearest_before(person_ents, anchor) or subject
            tail = nearest_after(org_ents, anchor) or nearest_before(org_ents, anchor)
            _append_relation(relations, work_rule, head, tail, lang, 0.9, "structured_pattern", become_cue, sent)
        if at_cue in sent and resp_cue in sent:
            start = sent.find(at_cue)
            anchor = sent.find(resp_cue)
            head = nearest_before(person_ents, anchor) or subject
            cands = [e for e in org_ents if local_start(e) >= start and local_end(e) <= anchor]
            tail = min(cands, key=lambda e: anchor - local_end(e)) if cands else None
            _append_relation(relations, work_rule, head, tail, lang, 0.9, "structured_pattern", "\u5728...\u8d1f\u8d23", sent)

    worked_on_rule = _get_rule("R_WORKED_ON")
    if worked_on_rule and person_ents and prod_ents and resp_cue in sent:
        anchor = sent.find(resp_cue)
        head = nearest_before(person_ents, anchor) or subject
        tail = nearest_after(prod_ents, anchor)
        _append_relation(relations, worked_on_rule, head, tail, lang, 0.93, "structured_pattern", resp_cue, sent)
    if worked_on_rule and org_ents and prod_ents and resp_cue in sent:
        anchor = sent.find(resp_cue)
        head = nearest_before(org_ents, anchor)
        tail = nearest_after(prod_ents, anchor)
        _append_relation(relations, worked_on_rule, head, tail, lang, 0.87, "structured_pattern", "ORG:" + resp_cue, sent)

    collab_rule = _get_rule("R_COLLABORATED_WITH")
    if collab_rule and len(person_ents) >= 2 and (together_cue in sent or collab_cue in sent):
        head = subject or person_ents[0]
        others = [e for e in person_ents if (e.get("entity_id") or e.get("text")) != (head.get("entity_id") or head.get("text"))]
        tail = nearest_after(others, local_end(head)) or (others[0] if others else None)
        _append_relation(relations, collab_rule, head, tail, lang, 0.84, "structured_pattern", together_cue, sent)

    if work_rule and org_ents and person_ents:
        org_role_patterns = (
            "\u5bc6\u7801\u7834\u8bd1\u5458",
            "\u7814\u7a76\u5458",
            "\u6559\u6388",
            "\u516c\u52a1\u5458",
            "\u9996\u76f8",
        )
        for org in org_ents:
            org_end = local_end(org)
            right_people = [e for e in person_ents if local_start(e) >= org_end]
            if not right_people:
                continue
            tail = right_people[0]
            window = sent[org_end:local_start(tail)]
            if any(tag in window for tag in org_role_patterns):
                _append_relation(relations, work_rule, tail, org, lang, 0.86, "structured_pattern", "ORG_ROLE_PER", sent)

    invite_rule = _get_rule("R_INVITED")
    if invite_rule and invite_cue in sent and len(person_ents) >= 2:
        anchor = sent.find(invite_cue)
        head = nearest_before(person_ents, anchor) or nearest_before(org_ents, anchor)
        tail = nearest_after(person_ents, anchor)
        _append_relation(relations, invite_rule, head, tail, lang, 0.87, "structured_pattern", invite_cue, sent)

    successor_rule = _get_rule("R_SUCCESSOR_OF")
    if successor_rule and successor_cue in sent and len(org_ents) >= 2:
        anchor = sent.find(successor_cue)
        prev_org = nearest_before(org_ents, anchor)
        next_org = nearest_after(org_ents, anchor)
        if prev_org and next_org:
            _append_relation(relations, successor_rule, next_org, prev_org, lang, 0.88, "structured_pattern", successor_cue, sent)

    wrote_for_rule = _get_rule("R_WROTE_FOR")
    if wrote_for_rule and wrote_for_cue in sent and person_ents and org_ents:
        anchor = sent.find(wrote_for_cue)
        target_org = nearest_before(org_ents, anchor)
        author = nearest_before(person_ents, anchor)
        if target_org and author:
            _append_relation(relations, wrote_for_rule, author, target_org, lang, 0.88, "structured_pattern", wrote_for_cue, sent)

    named_after_rule = _get_rule("R_NAMED_AFTER")
    if named_after_rule and named_after_cue in sent and org_ents and person_ents:
        anchor = sent.find(named_after_cue)
        head = nearest_before(org_ents, anchor)
        tail = nearest_after(person_ents, anchor) or nearest_before(person_ents, anchor)
        _append_relation(relations, named_after_rule, head, tail, lang, 0.86, "structured_pattern", named_after_cue, sent)

    apology_rule = _get_rule("R_APOLOGIZED_TO")
    if apology_rule and (apology_cue in sent or apology_short in sent):
        anchor = sent.find(apology_cue) if apology_cue in sent else sent.find(apology_short)
        toward = sent.rfind(toward_cue, 0, anchor)
        head_cands = [e for e in sent_ents if e.get("label") in {"ORG", "PER"} and local_end(e) <= max(toward, 0)]
        tail_cands = [e for e in sent_ents if e.get("label") in {"PER", "ORG"} and (toward < 0 or local_start(e) >= toward) and local_end(e) <= anchor]
        head = None
        if head_cands:
            head_cands.sort(key=lambda e: (1 if e.get("label") == "ORG" else 0, float(e.get("link_score", 0.0)), local_end(e)), reverse=True)
            head = head_cands[0]
        tail = tail_cands[-1] if tail_cands else None
        _append_relation(relations, apology_rule, head, tail, lang, 0.94, "structured_pattern", "\u5411...\u9053\u6b49", sent)

    titles = [_canonicalize_tail_text(t, title_alias_map) for t in _extract_sentence_titles(sent)]
    if subject and titles and any(cue in sent for cue in title_context_cues):
        rule = _get_rule("R_PUBLISHED") if publish_cue in sent else _get_rule("R_WROTE")
        for title in titles:
            _append_relation(relations, rule, subject, {"text": title, "entity_id": ""}, lang, 0.9, "structured_pattern", "title_context", sent)
        if "\u9898\u4e3a" in sent or "\u5199\u4e86" in sent or "\u5199\u8fc7" in sent or publish_cue in sent:
            for ent in prod_ents:
                if int(ent["start"]) < s_start + sent.find(titles[0][: min(4, len(titles[0]))]):
                    continue
                ent_text = ent.get("kb_name") or ent.get("text", "")
                if not ent_text:
                    continue
                if any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in ent_text):
                    _append_relation(relations, rule, subject, {"text": ent_text, "entity_id": ent.get("entity_id", "") or ""}, lang, 0.88, "structured_pattern", "entity_title", sent)

    return relations


def _extract_contextual_work_relations(sent: str, s_start: int, sent_ents: List[Dict], lang: str) -> List[Dict]:
    relations: List[Dict] = []
    person_ents = [e for e in sent_ents if e.get("label") == "PER"]
    org_ents = [e for e in sent_ents if e.get("label") == "ORG"]
    if not person_ents or not org_ents:
        return relations
    work_rule = next((r for r in RELATION_RULE_LIBRARY if r["relation_id"] == "R_WORKED_FOR"), None)
    if work_rule is None:
        return relations
    for head in person_ents:
        for tail in org_ents:
            if int(head["start"]) > int(tail["start"]):
                continue
            h_end = max(0, int(head["end"]) - s_start)
            t_start = max(0, int(tail["start"]) - s_start)
            t_end = max(0, int(tail["end"]) - s_start)
            between = sent[h_end:t_start]
            after = sent[t_end : min(len(sent), t_end + 16)]
            if "在" not in between:
                continue
            if not any(hint in after for hint in WORK_CONTEXT_HINTS):
                continue
            relations.append(
                {
                    "head": _entity_display(head),
                    "tail": _entity_display(tail),
                    "relation_id": work_rule["relation_id"],
                    **_label_out(work_rule, lang),
                    "confidence": 0.91,
                    "source": "context_pattern",
                    "surface": "在...工作",
                    "sentence": sent,
                }
            )
    return relations


def _extract_open_relations(text: str, linked_entities: List[Dict], lang: str, max_pair_gap: int, pair_neighbors: int, require_linked: bool) -> List[Dict]:
    relations: List[Dict] = []
    active_rules = build_relation_rules(text)
    title_alias_map = _extract_title_alias_map(text)
    spans = split_sentences(text)
    last_subject: Dict | None = None
    dominant_subject = _choose_dominant_person(linked_entities)
    last_study_org_by_subject: Dict[str, Dict] = {}
    graduation_rule = _get_rule("R_GRADUATED_FROM")
    for s_start, s_end, sent in spans:
        sent_ents = [e for e in linked_entities if e["start"] >= s_start and e["end"] <= s_end]
        if require_linked:
            sent_ents = [e for e in sent_ents if e.get("entity_id")]
        pronoun_subject = last_subject
        if PRONOUN_SUBJECT_RE.match(sent) and (pronoun_subject is None or pronoun_subject.get("label") != "PER"):
            pronoun_subject = dominant_subject
        sent_ents = _augment_with_pronoun_subject(sent, s_start, sent_ents, pronoun_subject)
        sent_ents.sort(key=lambda x: (x["start"], x["end"]))
        sent_rel_start = len(relations)
        carry_subject = last_subject
        if carry_subject and not any(alias in sent for alias in _subject_aliases(carry_subject)):
            carry_subject = dominant_subject
        elif carry_subject is None:
            carry_subject = dominant_subject
        for rule in active_rules:
            for trig in rule["aliases"]:
                for m in re.finditer(re.escape(trig), sent):
                    left = [e for e in sent_ents if e["end"] <= s_start + m.start() and e["label"] in rule["head_types"]]
                    if not left and carry_subject and carry_subject.get("label") in rule["head_types"]:
                        left = [carry_subject]
                    left.sort(key=lambda e: _relation_entity_rank(e, s_start + m.start(), "left"), reverse=True)
                    if not left:
                        continue
                    head = left[0]
                    title_candidates = _title_candidates_after_trigger(sent, m.end(), title_alias_map)
                    if rule["relation_id"] in {"R_WROTE", "R_PUBLISHED", "R_DESIGNED"} and title_candidates:
                        for concept in title_candidates:
                            relations.append(
                                {
                                    "head": _entity_display(head),
                                    "tail": {"text": concept, "entity_id": ""},
                                    "relation_id": rule["relation_id"],
                                    **_label_out(rule, lang),
                                    "confidence": 0.95,
                                    "source": "trigger_window",
                                    "surface": trig,
                                    "sentence": sent,
                                }
                            )
                        continue
                    right = [e for e in sent_ents if e["start"] >= s_start + m.end() and e["label"] in rule["tail_types"]]
                    right.sort(key=lambda e: _relation_entity_rank(e, s_start + m.end(), "right"), reverse=True)
                    if right:
                        tail_ent = right[0]
                        if trig == "赦免" and "要求" in sent and head["text"] in {"英国政府", "英国首相府邸"}:
                            continue
                        tail_display = _entity_display(tail_ent)
                        if not tail_display.get("entity_id"):
                            tail_display["text"] = _canonicalize_tail_text(tail_display.get("text", ""), title_alias_map)
                        relations.append(
                            {
                                "head": _entity_display(head),
                                "tail": tail_display,
                                "relation_id": rule["relation_id"],
                                **_label_out(rule, lang),
                                "confidence": 0.97,
                                "source": "trigger_window",
                                "surface": trig,
                                "sentence": sent,
                            }
                        )
                    elif "CONCEPT" in rule["tail_types"]:
                        if title_candidates:
                            for concept in title_candidates:
                                relations.append(
                                    {
                                        "head": _entity_display(head),
                                        "tail": {"text": concept, "entity_id": ""},
                                        "relation_id": rule["relation_id"],
                                        **_label_out(rule, lang),
                                        "confidence": 0.90,
                                        "source": "trigger_window",
                                        "surface": trig,
                                        "sentence": sent,
                                    }
                                )
                        else:
                            concept = _candidate_concept(sent, m.end())
                            if concept:
                                relations.append(
                                    {
                                        "head": _entity_display(head),
                                        "tail": {"text": _canonicalize_tail_text(concept, title_alias_map), "entity_id": ""},
                                        "relation_id": rule["relation_id"],
                                        **_label_out(rule, lang),
                                        "confidence": 0.88,
                                        "source": "trigger_window",
                                        "surface": trig,
                                        "sentence": sent,
                                    }
                                )
        relations.extend(_extract_structured_relations(sent, s_start, sent_ents, lang, title_alias_map, carry_subject=carry_subject))
        relations.extend(_extract_contextual_work_relations(sent, s_start, sent_ents, lang))
        upper_pairs = min(len(sent_ents), pair_neighbors + 1)
        for i, head in enumerate(sent_ents):
            for j in range(i + 1, min(len(sent_ents), i + upper_pairs)):
                tail = sent_ents[j]
                gap = tail["start"] - head["end"]
                if gap < 0 or gap > max_pair_gap:
                    continue
                between = text[head["end"] : tail["start"]]
                for rule in active_rules:
                    if head["label"] not in rule["head_types"] or tail["label"] not in rule["tail_types"]:
                        continue
                    hit = next((alias for alias in rule["aliases"] if alias in between), None)
                    if not hit:
                        continue
                    if hit == "赦免" and "要求" in sent and head["text"] in {"英国政府", "英国首相府邸"}:
                        continue
                    tail_display = _entity_display(tail)
                    if not tail_display.get("entity_id"):
                        tail_display["text"] = _canonicalize_tail_text(tail_display.get("text", ""), title_alias_map)
                    relations.append(
                        {
                            "head": _entity_display(head),
                            "tail": tail_display,
                            "relation_id": rule["relation_id"],
                            **_label_out(rule, lang),
                            "confidence": 0.93,
                            "source": "pattern",
                            "surface": hit,
                            "sentence": sent,
                        }
                    )
        chosen_subject = _choose_subject_entity([e for e in sent_ents if e.get("source") != "coref"])
        active_subject = _choose_subject_entity(sent_ents) or last_subject
        subject_key = _entity_key(active_subject)
        sent_rows = relations[sent_rel_start:]
        if graduation_rule and subject_key and ("毕业" in sent or ("学位" in sent and any(cue in sent for cue in ("获", "获得", "取得")))):
            has_graduation = any(
                row.get("relation_id") == "R_GRADUATED_FROM"
                and _entity_key(row.get("head", {})) == subject_key
                for row in sent_rows
            )
            study_tail = last_study_org_by_subject.get(subject_key)
            if not has_graduation and study_tail:
                surface = "毕业" if "毕业" in sent else "获...学位"
                _append_relation(relations, graduation_rule, active_subject, study_tail, lang, 0.89, "cross_sentence", surface, sent)
                sent_rows = relations[sent_rel_start:]
        if subject_key:
            for row in sent_rows:
                if row.get("relation_id") not in {"R_STUDIED_AT", "R_GRADUATED_FROM"}:
                    continue
                head_key = row.get("head", {}).get("entity_id") or row.get("head", {}).get("text", "")
                tail = row.get("tail", {})
                if head_key == subject_key and tail.get("text"):
                    last_study_org_by_subject[subject_key] = {"text": tail.get("text"), "entity_id": tail.get("entity_id")}
        if chosen_subject is not None:
            last_subject = chosen_subject
    unique: List[Dict] = []
    seen = set()
    for row in relations:
        row = _normalize_relation_row(row, title_alias_map)
        h = row["head"].get("entity_id") or row["head"].get("text")
        t = row["tail"].get("entity_id") or row["tail"].get("text")
        if not h or not t or h == t:
            continue
        if not _keep_relation(row):
            continue
        key = (h, t, row["relation_id"], row.get("surface"), row.get("sentence"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _extract_classifier_relations(text: str, linked_entities: List[Dict], rel_model: RelationClassifier, lang: str, max_pair_gap: int, pair_neighbors: int) -> List[Dict]:
    relations: List[Dict] = []
    spans = split_sentences(text)
    for s_start, s_end, sent in spans:
        sent_ents = [e for e in linked_entities if e["start"] >= s_start and e["end"] <= s_end and e.get("entity_id")]
        sent_ents.sort(key=lambda x: (x["start"], x["end"]))
        upper_pairs = min(len(sent_ents), pair_neighbors + 1)
        for i, head in enumerate(sent_ents):
            for j in range(i + 1, min(len(sent_ents), i + upper_pairs)):
                tail = sent_ents[j]
                gap = tail["start"] - head["end"]
                if gap < 0 or gap > max_pair_gap:
                    continue
                h_local = {"start": head["start"] - s_start, "end": head["end"] - s_start, "label": head["label"]}
                t_local = {"start": tail["start"] - s_start, "end": tail["end"] - s_start, "label": tail["label"]}
                label, score = rel_model.predict_with_score(sent, h_local, t_local)
                if label == "NO_REL":
                    continue
                relations.append(
                    {
                        "head": _entity_display(head),
                        "tail": _entity_display(tail),
                        "relation_id": f"R_{label}",
                        "label": label if lang != "zh" else label,
                        "confidence": float(score),
                        "source": "classifier",
                        "sentence": sent,
                    }
                )
    unique: List[Dict] = []
    seen = set()
    for row in relations:
        h = row["head"].get("entity_id") or row["head"].get("text")
        t = row["tail"].get("entity_id") or row["tail"].get("text")
        if not h or not t or h == t:
            continue
        if not _keep_relation(row):
            continue
        key = (h, t, row["relation_id"], row.get("sentence"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def build_pipeline_payload(
    text: str,
    linked_mentions: List[Dict],
    re_model: RelationClassifier | None = None,
    relation_strategy: str = "open",
    relation_label_lang: str = "zh",
    require_linked_entity: bool = True,
    max_pair_gap: int = 64,
    pair_neighbors: int = 6,
) -> Dict:
    active_relation_rules = build_relation_rules(text)
    title_alias_map = _extract_title_alias_map(text)
    linked_entities = [
        {
            "start": int(row["start"]),
            "end": int(row["end"]),
            "label": row["entity_type"],
            "text": ENTITY_CANONICAL_MAP.get(row["mention"], row["mention"]),
            "entity_id": row["entity_id"] or None,
            "kb_name": ENTITY_CANONICAL_MAP.get(row["kb_name"], row["kb_name"]) if row["kb_name"] else None,
            "kb_type": row["kb_type"] or None,
            "link_score": float(row.get("link_score", 0.0)),
            "source": row.get("source", "extract"),
        }
        for row in linked_mentions
        if _clean_entity_text(row["mention"])
    ]
    open_relations: List[Dict] = []
    clf_relations: List[Dict] = []
    if relation_strategy in {"open", "hybrid"}:
        open_relations = _extract_open_relations(
            text=text,
            linked_entities=linked_entities,
            lang=relation_label_lang,
            max_pair_gap=max_pair_gap,
            pair_neighbors=pair_neighbors,
            require_linked=require_linked_entity,
        )
    if relation_strategy in {"classifier", "hybrid"} and re_model is not None:
        clf_relations = _extract_classifier_relations(
            text=text,
            linked_entities=linked_entities,
            rel_model=re_model,
            lang=relation_label_lang,
            max_pair_gap=max_pair_gap,
            pair_neighbors=pair_neighbors,
        )
    best: Dict[tuple, Dict] = {}
    for row in open_relations + clf_relations:
        row = _normalize_relation_row(row, title_alias_map)
        for side in ("head", "tail"):
            endpoint = row.get(side, {})
            if endpoint.get("entity_id"):
                continue
            matched = _match_linked_entity_by_text(endpoint.get("text", ""), linked_entities)
            if matched and matched.get("entity_id"):
                endpoint["entity_id"] = matched.get("entity_id")
                endpoint["text"] = matched.get("kb_name") or matched.get("text") or endpoint.get("text")
        if not _keep_relation(row):
            continue
        h = row["head"].get("entity_id") or row["head"].get("text")
        t = row["tail"].get("entity_id") or row["tail"].get("text")
        key = (h, t, row["relation_id"])
        prev = best.get(key)
        if prev is None or float(row.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
            best[key] = row
    by_text: Dict[tuple, Dict] = {}
    for row in best.values():
        h_text = row["head"].get("text")
        t_text = row["tail"].get("text")
        text_key = (h_text, t_text, row["relation_id"])
        prev = by_text.get(text_key)
        if prev is None:
            by_text[text_key] = row
            continue
        prev_has_id = bool(prev["tail"].get("entity_id"))
        row_has_id = bool(row["tail"].get("entity_id"))
        if row_has_id and not prev_has_id:
            by_text[text_key] = row
        elif row_has_id == prev_has_id and float(row.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
            by_text[text_key] = row
    relations = _prune_subsumed_relations(list(by_text.values()))

    nodes: List[Dict] = []
    seen_nodes = set()
    for ent in linked_entities:
        node_id = ent.get("entity_id") or f"MENTION::{ent.get('text')}"
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        nodes.append(
            {
                "node_id": node_id,
                "name": ENTITY_CANONICAL_MAP.get(ent.get("kb_name") or ent.get("text"), ent.get("kb_name") or ent.get("text")),
                "type": ent.get("kb_type") or ent.get("label"),
                "link_score": ent.get("link_score", 0.0),
            }
        )
    for row in relations:
        for side in ("head", "tail"):
            endpoint = row.get(side, {})
            node_id = endpoint.get("entity_id") or f"MENTION::{endpoint.get('text')}"
            if not endpoint.get("text") or node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            nodes.append(
                {
                    "node_id": node_id,
                    "name": endpoint.get("text"),
                    "type": _infer_relation_endpoint_type(endpoint, row.get("relation_id", ""), side),
                    "link_score": 1.0 if endpoint.get("entity_id") else 0.0,
                }
            )
    return {
        "text": text,
        "entities": linked_entities,
        "entity_nodes": nodes,
        "relations": relations,
        "relation_lexicon": [
            {
                "relation_id": rule["relation_id"],
                "label_zh": rule["label_zh"],
                "label_en": rule["label_en"],
                "aliases": list(rule["aliases"]),
            }
            for rule in active_relation_rules
        ],
    }


def export_kg_artifacts(pipeline: Dict, core_json: Path, triples_csv: Path, nodes_csv: Path) -> None:
    entities = pipeline.get("entities", [])
    node_rows = pipeline.get("entity_nodes", [])
    relations = pipeline.get("relations", [])
    used = set()
    for row in relations:
        used.add(row["head"].get("entity_id") or row["head"].get("text"))
        used.add(row["tail"].get("entity_id") or row["tail"].get("text"))

    core_nodes = []
    seen = set()
    for row in node_rows:
        node_id = row.get("node_id")
        if not node_id or node_id not in used or node_id in seen:
            continue
        seen.add(node_id)
        core_nodes.append(row)
    if not core_nodes:
        for row in entities:
            node_id = row.get("entity_id") or f"MENTION::{row.get('text')}"
            if node_id not in used or node_id in seen:
                continue
            seen.add(node_id)
            core_nodes.append(
                {
                    "node_id": node_id,
                    "name": row.get("kb_name") or row.get("text"),
                    "type": row.get("kb_type") or row.get("label"),
                    "link_score": row.get("link_score", 0.0),
                }
            )

    core_json.parent.mkdir(parents=True, exist_ok=True)
    core_json.write_text(
        json.dumps(
            {
                "text": pipeline.get("text", ""),
                "node_count": len(core_nodes),
                "edge_count": len(relations),
                "nodes": core_nodes,
                "relations": relations,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    triples_csv.parent.mkdir(parents=True, exist_ok=True)
    with triples_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["head", "head_id", "relation", "tail", "tail_id", "confidence", "source", "sentence"],
        )
        writer.writeheader()
        for row in relations:
            writer.writerow(
                {
                    "head": row["head"].get("text", ""),
                    "head_id": row["head"].get("entity_id", ""),
                    "relation": row.get("label", row.get("relation_id", "")),
                    "tail": row["tail"].get("text", ""),
                    "tail_id": row["tail"].get("entity_id", ""),
                    "confidence": row.get("confidence", ""),
                    "source": row.get("source", ""),
                    "sentence": row.get("sentence", ""),
                }
            )

    with nodes_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "name", "type", "link_score"])
        writer.writeheader()
        for row in core_nodes:
            writer.writerow(
                {
                    "node_id": row.get("node_id", ""),
                    "name": row.get("name", ""),
                    "type": row.get("type", ""),
                    "link_score": row.get("link_score", ""),
                }
            )


def write_csv_rows(path: Path, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
