# 图灵文本知识图谱实验报告（最终版）

## 1. 数据与任务
- 输入文本：`data/sample/input.txt`（图灵传记长文本）
- 任务通路：实体抽取 -> 实体链接 -> 关系抽取 -> 图谱构建
- 关键原则：训练域与抽取域对齐（同源文本构建训练集）

## 2. 本次模型与产物
- NER模型：`artifacts/ner_crf_turing.joblib`
- 链接模型：`artifacts/linker_turing.joblib`
- 关系模型：`artifacts/re_clf_turing.joblib`（本次主用 open 规则抽取）
- 图谱JSON：`outputs/reference_style_chain/knowledge_graph/input_pipeline.json`
- 核心图谱JSON：`outputs/reference_style_chain/knowledge_graph/input_core.json`
- 三元组CSV：`outputs/reference_style_chain/knowledge_graph/input_triples.csv`
- 节点CSV：`outputs/reference_style_chain/knowledge_graph/input_nodes.csv`
- 可视化：`outputs/reference_style_chain/figures/pipeline_relation_graph.png`

## 3. 结果统计
- 抽取实体数：193
- 抽取关系数：43
- 核心节点数（参与关系）：36
- 关系类型分布：{'被誉为': 1, '负责研究': 5, '设计': 2, '撰写': 4, '提出': 2, '父亲': 1, '母亲': 1, '出生于': 1, '就读于': 4, '以…命名': 1, '毕业于': 2, '聘请': 1, '任职于': 6, '破译': 2, '合作': 1, '继任于': 1, '获授': 1, '发表': 1, '迫害': 1, '撰文于': 1, '公开道歉': 1, '定罪': 1, '赦免': 2}

## 4. 代表性三元组
- (艾伦·图灵, 出生于, 帕丁顿)
- (艾伦·图灵, 就读于, St. Michael's)
- (艾伦·图灵, 毕业于, 剑桥大学国王学院)
- (艾伦·图灵, 任职于, GC&CS)
- (艾伦·图灵, 破译, 恩尼格玛密码机)
- (艾伦·图灵, 发表, 形态发生的化学基础)
- (英国司法部, 赦免, 艾伦·图灵)

## 5. 质量说明
- 已抑制自环关系与明显误触发（如“要求…赦免”导致的伪关系）。
- 当前关系以事件触发词高精规则为主，已补上出生、就读、毕业等关键事实链，并覆盖普林斯顿求学阶段。
- 仍有可继续补强的点：部分英文头衔实体类型、长句事件关系覆盖、少量未链接尾实体。
- 若用于最终提交，建议补充少量人工校验三元组。

