# knowledgeproj

面向知识工程课程的最小可运行管线：

- 实体抽取（传统 `CRF` 基线 + `BERT-CRF`）
- 实体消歧（候选检索 + 上下文重排）
- 关系抽取（监督分类）

## 1. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. 训练

```bash
python scripts/train_ner_crf.py --train data/sample/ner_train.jsonl --dev data/sample/ner_dev.jsonl --model-out artifacts/ner_crf.joblib
python scripts/train_linker.py --kb data/sample/kb.jsonl --model-out artifacts/linker.joblib
python scripts/train_re.py --train data/sample/re_train.jsonl --dev data/sample/re_dev.jsonl --model-out artifacts/re_clf.joblib
```

可做 NER 消融：

```bash
python scripts/run_ner_ablation.py --train data/sample/ner_train.jsonl --dev data/sample/ner_dev.jsonl --output artifacts/ner_ablation.json
python scripts/analyze_ner_errors.py --data data/sample/ner_dev.jsonl --model artifacts/ner_crf.joblib --output artifacts/ner_errors.jsonl
```

可选（需要下载预训练模型）：

```bash
python scripts/train_ner_bert.py --train data/sample/ner_train.jsonl --dev data/sample/ner_dev.jsonl --output-dir artifacts/bert_crf
```

若提示 `BERT-CRF依赖未就绪`，先修复本地 PyTorch 环境再训练。

## 3. 端到端推理

```bash
python scripts/run_pipeline.py \
  --text "阿里巴巴总部位于杭州。马云曾担任阿里巴巴董事局主席。" \
  --ner-type crf \
  --ner-model artifacts/ner_crf.joblib \
  --linker-model artifacts/linker.joblib \
  --re-model artifacts/re_clf.joblib
```

## 4. 数据格式

### NER (`*.jsonl`)

```json
{"text": "马云创立了阿里巴巴", "entities": [{"start": 0, "end": 2, "label": "PER"}, {"start": 5, "end": 9, "label": "ORG"}]}
```

`start`/`end` 采用左闭右开字符下标。

### 实体库 (`kb.jsonl`)

```json
{"entity_id": "E1", "name": "阿里巴巴集团", "aliases": ["阿里巴巴", "Alibaba"], "type": "ORG", "description": "中国互联网公司，总部位于杭州"}
```

### 关系抽取 (`re_*.jsonl`)

```json
{"text": "马云创立了阿里巴巴", "head": {"start": 0, "end": 2, "label": "PER"}, "tail": {"start": 5, "end": 9, "label": "ORG"}, "label": "FOUNDED"}
```

## 5. 作业建议

- 不直接导入结构化数据，必须展示非结构化文本抽取流程。
- 实体抽取至少报告 `CRF` 与 `BERT-CRF` 两个模型对比。
- 展示消融和误差分析：边界错误、类型错误、实体消歧冲突等。
