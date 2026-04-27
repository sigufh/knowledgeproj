# knowledgeproj

面向知识工程课程的最小可运行知识图谱管线。

当前仓库按两类脚本整理：

- 训练/数据脚本：负责造数、训练、分析
- 分阶段管线脚本：统一收敛到 `scripts/kg.py`

## 1. 环境准备

首次在 PowerShell 使用 conda：

```powershell
conda init powershell
```

重开 PowerShell 后创建环境：

```powershell
conda create -n knowledgeproj python=3.10 -y
conda activate knowledgeproj
pip install -e .
```

如果 `conda activate` 失败，可先执行：

```powershell
& "$env:USERPROFILE\anaconda3\shell\condabin\conda-hook.ps1"
conda activate knowledgeproj
```

## 2. 训练

```powershell
python scripts/train_ner_crf.py --train data/sample/ner_train.jsonl --dev data/sample/ner_dev.jsonl --model-out artifacts/ner_crf_turing.joblib
python scripts/train_linker.py --kb data/sample/kb.jsonl --model-out artifacts/linker_turing.joblib
python scripts/train_re.py --train data/sample/re_train.jsonl --dev data/sample/re_dev.jsonl --model-out artifacts/re_clf_turing.joblib
```

如果要基于当前 `data/sample/input.txt` 重新构造图灵领域样本，可先执行：

```powershell
python scripts/build_turing_domain_data.py
```

## 3. 分阶段运行

统一入口：

```powershell
python scripts/kg.py --help
```

### 3.1 Stage 1: 实体抽取

```powershell
python scripts/kg.py extract `
  --text-file data/sample/input.txt `
  --ner-model artifacts/ner_crf_turing.joblib `
  --linker-model artifacts/linker_turing.joblib `
  --out-dir outputs/reference_style_chain
```

输出：

- `outputs/reference_style_chain/entity_extraction/input.csv`
- `outputs/reference_style_chain/entity_extraction/input_lexicon.jsonl`

### 3.2 Stage 2: 实体消歧

```powershell
python scripts/kg.py disambiguate `
  --text-file data/sample/input.txt `
  --linker-model artifacts/linker_turing.joblib `
  --out-dir outputs/reference_style_chain `
  --min-link-score 0.45
```

输出：

- `outputs/reference_style_chain/entity_disambiguation/input_mentions.csv`
- `outputs/reference_style_chain/entity_disambiguation/input.csv`

### 3.3 Stage 3: 关系抽取与图谱构建

```powershell
python scripts/kg.py build `
  --text-file data/sample/input.txt `
  --out-dir outputs/reference_style_chain `
  --re-model artifacts/re_clf_turing.joblib `
  --relation-strategy open `
  --relation-label-lang zh `
  --require-linked-entity
```

输出：

- `outputs/reference_style_chain/knowledge_graph/input_pipeline.json`
- `outputs/reference_style_chain/knowledge_graph/input_relation_lexicon.json`
- `outputs/reference_style_chain/knowledge_graph/input_core.json`
- `outputs/reference_style_chain/knowledge_graph/input_nodes.csv`
- `outputs/reference_style_chain/knowledge_graph/input_triples.csv`

### 3.4 Stage 4: 报告与提交版导出

```powershell
python scripts/kg.py export `
  --text-file data/sample/input.txt `
  --out-dir outputs/reference_style_chain
```

输出：

- `reports/final_turing_kg_report.md`
- `reports/submission_core_kg.md`
- `outputs/reference_style_chain/knowledge_graph/input_submission_core.json`
- `outputs/reference_style_chain/knowledge_graph/input_submission_core.csv`

### 3.5 Stage 5: 可视化

```powershell
python scripts/kg.py visualize `
  --text-file data/sample/input.txt `
  --out-dir outputs/reference_style_chain
```

输出：

- `outputs/reference_style_chain/figures/pipeline_relation_graph.png`

## 4. 一键全链路

推荐直接使用：

```powershell
python scripts/kg.py all `
  --text-file data/sample/input.txt `
  --ner-model artifacts/ner_crf_turing.joblib `
  --linker-model artifacts/linker_turing.joblib `
  --re-model artifacts/re_clf_turing.joblib `
  --out-dir outputs/reference_style_chain `
  --min-link-score 0.45 `
  --relation-strategy open `
  --relation-label-lang zh `
  --require-linked-entity
```

## 5. 当前推荐配置

对于当前图灵文本样例，推荐：

- NER：`artifacts/ner_crf_turing.joblib`
- Linker：`artifacts/linker_turing.joblib`
- RE：`open`
- 参数：`--require-linked-entity`
- 参数：`--min-link-score 0.45`

原因：

- 当前小样本 RE 分类器泛化仍弱
- 规则关系抽取在传记长文本上更稳定
- linked-only 能显著减少伪实体和伪关系

## 6. 主要脚本说明

### 6.1 管线入口

- `scripts/kg.py`
  - 统一分阶段 KG 入口
  - 支持 `extract / disambiguate / build / export / visualize / all`

### 6.2 训练与数据

- `scripts/build_turing_domain_data.py`
  - 从 `input.txt` 构造图灵领域训练样本
- `scripts/augment_turing_kb.py`
  - 补充当前输入文本需要的 KB 实体
- `scripts/train_ner_crf.py`
  - 训练 CRF NER
- `scripts/train_linker.py`
  - 训练实体链接模型
- `scripts/train_re.py`
  - 训练关系分类器
- `scripts/train_ner_bert.py`
  - BERT-CRF 训练入口

### 6.3 分析与可视化

- `scripts/run_ner_ablation.py`
  - NER 消融实验
- `scripts/analyze_ner_errors.py`
  - NER 错误分析
- `scripts/visualize_results.py`
  - NER 图表和关系图可视化实现

