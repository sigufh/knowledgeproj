# 知识抽取实验报告模板

## 1. 实验信息
- 项目名称：
- 日期：
- 环境（OS / Python / conda env）：
- 代码版本（commit）：

## 2. 任务与数据
- 任务范围：NER / 实体链接 / 关系抽取 / 知识图谱构建
- 数据集说明：
- 数据量（train/dev/test）：
- 标签集合：

## 3. 方法说明
### 3.1 NER
- 模型：
- 关键特征 / 超参数：

### 3.2 实体链接
- 候选召回方法：
- 重排方法：

### 3.3 关系抽取
- 模型 / 策略：
- 输入表示：

## 4. 训练与运行命令
```bash
# 训练
python scripts/train_ner_crf.py --train ... --dev ... --model-out artifacts/ner_crf_turing.joblib
python scripts/train_linker.py --kb ... --model-out artifacts/linker_turing.joblib
python scripts/train_re.py --train ... --dev ... --model-out artifacts/re_clf_turing.joblib

# 一键全链路
python scripts/kg.py all --text-file ... --ner-model artifacts/ner_crf_turing.joblib --linker-model artifacts/linker_turing.joblib --re-model artifacts/re_clf_turing.joblib --out-dir outputs/reference_style_chain --relation-strategy open --relation-label-lang zh --require-linked-entity
```

## 5. 结果汇总
### 5.1 NER 指标
| 模型 | Precision | Recall | F1 |
|---|---:|---:|---:|
| CRF |  |  |  |
| BERT-CRF |  |  |  |

### 5.2 关系抽取指标
| 指标 | 数值 |
|---|---:|
| Macro F1 |  |
| Micro F1 |  |

### 5.3 可视化结果
- `artifacts/figures/ner_ablation.png`
- `artifacts/figures/ner_error_analysis.png`
- `outputs/reference_style_chain/figures/pipeline_relation_graph.png`

## 6. 案例分析
### 6.1 正确案例
- 输入文本：
- 抽取结果：
- 原因分析：

### 6.2 错误案例
- 输入文本：
- 错误现象（边界 / 类型 / 链接 / 关系）：
- 可能原因：
- 修复思路：

## 7. 合理性检查
- 实体边界是否正确：
- 实体类型是否正确：
- 链接是否合理（`entity_id` / `kb_name`）：
- 关系方向和语义是否合理：
- 不合理项与修复建议：

## 8. 改进计划
- 消融结论：
- 下一步改进（按优先级）：
1.
2.
3.

## 9. 附录
- 关键配置：
- 额外日志：
