# 掌柜智库离线评测

这套脚本用于生成真实、可复现的项目指标，不会预填或伪造准确率。

## 1. 标注数据集

复制 `datasets/query_eval.example.jsonl`，命名为 `datasets/query_eval.jsonl`。
为每个问题填写 Milvus/Attu 中能够回答问题的真实 `expected_chunk_ids`。空列表会被评测脚本跳过，不会错误地计入 Recall。

建议至少准备 20～30 条问题，覆盖：明确型号、简称、型号混淆、历史指代、操作步骤、故障排查和知识库无答案。

## 2. 运行端到端评测

在项目根目录执行：

```powershell
conda run -n know python evaluation/run_evaluation.py `
  --dataset evaluation/datasets/query_eval.jsonl `
  --output evaluation/reports/query_results.jsonl
```

该命令会真实使用本地 BGE 模型、Milvus、MCP 和百炼 API。失败问题会记录错误并继续执行。

## 3. 计算指标

```powershell
conda run -n know python evaluation/retrieval_metrics.py `
  --input evaluation/reports/query_results.jsonl `
  --output evaluation/reports/metrics_at_3.json `
  --k 3
```

报告包括成功率、Hit@K、Recall@K、MRR、商品名识别准确率、平均耗时和 P95 耗时。

节点级耗时默认同步写入 `knowledge/logs/query_metrics.jsonl`。可通过环境变量关闭或修改路径：

```env
QUERY_METRICS_ENABLED=true
QUERY_METRICS_PATH=knowledge/logs/query_metrics.jsonl
```

## 4. 简历使用原则

只有完成真实标注和运行后，才可以把报告数字写进简历。建议同时保存失败案例和改进记录，用于解释商品过滤、HyDE、RRF 与 Reranker 的实际贡献。

## 5. 松下吸尘器真实实验

本轮实验使用 `appliance_*_v2` 独立集合，不影响原有知识库数据。主要命令如下：

```powershell
$env:CHUNKS_COLLECTION="appliance_chunks_v2"
$env:ITEM_NAME_COLLECTION="appliance_items_v2"
$env:ENTITY_NAME_COLLECTION="appliance_entities_v2"

conda run -n know python evaluation/run_retrieval_ablation.py `
  --dataset evaluation/datasets/panasonic_vacuum_retrieval_v1.jsonl `
  --output evaluation/reports/panasonic_ablation_full_v1.jsonl `
  --summary evaluation/reports/panasonic_ablation_full_v1_summary.json

conda run -n know python evaluation/run_item_routing_evaluation.py `
  --dataset evaluation/datasets/panasonic_item_routing_v1.jsonl `
  --output evaluation/reports/panasonic_item_routing_v2.jsonl `
  --summary evaluation/reports/panasonic_item_routing_v2_summary.json

conda run -n know python evaluation/run_answer_fact_evaluation.py `
  --dataset evaluation/datasets/panasonic_answer_facts_v1.jsonl `
  --retrieval-results evaluation/reports/panasonic_ablation_full_v1.jsonl `
  --output evaluation/reports/panasonic_answer_facts_v2.jsonl `
  --summary evaluation/reports/panasonic_answer_facts_v2_summary.json
```

查询图默认使用本轮实验选出的 Dense 快速路径；需要复现实验中的完整增强链路时设置：

```powershell
$env:QUERY_RETRIEVAL_MODE="full"
```

完整数据、指标、决策与限制见 `evaluation/reports/panasonic_experiment_report_v1.md`。
