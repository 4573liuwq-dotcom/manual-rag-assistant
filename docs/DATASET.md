# 数据集说明

## 文档语料

实验使用4份松下吸尘器官方说明书，覆盖MC-6RB77A、MC-8D56C、MC-8R76C、MC-CA781/MC-CA783，共88页。PDF不随GitHub仓库再分发；复现实验者应从厂商公开下载页获取，并按入库清单校验文件名。

| 文档 | 页数 | Chunk | 有效图片 |
|---|---:|---:|---:|
| MC-6RB77A.pdf | 26 | 44 | 55 |
| MC-8D56C.pdf | 20 | 34 | 31 |
| MC-8R76C.pdf | 26 | 41 | 63 |
| MC-CA781_MC-CA783.pdf | 16 | 23 | 36 |
| 合计 | 88 | 142 | 185 |

完整入库记录见`evaluation/reports/panasonic_import_manifest_v2.jsonl`。MinerU最初提取508个图片对象，经面积过滤后保留185张进入VLM描述流程，过滤比例63.6%。

## 人工评测集

- `panasonic_vacuum_retrieval_v1.jsonl`：32道检索题，含目标型号和人工标注相关Chunk。
- `panasonic_item_routing_v1.jsonl`：16道型号路由题，覆盖全称、简称和泛称。
- 答案事实集：8题，检查关键事实、引用ID有效性和整体通过条件。

数据集规模较小，目的是支持可重复的工程消融和回归测试，不代表真实生产流量。逐题输出与汇总报告均保存在`evaluation/reports/`，可通过`evaluation/verify_offline_artifacts.py`离线校验。
