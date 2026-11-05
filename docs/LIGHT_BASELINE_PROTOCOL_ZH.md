# CLINC 400 标签 TF-IDF 监督基线

2026-11-04。实验标识 `clinc-tfidf400-v1`，机器可读方案为
[clinc_tfidf400_protocol.json](../configs/clinc_tfidf400_protocol.json)。
这是原月度计划中廉价分类基线的第一批，独立于已完成的 V1/V2/V3。

## 要回答的问题

在原 CLINC 15 个 banking intent 加 `oos` 的任务上，一个只使用 400 个项目内标签的
TF-IDF＋LogisticRegression 分类器能达到什么质量，需要多少 CPU 训练和批量预测时间？
它直接预测任务标签，不预测是否转交给 Qwen。语言模型不训练、不推理。

这批工作定位为监督预算明确的回顾性描述。已有 test/stress 被用于先前结果分析，不能
成为新的未见确认集。分类器使用标签训练，与原 frozen LLM 的条件不同；原 LLM 的概率
校准和 gate 也使用了 calibration/validation，不能把整个流程称为同条件零样本比较。

## 数据角色与来源

| 角色 | 数量和构成 | 本轮用途 |
|---|---|---|
| calibration | 15 类各 20 条＋100 OOS，共 400 | 词表、IDF、分类器的唯一拟合池 |
| validation | 15 类各 20 条＋100 OOS，共 400 | 冻结方案的诊断，无选参或阈值选择 |
| main test | 15 类各 30 条＋150 OOS，共 600 | 主要描述性质量结果 |
| stored stress | 850 条 OOS | 单列 OOS 检出率 |
| historical mixed stress | main test 的 450 条 banking＋上述 850 条 OOS，共 1,300 | 与原工作负载口径一致的混合描述；与 main test 重叠 |

各角色输入取自 V1 `data/examples.jsonl`，训练及 validation 行再与 V2
`data/examples.jsonl` 逐条核对一致；拟合标签取 V2 `data/fit_gold.json` 中的 calibration
行。特征只使用 `state` 文本。V1 `data/gold.json` 只在冻结模型和全部预测后进入评价阶段读取。
V2 fitting 标签文件也含 validation 标签；代码只把 calibration 的 400 个标签传入拟合，
不声称该文件本身是操作系统隔离的 validation 保密容器。

四份数据文件及 V1 manifest 共五份文件的 SHA 固定在方案中，不自动下载或改写旧数据。保留 V1
样本 ID 与 16 类顺序；不重新抽样，不把其他 135 个 CLINC intent 临时标成 OOS。

## 单一固定方案

- word unigram/bigram，lowercase，保留长度至少为 2 的 word token；无停用词过滤。
- `min_df=1`、`max_df=1.0`、无特征数上限，普通 TF、smooth IDF、L2 行归一化。
- multinomial LogisticRegression，L2、`C=1`、`lbfgs`、`tol=1e-4`、`max_iter=1000`、
  拟合 intercept、无 class weight、seed 6998。OOS 训练先验为 25%，每类计数另存。
- 依赖版本固定，单线程；收敛警告视为失败。不得在看到评价结果后更改迭代数或配方并
  覆盖这一轮结果。需要修改时另建方案身份和输出目录。
- vocab、IDF、系数、intercept、训练 ID、参数、依赖和输入身份连同全部预测写入冻结产物。
  概率列显式映射回 V1 标签顺序，不假定 sklearn 的 `.classes_` 顺序与之相同；argmax
  如有并列则按该固定标签顺序取第一个。

`scikit-learn==1.9.1` 使用 `l1_ratio=0` 指定此处的 L2 配置，避免使用已弃用的
`penalty` 参数。参见 [LogisticRegression 官方 API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
和 [TfidfVectorizer 官方 API](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)。

## 评价与产物边界

先 `fit` 冻结模型和预测，再 `evaluate` 验证身份并写入评价访问收据，随后读取 V1 gold。
评价不隐式重拟合；已有产物若缺失、改动或身份不符就停止，不自动重跑来覆盖证据。

main/validation/混合 stress 报告 accuracy、macro-F1、混淆矩阵、banking accuracy、
OOS precision/recall/F1。850 条 OOS 单独报告其检出表现，不能把它的 accuracy 冒称
混合 stress accuracy。此批不执行 LLM 配对显著性分析；后续如做，必须严格匹配原预测
ID、数据角色和源身份，并说明区间条件于已拟合模型。

报告训练耗时、模型产物大小和 validation/main/stored-stress 共 1,850 条输入的单批 CPU
预测耗时，记录环境与线程设置。它们属于
classifier-only 计时，不是 HTTP 服务吞吐、p95 或 SLO goodput，不能直接替代原 L40S
系统测量。没有 GPU allocation 或 session 续期步骤。

完整 official banking train＋OOS 共 1,600 条的第二种训练预算，以及冻结 embedding
分类头仍是待办。前者包含既有 engineering-development 样本，OOS 先验变为 6.25%；
不能把它与 400 条方案的差别全归因于数据量。这一批不取消 StartLux9B 或加强性能测量。

执行入口见 [RUN_LIGHT_BASELINE_ZH.md](RUN_LIGHT_BASELINE_ZH.md)。
