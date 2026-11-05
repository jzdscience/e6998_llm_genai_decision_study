# CLINC 完整标签预算与冻结 embedding 基线

2026-11-04；固定实验 ID `clinc-cheap-v1`。机器可读方案为
[clinc_cheap_protocol.json](../configs/clinc_cheap_protocol.json)。
本轮承接已验收的 [400 标签 TF-IDF](../reports/light-baseline-2026-11-04/RESULTS_ZH.md)，
补齐原计划的完整训练预算与冻结 embedding 分类头，不修改 V1/V2/V3 或原400标签配方。

## 固定的三个条件

| 条件 | 特征 | 实际传入分类头的训练标签 |
|---|---|---:|
| `tfidf1600` | word unigram/bigram TF-IDF | 1,600 |
| `minilm400` | 冻结 all-MiniLM-L6-v2，384维 | 原 calibration400 |
| `minilm1600` | 同一冻结编码器 | 1,600 |

每个条件都只运行一套固定参数。原 `tfidf400` 使用既有验收结果作历史参照，不重新拟合。
TF-IDF 和分类器参数沿用原配方：普通 TF、smooth IDF、L2 行归一化、无词表上限，
LR 使用 `C=1`、L2、`lbfgs`、`tol=1e-4`、`max_iter=1000`、无 class weight、seed6998。
收敛警告阻断，不在看到 test 后增加迭代、换编码器或调整正则强度。

这三项将分别回答同一400标签预算下的特征差异、完整预算下两类特征的表现，以及
固定特征下两种训练池的表现。它们都是监督任务分类器，不是预测是否转交的 router。

## 数据角色与解释边界

完整训练池取官方 banking 15类各100条与官方 oos_train100，共1,600条。
原 calibration400是其精确子集。准备审计确认，与 validation400、main600、OOS-stress850
没有样本 ID 或 NFKC/casefold/空格归一化文本重叠；这不证明不存在语义近重复。

完整池包含此前 engineering-development 的60条银行文本，不包含4条 authored OOS。
OOS训练比例从400池的25%变为1600池的6.25%，每个银行类从20条变为100条。
因此400→1600同时改变样本量和类别分布，不能把差异只归因于样本量。
不把其他135个CLINC intent重新标记成OOS。

`prepare` 阶段允许解析固定 SHA 的公开全分割原始 JSON，提取独立的 `train1600.jsonl`
和 manifest；它不读取项目 V1 gold。数据源固定官方 revision
`828f8093932c8fe6ca7936c3d2e52903b1c523de`。
`fit`/`preflight` 仅读取这两份准备文件、既有输入文本及 V2 fit_gold，不再读取或哈希
原始全分割 JSON、V1 gold。400条件只将calibration400标签传给拟合函数；完整训练文件
用于核验包含关系，并不是两个标签预算之间的保密容器。

validation仅诊断，不选参。main600、OOS-only850以及复用450条main银行请求的
mixed1300口径保持原样。每模型预测1,850条，三模型共5,550条；总样本量不能按这些
重复模型预测或重叠stress累加。原test/stress已经查看，本轮仅作回顾性描述，不能声称
新确认集或原LLM与监督分类头拥有相同训练条件。

## 固定编码器

使用 `sentence-transformers/all-MiniLM-L6-v2` 的官方未量化 ONNX 文件，revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。八份配置/权重/tokenizer文件的SHA与大小
写入协议；只下载这些文件，约90.9MB，不下载生成式LLM或CUDA环境。

ONNX输出 token hidden states；按发布的 pooling 配置执行 attention-mask mean pooling，
然后 L2 normalize，保存float32矩阵。LR输入显式转换float64。provider固定CPU，ORT
intra/inter线程均为1，tokenizer并行关闭；batch32，动态右侧padding，pad ID0。
这个ONNX后端单独固定，不主张与其他框架或硬件逐字节等价。

最大长度采用 `sentence_bert_config.json` 的256，包含特殊token；不采用另一份 tokenizer
配置中的512。先对全部输入做无截断tokenize检查，超长即失败，不静默截断。
本机输入预检3,450条最长25、p95为17，均未超限；未对此进行模型拟合或质量评分。

一次共享编码覆盖完整训练1600和三个评价角色1850，共3450条。固定预训练模型不因
编码其他条件或评价文本而更新参数；不拟合全池词表、IDF、PCA或其他数据适配器。
400分类头仅索引自己的训练行。模型的公开预训练数据暴露未知。

## 冻结、评分与中断

输出目录必须独立于源码、输入与权重目录。写入协议、共享embedding矩阵及ID/text-SHA
元数据、每条件的模型JSON、预测和计时；**三个条件全部完成后才写全局freeze**。
评分必须核验整个freeze及三组完整预测，先持久化访问收据，再首次读取V1 gold。
评分从不补拟合缺失模型。

完整freeze的重复调用核验后复用；没有freeze的部分输出保留并阻断，需要检查原因后另选
新输出目录。已有访问收据而无最终评价时也阻断。不要删freeze/lock来覆盖结果。
完整导出包含15个成员，即12个拟合产物加freeze、access和evaluation。

评价逐条件报告validation/main/mixed的accuracy、16类macro-F1、banking accuracy、
OOS precision/recall/F1及混淆矩阵；纯OOS池单列recall，不冒充混合多类表现。
没有依据本轮test选择最佳方案后的确认性检验或新路由政策。

## 计时与产物

编码器单独记录校验、tokenizer载入、tokenize、session载入、推理/pooling与总耗时。
共享编码成本不能给两个embedding条件都记作零。LR fit/predict与TF-IDF fit/transform
分别计时；其边界和输入数保存在产物中。单次CPU批处理不是单请求延迟、HTTP吞吐或
SLO goodput，不与原L40S系统数字直接相除计算实测速比。

新 `.venv-cheap-baselines` 固定24项运行依赖；源码、协议和精选验收进Git，原始数据、
embedding矩阵、模型和回传包留在Git外。`exports/cheap_baseline_bundle.zip`保存共享
特征、三个分类器、预测与完整冻结/评价证据，不含90MB编码器权重；权重由固定SHA恢复。

本机验证仅包括人工软件fixture、人工句子的真实encoder smoke和真实输入身份/长度
预检；实际三条件训练与评价在node执行后单独验收。
执行说明见 [RUN_CHEAP_BASELINES_ZH.md](RUN_CHEAP_BASELINES_ZH.md)。

## 官方依据

- [MiniLM模型说明与pooling示例](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [固定revision文件清单](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [CLINC固定数据源](https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de/data)
