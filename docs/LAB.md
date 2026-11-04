# Decision Lab（CPU 证据演示）

Decision Lab 把已完成的 v1 真实预测和性能测量做成可交互页面。当前版本支持真实结果回放、离线阈值探索和已测性能展示；不启动模型、不需要 GPU、不生成新的 v2 实验结果。

## 本机启动

在项目根目录使用独立 Lab 环境，避免改变模型运行环境：

```bash
python3 -m venv .venv-lab
.venv-lab/bin/python -m pip install -e '.[test]' 'gradio==6.31.0'
```

已安装环境可直接使用后续命令。Gradio 固定为 6.31.0；代码延迟导入 Gradio，其他 v2 命令不会因为缺少网页依赖而无法导入。

首次从两个已验收的回传目录构建便携证据包。输出目录必须尚不存在，而且必须位于原始证据树之外。

```bash
.venv-lab/bin/python -B -m decision_study.v2 lab-build \
  --v1-dir results/returned-experiment-2026-11-02-101054 \
  --benchmark-dir results/returned-benchmark-2026-11-02-130735 \
  --bundle reports/decision-lab-2026-11-02

.venv-lab/bin/python -B -m decision_study.v2 lab \
  --bundle reports/decision-lab-2026-11-02 \
  --port 7860
```

浏览器打开 `http://127.0.0.1:7860`。停止命令用 Ctrl-C。当前服务只绑定本机 loopback，关闭 Gradio public sharing；不会打开公司 GPU 端口或续期 allocation。

已构建的 `lab.json` 和 `manifest.json` 可以随项目复制到另一台机器；只运行页面不需要原始模型权重、虚拟环境或原始回传目录。便携文件不包含用户名、机器绝对路径、运行环境或服务 URL。

## 三个页面如何使用

### Cases

默认提供四个已验证的 StartLux/refitted 样例，分别说明本地正确、Qwen 救回、转交误伤、高置信度错误。样例按已知结果类别挑选，只用于解释，不代表总体频率。

左侧展示输入，中间展示前级概率和冻结 gate，右侧展示本次回放的路由及最终标签。可切换 Kev/StartLux 和 released/refitted 条件；切换后同一输入的路径可能变化。

展开全量样例浏览器，可按 ID 或输入片段搜索 3,450 个唯一 test/stress 输入。CLINC stress 工作负载会复用主测试 banking 输入，因此不是额外的独立测试。calibration/validation 未列入浏览器，因为 Qwen 在这些 split 尚无 v1 预测。

Evidence 展开区显示规范化的保存记录和路由规则，JSON 下载会包含原始输入、概率、标签、gate 和来源哈希。它不包含主机路径，也不声称是未经筛选的完整运行日志。

Qwen 当前只提供标签，没有概率。前级接受时，Qwen 显示的是另行采集的独立 baseline，页面明确说明该政策未使用它。页面的概率条和规则说明不代表模型 attention、归因或隐藏思维链。

### Policy sandbox

选择 CLINC main、CLINC OOS stress 或 SQuAD main。默认使用保存的冻结 gate，数值应与正式分析完全一致。

取消“Use saved frozen gate”后可调整阈值；也可显式选择 defer every request。接受边界是 `confidence >= threshold`，整个同分组一起接受。`accept_none` 是独立状态，不能用 threshold=1 替代。

页面报告本地覆盖率、接受错误率、最终准确率、救回和误伤。没有接受样本时风险显示为未定义，而非 0%。曲线使用 51 个示意阈值，当前和冻结 gate 的标记按精确规则计算；曲线不执行阈值优化。

该页在已见数据上探索，不重拟合温度、不回写正式 gate，也不把测试最优值发布成新结论。任意滑块工作点的真实吞吐与延迟尚未测量，页面不做线性插值或成本承诺。

### Measured performance

展示既有 v1 测量中的 CLINC/SQuAD、客户端并发 1/8、五种部署。每项是三个测量轮次的算术均值；p95 的均值不是合并请求的 p95，也不是置信区间。

性能工作负载每任务只有 200 个唯一输入，重复执行不增加质量样本量。以 SQuAD/StartLux→Qwen 为例，性能池的转交率是 5%，正式 2,000 输入质量池是 6.25%；两者各有分母。

请求时延从 controller worker 开始执行计时，排除此前排队、服务启动和预热。这些数据只对应冻结 refitted gate 和原测量配置，不能给 sandbox 新阈值代替测速。

## 文件与 API

- `decision_study/v2/lab/bundle.py`：`build(v1_dir, benchmark_dir, target_dir)` 返回 `lab.json` 路径；`load_bundle(bundle_dir)` 校验文件哈希及 schema。
- `sample_replay(data, model, sample_id, condition="refitted")`：纯 CPU 规则回放。
- `sandbox(data, model, task, workload="main", threshold=None, condition="refitted", accept_none=False)`：纯 CPU 工作负载统计，无性能推测。
- `measured_performance(data, task, concurrency)`：仅汇总保存的性能单元。
- `decision_study/v2/lab/app.py`：`create_app(bundle_dir)` 返回未启动的 Blocks；`launch(bundle_dir, host="127.0.0.1", port=7860)` 应用样式并启动本机页面。

Gradio 6 将 CSS/theme 参数放在 `launch()`。手动调用 `create_app(...).launch()` 时需显式传入本模块的 `CSS`，否则页面仍可用，但不会使用本项目样式。[Gradio 6 迁移说明](https://gradio.app/guides/gradio-6-migration-guide)

## 完整性与验收

构建器验证两个来源的完整文件 manifest，再检查预测分片、校准身份、benchmark settings 和三个工作负载的冻结路由指标。导出的 manifest 用于检测便携文件损坏；它不是数字签名，也不替代来源审计。

```bash
.venv-lab/bin/python -B -m pytest tests/test_v2_lab.py -q
```

测试覆盖等号边界、accept-none、未知 ID、非法阈值、HTML 转义、不可变数据、portable hash/schema、四例回放、真实冻结指标、stress 复用和性能分母。完整证据在本机存在时会进行真实 golden checks；只分发源码而没有数据包时，这部分会明确跳过，纯逻辑测试仍运行。

界面代码对所有进入 HTML 的数据文本转义。JSON 下载仅保存筛选后的公共证据，使用独立临时目录；运行页面和更改阈值都不写原始 predictions、calibration 或 analysis。

当前视觉设计采用冷白背景、海军蓝文字和青色决策路径，主要结构是输入、概率与结果的三列流程。工程元数据放在展开区，演示重点保持在实际决策和质量/成本取舍。

原始研究说明见[正式结果](../reports/formal-2026-11-02/RESULTS_ZH.md)、[真实性能](../reports/benchmark-2026-11-02/BENCHMARK_ZH.md)和[原四例回放](../reports/formal-2026-11-02/DEMO_REPLAY_ZH.md)。
