# Decision Lab：V1–V3 与 follow-up

这个 CPU 演示读取已验收的便携证据。V1 的原始回放与阈值探索保留，V2/V3 显示主比较、冻结政策、实际转交
预算、案例和性能证据；2026-11-10 新增的 Follow-ups 页显示 CLINC CPU 基线、CPU/GPU 服务测量和 ANLI 9B
服务研究。页面不运行模型，也不修改任何实验参数。

## 在本机启动

本机已为这个源码仓库准备独立 `.venv-lab`。在仓库根执行：

```bash
bash scripts/run_lab.sh
```

浏览器打开 `http://127.0.0.1:7860`，在启动它的终端按 Ctrl-C 停止。如果这个端口已被其他应用占用：

```bash
bash scripts/run_lab.sh --port 7861
```

在另一台电脑首次安装，使用 Python 3.11–3.13：

```bash
python3 -m venv .venv-lab
.venv-lab/bin/python -m pip install -e '.[lab]'
bash scripts/run_lab.sh
```

只看演示不需要公司 GPU、原始结果目录或模型权重，三个便携证据包都在 `reports/` 中。启动器也支持已存在的
`.venv`，或显式设置 `DECISION_LAB_PYTHON`；不会自动安装依赖、寻找远端服务或续期 allocation。服务只监听
localhost，公开分享与统计上报关闭。

## 建议的演示顺序

1. **Overview**：两项预定主比较及配对区间。V2/V3 区间均跨零，不宣称主方法改善。下方的 follow-up 表列出
   之后四项回顾性研究的要点。
2. **V2 · SQuAD**：冻结路由的准确率、救回与误伤，以及不同到达率下实测的 SLO goodput。主政策表采用
   validation 10% 目标，实际 holdout 转交率单列。
3. **V3 · ANLI**：positive_gain 零转交，confidence 在 validation 与 holdout 的预算差异。V3 本身没有同驻留
   性能测量，页面显示这一缺项，并指向 Follow-ups 页的独立测量。
4. **Follow-ups**：先在 9B 服务部分选 500 ms 与 40 请求/秒，再切到 80 请求/秒和 1,000 ms，观察排名与
   99% 规则的变化；随后看 CLINC 的 CPU/GPU 对比和四个保存的 ANLI 案例。
5. **V1 replay**：原有案例、阈值探索和性能页。滑块是在已见数据上的探索，不改变冻结 gate，也没有任意
   滑块工作点的实测时延承诺。

案例按已知结果类型挑选以帮助解释，不代表总体发生比例；总体指标来自完整评价池。

## Follow-ups 页

- **ANLI 9B 服务研究**：SLO 选 500 ms 或 1,000 ms，速率只能选已测的 10/20/40/80 请求/秒。每次选择显示
  三种部署的三轮数值、范围和中位数，每轮及时比例与 99% 规则，含晚到的完成速率、客户端拒绝比例和 p95；
  下方表格列出该 SLO 下全部已测速率，并标出三轮都达到 99% 的速率。质量与显存部分比较实时成功回答的准确率
  和同一批请求的保存预测，列出正式启动的 KV 与采样显存峰值。选择 20 请求/秒时会显示 4B 那一轮异常的说明。
- **CLINC CPU 与 GPU 服务**：速率选 5/25/100 请求/秒，比较 MiniLM-400/1600（单 CPU 线程）和
  StartLux-4B、Qwen3.5-4B（GPU 份额 0.40）的正确 goodput、完成速率、拒绝比例、p95 与资源占用。
- **CLINC 便宜分类器与 ANLI 规模对照**：TF-IDF 与 MiniLM 分类器的主测试、银行、OOS 与压力负载准确率，
  以及 StartLux-9B 相对 4B 的配对差。
- **保存的 ANLI 案例**：四个按结果挑选的例子（更大前级修正了路由保留的错误、后级修正了 9B 也答错的题、
  两个直接模型都答对而级联答错、80 请求/秒时一条正确但超过 1,000 ms 才返回的回答），并列出同一输入在服务
  运行中的实时标签计数。前三类的计数取自 3,200 条保存的 test 预测，并给出所属的更大类别（例如级联弄错的
  157 条）；第四类的计数取自该单元成功返回的实时请求。样本 ID、单元 ID 与源文件哈希放在折叠的证据区。

## 数据与复核

- V1 原便携包：`reports/decision-lab-2026-11-02/`。
- V2/V3 便携包：`reports/decision-lab-2026-11-04/`（原字节保持，V3 仍标注无同驻留性能）。
- Follow-up 便携包：`reports/decision-lab-followups-2026-11-10/`。
- 程序：`decision_study/lab/`（`followups.py` 负责 follow-up 证据，`followup_view.py` 负责页面）。本轮扩展了顶层
  Lab 的 `app.py` 与 `cli.py`；冻结的实验源码、原 V1 viewer（`decision_study/v2/lab/`）、`evidence.py` 与旧便携
  证据保持原字节。
- 原始结果、冻结参数和独立审计的来源见 [ARTIFACTS_ZH.md](ARTIFACTS_ZH.md)。

校验全部便携包（不启动服务）：

```bash
.venv-lab/bin/python -m decision_study.lab check
```

输出列出 V1–V3、36 个 9B 服务单元、162,801 次 offers、12 组 CLINC 服务数据、7 个基线模型和 4 个案例，
`model_calls` 为 0。follow-up 包的数值来自已跟踪、并由验收记录或报告清单固定 SHA-256 的汇总；加载时会核对
校验和、36 个单元的清单与计数算术、已验收的中位数和通过集，以及来源哈希。

只有需要重新制作便携包时，才需要外部原始证据。选择一个尚不存在的输出目录：

```bash
.venv-lab/bin/python -m decision_study.lab build \
  --evidence-root /path/to/verified-evidence-root \
  --bundle /path/to/new-lab-bundle
.venv-lab/bin/python -m decision_study.lab build-followups \
  --evidence-root /path/to/verified-evidence-root \
  --bundle /path/to/new-followup-bundle
```

`build-followups` 读取已跟踪的验收汇总，并为案例读取 9B 正式 ZIP 与 V3 密封 gold。2026-11-10 在本机用同一
证据重建，结果与已跟踪的 follow-up 包逐字节一致；`bash scripts/reproduce_cpu.sh` 的第 4 步重复这项比较。
新的演示包不替代原结果 ZIP，不用本机重拟合值覆盖原 freeze。
