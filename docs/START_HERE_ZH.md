# 入门指南：先读什么、怎么运行演示

2026-11-10 整理。仓库保存源码、协议、已验收的报告和演示用的便携证据；原始结果包、模型和 GPU 环境在 Git
之外。看报告和跑演示都只需要这个仓库和一台能运行 Python 的电脑。

## 1. 阅读顺序

| 顺序 | 文件 | 用时 | 看什么 |
|---|---|---|---|
| 1 | [paper/main.pdf](../paper/main.pdf) | 15 分钟 | 摘要、Figure 4–5、第 6 节 Discussion 与第 7 节 Limitations |
| 2 | [reports/upgrade-9b-serving-2026-11-09/RESULTS_ZH.md](../reports/upgrade-9b-serving-2026-11-09/RESULTS_ZH.md) | 10 分钟 | 最新的 9B 服务研究：两张 goodput 表、99% 规则的通过点、4B 20 请求/秒的异常轮 |
| 3 | [reports/README.md](../reports/README.md) | 5 分钟 | 全部已验收结果的一张表，可点进各轮报告 |
| 4 | [docs/PRESENTATION_ZH.md](PRESENTATION_ZH.md) | 10 分钟 | 课堂展示的大纲、讲稿、演示点击顺序和四个案例 |
| 5 | [docs/CURRENT_STATE.md](CURRENT_STATE.md) 最上面一节 | 3 分钟 | 当前进度和仍待人工确认的事项 |

想核对数字来源时，看 [paper/README.md](../paper/README.md) 的来源表和各报告目录里的 `audit/README.md`。

最终交付的分工按 Ju 11/10 的安排：Ju Zhang 负责实验设计、运行时与流水线实现、整合和最终技术审阅；Weiyu Chen
负责统计核验、结果解读和稿件审阅；Yizhi Hu 负责复现检查、演示讲解和展示材料。论文不写算力致谢。各项核对实际
完成后，在 `docs/CURRENT_STATE.md` 记录日期、执行人和结果。

## 2. 运行 CPU 演示

在 Ju 的 Mac 上，仓库就在下面这个目录，`.venv-lab` 环境已经装好：

```bash
cd /Users/zhanj289/Desktop/COMS6998/decision_study_repo
bash scripts/run_lab.sh
```

浏览器打开 <http://127.0.0.1:7860>。7860 被占用时换一个端口：

```bash
bash scripts/run_lab.sh --port 7861     # 然后打开 http://127.0.0.1:7861
```

停止演示：在运行它的终端按 Ctrl-C。只想检查演示证据、不启动网页时：

```bash
.venv-lab/bin/python -m decision_study.lab check
```

正常输出以 `"status": "verified"` 开头，列出 V1–V3、36 个 9B 服务单元和 4 个案例，最后是 `"model_calls": 0`。

演示只读取 `reports/decision-lab-*` 里的便携证据，不调用模型、不需要 GPU 或网络，服务只监听本机。

### 本机已验证的环境

2026-11-10 在 Ju 的 Mac 上用现有环境验证：`.venv-lab` 为 Python 3.13.3、gradio 6.31.0、matplotlib 3.11.2、
numpy 2.5.4、pytest 9.2.0；`.venv-cheap-baselines` 为 Python 3.13.3、scikit-learn 1.9.1、onnxruntime 1.31.1；
pdfTeX 来自 TeX Live 2025（BasicTeX）。同一天还从一份只含跟踪文件和本次新增文件的干净源码副本启动过演示，
用的是这些已装好的依赖。

### 在另一台电脑上第一次安装

需要 Python 3.11–3.13。在 clone 下来的仓库根目录执行：

```bash
python3 -m venv .venv-lab
.venv-lab/bin/python -m pip install -e '.[lab,test]'
bash scripts/run_lab.sh
```

这条安装命令需要联网下载依赖。本轮整合没有在全新环境里实际跑过它，第一次安装后建议先运行上面的
`check` 命令。

## 3. 演示里能看到的已测数据

| 标签页 | 内容 | 建议操作 |
|---|---|---|
| Overview | V2、V3 两项预定主比较的配对区间；下方是四项 follow-up 的要点 | 指出两个区间都跨零 |
| V2 · SQuAD | 冻结路由的准确率、救回与误伤；10/25/40 请求/秒实测 goodput | 把速率切到 40，看级联降到 4.9/秒 |
| V3 · ANLI | positive_gain 零转交；confidence 预算从 10% 变成 23.2% | 打开一个 rescue 案例和一个 harm 案例 |
| Follow-ups | 9B 服务研究、CLINC CPU 与 GPU 服务、便宜分类器、四个 ANLI 案例 | 500 ms + 40/s，再切 80/s 和 1,000 ms |
| V1 replay | 原 V1 回放、阈值探索与性能页 | 说明滑块只是在已见数据上探索 |

Follow-ups 页的 9B 部分只能选择已测的 10/20/40/80 请求/秒和 500/1,000 ms，不会在已测速率之间插值。
40 请求/秒、500 ms 时 StartLux-9B 的中位正确 goodput 最高（30.750/秒），但有一轮及时比例 98.87%，没有达到
99% 规则；切到 80 请求/秒后 StartLux-4B 最高（38.258/秒）。切到 1,000 ms 后，9B 在 40 请求/秒也满足规则。

## 4. 复现与核对

```bash
bash scripts/reproduce_cpu.sh            # 缺少外部证据的步骤显示 SKIP
bash scripts/reproduce_cpu.sh --strict   # 完整验收：任何跳过都算失败
```

完整验收需要外部证据根（原始结果包与密封 gold）。不在默认位置时设置 `EVIDENCE_ROOT=/path/to/evidence-root`，
布局见 [ARTIFACTS_ZH.md](ARTIFACTS_ZH.md)。脚本逐字节比较 11/6 两项回传的 12 份审计收据、9B 服务研究的 6 份
审计收据、Lab follow-up 证据、9B 报告表格和论文图，最后在临时目录编译论文；已跟踪文件不会被修改。

## 5. 常见情况

- 浏览器打不开页面：确认终端里仍在运行，地址用 `127.0.0.1`，端口与 `--port` 一致。
- 启动时报 `Decision Lab: ... hash/size mismatch` 或 `Accepted summary changed`：便携证据被改过。用
  `git status` 和 `git diff` 找出改动并恢复，不要重新生成去覆盖。
- 端口被占用：换 `--port 7861`。不要结束别人正在用的服务。
- 其他排障记录见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。
