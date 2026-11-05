# 运行完整预算和冻结 embedding 基线

本轮为 `tfidf1600`、`minilm400`、`minilm1600` 三个CPU条件。协议见
[CHEAP_BASELINES_PROTOCOL_ZH.md](CHEAP_BASELINES_PROTOCOL_ZH.md)。
旧 V1/V2/V3 结果、400标签基线与 GPU 环境继续保留原位。

## 一次完成

在公司 code-server 终端，将独立的 `/home/zhanj289/e6998/decision-study-src` 更新到本次
交付消息给出的完整 commit；确认没有任务使用该源码目录再checkout。不要在旧 runtime
`decision-study` 目录更新Git，也不要使用 `git pull` 合并未核对的版本。

在固定的新源码目录执行：

```bash
bash scripts/run_cheap_baseline_suite.sh /home/zhanj289/e6998/decision-study
```

该入口依次完成：检查工作区干净；用旧controller的Python创建新的CPU环境；按完整pins
安装binary wheels；下载并准备训练专用数据；下载并核验固定MiniLM；预检；运行三条件；
统一冻结、评价、导出。无需另传 prepared ZIP，也不需要新申请GPU。

要求Python3.11–3.13。固定ORT Linux wheel要求x86_64/glibc≥2.28；不满足时安装会失败，
不要自行取消版本pin。已有旧controller只是创建venv的解释器，依赖安装只写新
`.venv-cheap-baselines`。Mac已验证，Linux wheel解析已检查；实际node运行仍需回传验收。

| 位置 | 内容 |
|---|---|
| 新源码根 `.venv-cheap-baselines` | 独立CPU运行环境 |
| 旧runtime `results/aux-clinc-cheap-v1-inputs/raw` | 两份固定官方数据 |
| 同输入根 `prepared` | 训练专用文件与manifest |
| 同输入根 `model` | 八份固定编码器文件 |
| 旧runtime `results/aux-clinc-cheap-v1` | 三个新条件的结果 |

模型文件约90.9MB；下载时间取决于网络。终端打印 `RESULT BUNDLE` 后回传：

```text
/home/zhanj289/e6998/decision-study/results/aux-clinc-cheap-v1/exports/cheap_baseline_bundle.zip
```

## 出错或分阶段操作

完整成功后重复入口会校验并复用已有结果。下载可复用逐文件验证通过的缓存；最终文件
已存在但SHA不符则停止，保留检查。拟合阶段若中断而没有完整freeze，原目录也会停止，
请保留错误和目录；不要删除freeze、访问收据或锁来强行重跑。

需要分阶段诊断时，以下命令显示对应参数：

```bash
bash scripts/run_cheap_baselines.sh prepare --help
bash scripts/run_cheap_baselines.sh download-model --help
bash scripts/run_cheap_baselines.sh preflight --help
bash scripts/run_cheap_baselines.sh fit --help
bash scripts/run_cheap_baselines.sh evaluate --help
bash scripts/run_cheap_baselines.sh bundle --help
```

`preflight`只验证数据、依赖、模型身份，不拟合、不读V1 gold。`fit`固定所有三个模型，
`evaluate`评分已有完整freeze，`bundle`导出已有完整结果。运行途中不要更新checkout或移动
输入路径。回传内容将独立核验；单次CPU内部计时不等同于原GPU服务性能。
