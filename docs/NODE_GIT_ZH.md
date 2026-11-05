# 在 node 上用 Git 更新源码

2026-11-04。采用独立源码 clone，旧运行目录保持原位。

| 位置 | 用途 |
|---|---|
| `/home/zhanj289/e6998/decision-study-src` | 新的 Git 源码 checkout；实验使用固定 commit |
| `/home/zhanj289/e6998/decision-study` | 既有 controller、模型、结果和 runtime；原绝对路径不变 |

GitHub 仓库是 `jzdscience/e6998_llm_genai_decision_study`。node 下载使用普通 HTTPS URL，
不依赖 Mac 上的 `github-personal` SSH alias。不要在原 89 GB 运行树中强行 checkout。

## 首次配置

在公司 code-server 终端复制执行整段。目标目录已经存在且非空时 clone 会停止，先核对它的来源，
不要删除已有目录后重试。本段固定 clone 当时取得的提交；若本轮交付指定了完整 SHA，
则以该 SHA 替换 `STUDY_COMMIT` 的赋值。

```bash
(
  set -eu
  cd /home/zhanj289/e6998
  git clone --branch main https://github.com/jzdscience/e6998_llm_genai_decision_study.git decision-study-src
  cd decision-study-src
  STUDY_COMMIT="$(git rev-parse HEAD)"
  git checkout --detach "$STUDY_COMMIT"
  /home/zhanj289/e6998/decision-study/.venv-controller/bin/python scripts/prepare_node_checkout.py \
    --runtime-root /home/zhanj289/e6998/decision-study \
    --expected-commit "$STUDY_COMMIT"
)
```

配置脚本要求当前 Git 工作区干净、HEAD 与指定 commit 一致，且源码与运行根不重叠。
它通过旧 controller 检查 Python 实际导入位置，确认加载的是新 clone 的 `decision_study`。
不需要重新安装旧 controller 的 editable package，也不复制虚拟环境。

它按 V2/V3 发布清单核对八份 prepared 文件，只复制目标缺失的文件；所有已有目标和源文件
先通过哈希检查才开始复制。目标不同、路径含 symlink、缺文件或哈希不符时停止，保留现场。
这八份文件只用于文件身份校验，不表示新 clone 拥有完整旧实验结果。

最后执行原 V2/V3 发布校验器和三个 CLI 的 `--help`。这一步不续 session、不下载模型、
不初始化 allocation，也不启动旧 pipeline。成功输出及本机验证记录见本文末尾。

## 后续更新

只在没有实验使用这个 checkout 时更新。先在 Mac 完成修改、验证、commit 和 push；
然后在 node 执行下面整段。它拒绝来源不同或有本地改动的 clone，并把取得的版本固定为
detached HEAD。node 用于运行，开发提交继续在 Mac 上完成。

```bash
(
  set -eu
  cd /home/zhanj289/e6998/decision-study-src
  test "$(git remote get-url origin)" = "https://github.com/jzdscience/e6998_llm_genai_decision_study.git" || {
    echo "STOP: origin differs; inspect this checkout."
    exit 2
  }
  test -z "$(git status --porcelain)" || {
    echo "STOP: local changes exist; preserve and inspect them."
    git status --short
    exit 2
  }
  git fetch origin main
  STUDY_COMMIT="$(git rev-parse refs/remotes/origin/main)"
  git checkout --detach "$STUDY_COMMIT"
  /home/zhanj289/e6998/decision-study/.venv-controller/bin/python scripts/prepare_node_checkout.py \
    --runtime-root /home/zhanj289/e6998/decision-study \
    --expected-commit "$STUDY_COMMIT"
)
```

后续已冻结实验若指定某个完整 commit，就使用该 commit，而不是自动选最新 main。
`git checkout --detach` 不要求工作区本来就干净，因此上面的显式检查不能省略。
参见 [Git switch/checkout 的 detached 语义](https://git-scm.com/docs/git-switch)。

## 与下一轮实验的关系

当前版本有已完成的 V1–V3 和 CPU Lab，还没有下一轮实验的实现与冻结协议。
配置成功表示 Git 源码和旧环境可以配合做后续开发与核验，不表示新的实验已经运行。
不要把 `run_v3.sh pipeline` 当成下一轮入口。

下一轮按现有顺序先补轻量分类基线，然后安排 StartLux9B 和加强性能测量。发布时需要给出
新的协议、固定 commit、数据角色、输出目录与运行命令；旧 holdout 的既有访问记录继续保留。
需要复用 V1 runtime 的新命令应显式使用原绝对 `--v1-dir`，新实验使用新的输出目录。

V2/V3 launcher 支持显式 controller，例如只验证帮助入口：

```bash
cd /home/zhanj289/e6998/decision-study-src
PYTHON_BIN=/home/zhanj289/e6998/decision-study/.venv-controller/bin/python bash scripts/run_v3.sh --help
```

完整数据边界见 [ARTIFACTS_ZH.md](ARTIFACTS_ZH.md)，总体流程见
[REPO_MANAGEMENT_ZH.md](REPO_MANAGEMENT_ZH.md)。

## 成功输出与已验证范围

成功时第一行是 `READY` 加实际完整 commit，随后打印 `Source:`、`Runtime:` 和复制计数。
收据保存在新 clone 的 `.validation/node-checkout/receipt.json`，不进入 Git。
2026-11-04 本机用原始八份 prepared 文件（约 20.2 MB）、临时 Git clone 和已有 controller
环境验证，首次与第二次运行分别得到以下实际计数：

```text
Prepared files: 8 copied, 0 reused
Prepared files: 0 copied, 8 reused
```

两次均通过原 V2/V3 发布校验和三个 CLI 帮助检查；临时旧运行根的八份数据及 session
哨兵文件共九项字节未变，新源码 clone 保持干净。14 项针对性测试通过，含目标冲突、
源哈希错误、源码导入错误、版本/工作区检查、路径别名和并发目标保护。
这属于本机 CPU 验证；公司 node 只能由用户在 code-server 执行，尚未收到 node 验收输出。
