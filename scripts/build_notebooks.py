#!/usr/bin/env python3
"""Build the two portable, thin notebooks; experiment logic lives in the package."""
from pathlib import Path
import textwrap
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text, *, parameters=False):
    cell = nbf.v4.new_code_cell(textwrap.dedent(text).strip())
    if parameters:
        cell.metadata["tags"] = ["parameters"]
    return cell


BOOTSTRAP = r'''
from pathlib import Path
import importlib
import os
import stat
import subprocess
import sys
import tempfile
import zipfile

def locate_source(explicit=""):
    candidates = ([Path(explicit).expanduser()] if explicit else []) + [
        Path.cwd(), Path.cwd().parent,
        Path.cwd() / "decision-study", Path.cwd().parent / "decision-study",
    ]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "decision_study").is_dir():
            return candidate.resolve()
    return None

def extract_source_zip(archive):
    destination = Path(tempfile.mkdtemp(prefix="decision-study-source-"))
    with zipfile.ZipFile(archive) as bundle:
        if sum(item.file_size for item in bundle.infolist()) > 512 * 1024**2:
            raise ValueError("Source archive exceeds 512 MiB. Use the code bundle, not model weights.")
        for item in bundle.infolist():
            name = item.filename.replace("\\", "/")
            relative = Path(name)
            if (relative.is_absolute() or ".." in relative.parts or
                    (relative.parts and ":" in relative.parts[0]) or
                    stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError(f"Unsafe archive entry: {item.filename}")
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Archive path escapes extraction directory.")
        bundle.extractall(destination)
    candidates = [destination] + [p for p in destination.iterdir() if p.is_dir()]
    matches = [p for p in candidates if (p / "pyproject.toml").is_file() and (p / "decision_study").is_dir()]
    if len(matches) != 1:
        raise ValueError("Expected one decision-study project in the source ZIP.")
    return matches[0].resolve()

PROJECT_DIR = locate_source(PACKAGE_DIR)
if PROJECT_DIR is None:
    archive = Path(PACKAGE_ZIP).expanduser() if PACKAGE_ZIP else Path.cwd() / "decision-study-bundle.zip"
    if not archive.is_file():
        if PLATFORM != "colab":
            raise FileNotFoundError("Set PACKAGE_DIR to the extracted project or PACKAGE_ZIP to decision-study-bundle.zip.")
        from google.colab import files
        print("Upload decision-study-bundle.zip (the code package).")
        uploaded = files.upload()
        candidates = [Path(name) for name in uploaded if name.endswith(".zip") and "decision-study" in name]
        if len(candidates) != 1:
            raise ValueError("Upload exactly one decision-study code ZIP.")
        archive = candidates[0]
    PROJECT_DIR = extract_source_zip(archive)
if INSTALL_PACKAGE:
    install_command = [sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT_DIR)]
    if globals().get("INSTALL_RUNTIME_TOOLS", False):
        install_command.append("uv")
    subprocess.run(install_command, check=True)
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, str(PROJECT_DIR))
importlib.invalidate_caches()
print("Experiment package:", PROJECT_DIR)
'''


def build_run():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
        # Decision Model 实验：公司 code-server／Colab／云 VM 运行入口

        本 notebook 调用同一套 Python 实验包，负责准备数据、启动隔离的模型服务、恢复预测和导出结果。
        **第一次只跑 `smoke`。拿到 `pilot_bundle.zip` 后先检查验收结果，再跑正式预测。**

        默认针对公司 code-server 的单张 L40S 48 GB、24 小时可续期 session。也可选择 Colab GPU runtime 或 NVIDIA 云 VM。
        使用前准备 `decision-study-bundle.zip`；需允许访问官方数据与模型源，且结果目录应位于可跨 session 保留的存储。
        不需要在 notebook kernel 中手工安装 Torch、vLLM 或三个模型的依赖。

        真实 GPU 兼容性由 smoke 确认；本包的 CPU 测试通过不等于你的 GPU 已通过验收。
        """),
        md("""
        ## 1. 只修改这些参数

        - 公司 code-server 保留 `PLATFORM="code-server"`、`COST_MODE="provided"`、`HOURLY_USD=0`；记录 GPU 时间，不虚构费用。
        - 付费 VM／Colab 使用对应 `PLATFORM` 和 `COST_MODE="metered"`，填写正数 `HOURLY_USD`；Colab 用 compute-unit 美元等价估计。
        - 新 24 小时 session 默认 `SESSION_HOURS_LEFT=22`，预留两小时交接。已有 session 应填写实际剩余可用时间以内的值。
          只有平台已经续期／重新分配时才设 `NEW_SESSION=True`，执行后改回 `False`；重跑命令不会自动延长原截止时间。
        - 第一次 `JOB="smoke"`。之后依次 `kev_full`、`startlux_full`、`qwen_test`，最后 `benchmark`。
        - 48 GB 主方案保留 `GPU_MEMORY_UTILIZATION=0.40`。24 GB Colab 单模型预测可在第一次 smoke 前设为 `0.80`；
          该设置不能直接用于同卡双模型 cascade。更换 GPU 型号需要新的输出目录并重新运行质量实验。
        - 所有阶段使用**同一个 `OUTPUT_DIR`**，恢复自动开启。要开始独立实验，选择一个新目录。
        - CPU 检查可设 `EXECUTE_GPU=False`；这不会生成模型实验结果。
        """),
        code('''
        PLATFORM = "code-server"                   # "code-server", "colab" or "vm"
        JOB = "smoke"                              # smoke | kev_full | startlux_full | qwen_test | benchmark
        COST_MODE = "provided" if PLATFORM == "code-server" else "metered"
        HOURLY_USD = 0.0 if COST_MODE == "provided" else None  # metered: replace None with actual positive USD/hour estimate
        SESSION_HOURS_LEFT = 22.0                   # <= real remaining time; 22 reserves 2h of a new 24h allocation
        NEW_SESSION = False                        # True ONLY after provider confirms renewed/new allocation; then reset False
        GPU_MEMORY_UTILIZATION = 0.40               # 48 GB dual residency; 24 GB standalone only: 0.80 before first smoke
        OUTPUT_DIR = ""                            # blank: Drive/.../solo-v1 on Colab, ./results/solo-v1 on VM
        PACKAGE_DIR = ""                           # existing extracted project; optional
        PACKAGE_ZIP = ""                           # code ZIP path; blank prompts upload in Colab
        EXECUTE_GPU = True
        RUN_PREPARE = True
        INSTALL_PACKAGE = True
        INSTALL_RUNTIME_TOOLS = True               # installs uv for isolated GPU environments
        DOWNLOAD_RESULT = True                     # browser download on Colab after export
        ''', parameters=True),
        code('''
        import math
        VALID_JOBS = {"smoke", "kev_full", "startlux_full", "qwen_test", "benchmark"}
        if PLATFORM not in {"code-server", "colab", "vm"} or JOB not in VALID_JOBS:
            raise ValueError("Check PLATFORM and JOB against the values listed above.")
        if COST_MODE not in {"provided", "metered"}:
            raise ValueError("COST_MODE must be provided or metered.")
        if EXECUTE_GPU and (not isinstance(HOURLY_USD, (int, float)) or not math.isfinite(HOURLY_USD)):
            raise ValueError("HOURLY_USD must be finite: 0 for provided allocation, positive for metered compute.")
        if EXECUTE_GPU and COST_MODE == "provided" and HOURLY_USD != 0:
            raise ValueError("Set HOURLY_USD=0 for provided allocation; no dollar charges are inferred.")
        if EXECUTE_GPU and COST_MODE == "metered" and HOURLY_USD <= 0:
            raise ValueError("Enter a positive actual HOURLY_USD for metered GPU work.")
        if not isinstance(SESSION_HOURS_LEFT, (int, float)) or not math.isfinite(SESSION_HOURS_LEFT) or not 0 < SESSION_HOURS_LEFT <= 24:
            raise ValueError("SESSION_HOURS_LEFT must be positive and no more than the real remaining 24h allocation.")
        if not isinstance(GPU_MEMORY_UTILIZATION, (int, float)) or not math.isfinite(GPU_MEMORY_UTILIZATION) or not 0 < GPU_MEMORY_UTILIZATION < 1:
            raise ValueError("GPU_MEMORY_UTILIZATION must be a finite fraction strictly between 0 and 1.")
        if JOB == "benchmark" and GPU_MEMORY_UTILIZATION > 0.40:
            raise ValueError("The main same-GPU cascade benchmark requires the validated 0.40 residency configuration. Use a new matched-GPU run if changing hardware/configuration.")
        print("Mode:", "REAL GPU EXPERIMENT" if EXECUTE_GPU else "CPU CHECK ONLY — no model results")
        '''),
        md("""
        ## 2. Load the experiment package

        Colab will ask for the code ZIP if it cannot find the package. The archive is checked for unsafe paths before extraction.
        In code-server or on a VM, open this notebook inside the extracted project, or set `PACKAGE_DIR`.
        """),
        code(BOOTSTRAP),
        md("""
        ## 3. Connect persistent results storage

        公司 code-server 请将 `OUTPUT_DIR` 指向会话续期后仍保留的挂载目录。若存储不保留，每阶段导出并下载 ZIP，续期前完成备份。
        不确定存储策略时先确认目录能跨 session 保留，或使用已验证的外部持久存储。

        Colab keeps small experiment artifacts on Drive. Model weights and isolated environments stay on local runtime disk;
        after a runtime reset they can be rebuilt from the saved lock. A disconnect can lose the latest unsynchronized shard,
        which will be rerun on recovery. Do not delete the output folder between jobs.
        """),
        code('''
        if PLATFORM == "colab":
            from google.colab import drive
            drive.mount("/content/drive")
            os.environ["DECISION_STUDY_RUNTIME_DIR"] = "/content/decision-study-runtime"
            result_dir = Path(OUTPUT_DIR or "/content/drive/MyDrive/decision-study/solo-v1")
        else:
            result_dir = Path(OUTPUT_DIR).expanduser() if OUTPUT_DIR else PROJECT_DIR / "results" / "solo-v1"
        result_dir = result_dir.resolve()
        result_dir.mkdir(parents=True, exist_ok=True)

        def cli(command, *arguments, check=True):
            args = [sys.executable, "-m", "decision_study", command,
                    "--output-dir", str(result_dir), *map(str, arguments)]
            print("Running:", command)
            return subprocess.run(args, cwd=PROJECT_DIR, check=check)

        print("Persistent results:", result_dir)
        '''),
        md("""
        ## 4. Inspect the machine and prepare official data

        Data preparation verifies the frozen source hashes, isolates contradictory SQuAD records, and creates deterministic splits.
        It never downloads model weights or starts GPU inference. No paragraph is truncated.
        """),
        code('''
        doctor_result = cli("doctor", check=False)
        if RUN_PREPARE:
            cli("prepare")
            from decision_study.data import verify_data
            manifest = verify_data(result_dir)
            print("Dataset fingerprint:", manifest["fingerprint"])
            print("Counts:", manifest["counts"])
        else:
            print("Data preparation skipped by parameter; this is not a complete experiment setup.")
        '''),
        md("""
        ## 5. Run the selected stage

        The runner freezes revisions, creates isolated environments, performs token-length preflight and checks smoke results
        before formal prediction. A failed check returns diagnostics instead of silently changing models or precision.
        Initialization and kernel compilation can take time; follow the stage logs.

        Re-run the same cell with the same output directory to resume. Completed terminal success/failure records are retained;
        recovery does not grant unlimited retries to failed examples.
        """),
        code('''
        job_returncode = None
        if EXECUTE_GPU:
            if NEW_SESSION:
                cli("session", "--renew-session", "--session-hours-left", SESSION_HOURS_LEFT)
                NEW_SESSION = False
                print("New session deadline recorded. Set NEW_SESSION=False before the next run.")
            completed = cli("run", "--job", JOB, "--hourly-usd", HOURLY_USD,
                            "--cost-mode", COST_MODE, "--session-hours-left", SESSION_HOURS_LEFT,
                            "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
                            "--project-dir", PROJECT_DIR, check=False)
            job_returncode = completed.returncode
            if job_returncode:
                print("Stage did not complete. Export the diagnostic bundle below before changing configuration.")
            else:
                print("Stage finished; consult status.json for checks and completeness.")
        else:
            print("GPU execution disabled. No predictions, calibration claims or performance measurements were produced.")
        '''),
        md("""
        ## 6. Export and send back the result ZIP

        `pilot_bundle.zip` contains setup/smoke evidence; `experiment_bundle.zip` contains per-example predictions and calibration;
        `benchmark_bundle.zip` adds live timing traces. Partial bundles carry an explicit status and are useful for diagnosis.

        **After saving the bundle, stop your cloud VM or disconnect/delete the Colab runtime when no GPU work remains.**
        A finished Python process does not automatically stop VM billing.
        公司提供的 allocation 不估算美元账单；仍须在实际 session 结束前保存结果。续期后恢复不代表可以省略备份。
        """),
        code('''
        import json
        export_kind = "pilot" if JOB == "smoke" or not EXECUTE_GPU else "benchmark" if JOB == "benchmark" else "experiment"
        export_result = cli("export", "--kind", export_kind, check=False)
        status_path = result_dir / "status.json"
        if status_path.exists():
            print(status_path.read_text(encoding="utf-8"))
        candidates = sorted(result_dir.rglob(f"{export_kind}_bundle.zip"), key=lambda p: p.stat().st_mtime)
        if candidates:
            bundle_path = candidates[-1]
            print("Send this file back:", bundle_path)
            if PLATFORM == "colab" and DOWNLOAD_RESULT:
                from google.colab import files
                files.download(str(bundle_path))
        else:
            print("No bundle found. Keep the output folder and send the displayed export error.")
        '''),
        md("""
        ## What to run next

        1. First return the pilot ZIP. Resolve any blocked checks before a full job.
        2. Run `kev_full`, `startlux_full`, and `qwen_test` into the same folder. Their combined formal prediction count is 15,950.
        3. Use `Analyze_and_Demo.ipynb` on CPU to check the full experiment bundle and produce figures.
        4. Run `benchmark` on the fixed GPU used for final performance comparison, then return its ZIP.

        A high error rate or zero local coverage is a valid result. Do not change prompts or thresholds after inspecting test outcomes.
        """),
    ]
    return nb


def build_analysis():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
        # Decision Model 实验：结果复算与 Demo

        本 notebook 从导出的结果 ZIP 复算统计、显示图表和回放真实记录。默认只用 CPU，不启动模型服务。
        图表来自结果包中的逐样本预测；结果不完整时停止分析并列出缺失项，不补造数据。

        需要 `decision-study-bundle.zip`（代码）和 pilot／experiment／benchmark 结果 ZIP。
        """),
        md("""
        ## 1. Parameters

        用新的 `OUTPUT_DIR` 导入一个结果包，避免混入其他实验。`RESULT_ZIP` 留空时，Colab 会提示上传；
        VM 上请填写实际文件路径。已经导入过的结果或原始实验目录可设 `USE_EXISTING_OUTPUT=True`，
        不再覆盖导入；实时 demo 使用这一模式，仅在最后一节显式开启。
        """),
        code('''
        PLATFORM = "code-server"                   # "code-server", "colab" or "vm"
        PACKAGE_DIR = ""
        PACKAGE_ZIP = ""
        RESULT_ZIP = ""
        OUTPUT_DIR = ""                            # blank: ./decision-study-analysis
        USE_EXISTING_OUTPUT = False                 # True for reruns or the original experiment directory; no ZIP import
        INSTALL_PACKAGE = True
        INSTALL_RUNTIME_TOOLS = False              # CPU result analysis does not need uv
        RECOMPUTE = True                            # formal analysis uses 2,000 bootstrap replicates
        DEMO_MODEL = "kev"                         # "kev" or "startlux"
        SAMPLE_ID = None                            # None: choose a saved case deterministically
        RUN_LIVE = False                            # requires compatible GPU + frozen model/gate configuration
        LIVE_TASK = "clinc"                        # "clinc" or "squad"
        LIVE_TEXT = "I need to check my account balance."
        LIVE_PASSAGE = "The museum opened in 1984."
        LIVE_QUESTION = "When did the museum open?"
        ''', parameters=True),
        md("## 2. Load the same experiment code"),
        code(BOOTSTRAP),
        md("""
        ## 3. Import the verified result bundle

        Import checks the bundle manifest and file checksums. Model weights and credentials are not part of the result bundle.
        Status remains explicit if only the pilot or a subset of formal jobs is present.
        """),
        code('''
        import json
        from decision_study.artifacts import import_bundle, bundle_readiness
        result_dir = (Path(OUTPUT_DIR).expanduser() if OUTPUT_DIR else Path.cwd() / "decision-study-analysis").resolve()
        if USE_EXISTING_OUTPUT:
            from decision_study.data import verify_data
            verified = verify_data(result_dir)
            imported = {"mode": "existing output; no import or overwrite", "data_fingerprint": verified["fingerprint"]}
        else:
            if RESULT_ZIP:
                archive = Path(RESULT_ZIP).expanduser()
            elif PLATFORM == "colab":
                from google.colab import files
                print("Upload one pilot_bundle.zip, experiment_bundle.zip or benchmark_bundle.zip.")
                uploaded = files.upload()
                candidates = [Path(name) for name in uploaded if name.endswith("_bundle.zip")]
                if len(candidates) != 1:
                    raise ValueError("Upload exactly one result bundle.")
                archive = candidates[0]
            else:
                raise ValueError("Set RESULT_ZIP, or use USE_EXISTING_OUTPUT=True with a prepared results directory.")
            imported = import_bundle(archive, result_dir)
        readiness = bundle_readiness(result_dir)
        print(json.dumps({"import": imported, "readiness": readiness}, ensure_ascii=False, indent=2))
        '''),
        md("""
        ## 4. Recompute statistics on CPU

        Calibration and validation gates remain frozen. Test uncertainty uses paired resampling (article clusters for SQuAD).
        The evaluator keeps failures in accuracy denominators and reports probability metrics only for valid distributions.
        This cell does not make model calls.
        """),
        code('''
        if readiness["experiment_ready"] and RECOMPUTE:
            subprocess.run([sys.executable, "-m", "decision_study", "evaluate", "--output-dir", str(result_dir)],
                           cwd=PROJECT_DIR, check=True)
            print("Recomputed from saved predictions. No GPU inference was run.")
        elif not readiness["experiment_ready"]:
            print("Analysis skipped: the formal experiment is incomplete.")
            print(json.dumps(readiness.get("missing", []), ensure_ascii=False, indent=2))
        else:
            print("Recomputation disabled; figures below, if any, are the saved bundle artifacts.")
        '''),
        md("""
        ## 5. Inspect result figures and exact-value tables

        Read risk and coverage together. Calibration improvements need not improve ranking, especially in the binary task.
        Cached predictions support accuracy analysis; only live benchmark traces support latency claims.
        """),
        code('''
        from IPython.display import display, Image, Markdown
        import csv
        if readiness["experiment_ready"]:
            presentation_dirs = [result_dir / "analysis"]
            if readiness["benchmark_ready"]:
                presentation_dirs.append(result_dir / "benchmark")
            figures = sorted(p for folder in presentation_dirs for p in folder.rglob("*.png"))
            for figure in figures:
                display(Markdown(f"**{figure.stem.replace('_', ' ')}**"))
                display(Image(filename=str(figure), width=950))
            if not figures:
                print("No analysis figures found; inspect the evaluate output above.")
            for table in sorted(p for folder in presentation_dirs for p in folder.rglob("*.csv")):
                with table.open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    preview = [row for _, row in zip(range(5), reader)]
                print("Table:", table.relative_to(result_dir), "(first 5 rows)")
                display(preview)
        else:
            print("No formal result figures displayed for an incomplete experiment.")
        '''),
        md("""
        ## 6. Recorded demo — explicitly a replay

        The example below comes from saved predictions and the frozen gate. It is not a fresh model call.
        Change `SAMPLE_ID` to inspect a particular input. Replays can show failures and harmful fallback as well as successful decisions.
        """),
        code('''
        from decision_study.demo import replay
        if readiness["experiment_ready"]:
            replay_result = replay(result_dir, DEMO_MODEL, sample_id=SAMPLE_ID)
            print("RECORDED REPLAY — no live inference")
            display(replay_result)
        else:
            print("Replay skipped: complete the formal prediction jobs first.")
        '''),
        md("""
        ## 7. Optional live demo

        Enable `RUN_LIVE=True` only on the **original matched experiment GPU/runtime**, with its model environments and weights still available,
        after completing calibration and smoke. Set `OUTPUT_DIR` to that original experiment directory and `USE_EXISTING_OUTPUT=True`.
        A result ZIP imported into a fresh CPU machine does not install the GPU environments or weights and is sufficient only for analysis/replay.
        This starts real decision and Qwen services and consumes GPU time. It uses the same adapters and frozen gate as the experiments.
        Enter text for CLINC, or a complete passage and question for SQuAD. Stop the VM when finished.
        """),
        code('''
        if RUN_LIVE:
            if not USE_EXISTING_OUTPUT:
                raise RuntimeError("For live inference, use the original experiment OUTPUT_DIR with USE_EXISTING_OUTPUT=True.")
            if not readiness["experiment_ready"]:
                raise RuntimeError("Live demo requires the completed experiment and frozen gates.")
            if PLATFORM == "colab":
                os.environ["DECISION_STUDY_RUNTIME_DIR"] = "/content/decision-study-runtime"
            from decision_study.demo import live
            display(live(PROJECT_DIR, result_dir, DEMO_MODEL, LIVE_TASK,
                         text=LIVE_TEXT, passage=LIVE_PASSAGE, question=LIVE_QUESTION))
        else:
            print("Live demo disabled. This notebook has only inspected/recomputed saved results.")
        '''),
        md("""
        ## Send back / write up

        Keep the original result ZIP plus the generated `analysis` directory. For interpretation, return the full per-example bundle,
        not only screenshots or summary accuracy. The next discussion can use the plots, paired differences, selected errors and live
        cost traces to decide the report's main findings. No positive result is assumed.
        """),
    ]
    return nb


def main():
    output = ROOT / "notebooks"
    output.mkdir(exist_ok=True)
    for name, notebook in (("Run_Experiments.ipynb", build_run()), ("Analyze_and_Demo.ipynb", build_analysis())):
        notebook.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"name": name, "provenance": []},
        }
        nbf.validate(notebook)
        nbf.write(notebook, output / name)
        print(output / name)


if __name__ == "__main__":
    main()
