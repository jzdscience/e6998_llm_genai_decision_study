"""Gradio 6 CPU Decision Lab; all inference shown here is archived evidence."""
from __future__ import annotations

import hashlib
from html import escape
import json
import os
from pathlib import Path
import tempfile

from .bundle import load_bundle, measured_performance, sample_replay, sandbox

MODEL_NAMES = {"kev": "Kev", "startlux": "StartLux", "qwen": "Qwen"}
DEPLOYMENT_NAMES = {
    "kev_direct": "Kev direct", "startlux_direct": "StartLux direct", "qwen_direct": "Qwen direct",
    "kev_to_qwen": "Kev → Qwen", "startlux_to_qwen": "StartLux → Qwen",
}
CSS = """
.gradio-container {width:100%!important;min-width:0!important;max-width:1280px!important; margin:auto!important; color:#172f46!important;
  background:#f6f8fc!important; font-family:Aptos,'Trebuchet MS',Arial,sans-serif!important;}
body {background:#f6f8fc;} .lab-hero {padding:22px 0 25px;}
.lab-hero h1 {font-size:clamp(29px,4.1vw,48px);line-height:1.1;letter-spacing:-1.5px;
  color:#172f46;margin:4px 0 14px;font-weight:700;max-width:900px;}
.lab-hero p {color:#546579;max-width:760px;font-size:16px;line-height:1.55;margin:0;}
.lab-title {font-size:18px;font-weight:700;color:#087e83;margin-bottom:14px;}
.lab-mode {display:inline-block;margin-top:17px;padding:6px 12px;border:1px solid #a7cbd0;
  border-radius:4px;font-size:13px;color:#215e6b;background:#eef7f7;}
.lab-panel {background:white;padding:22px 24px;min-height:320px;border-radius:7px;border:1px solid #d8e1eb;}
.lab-panel h2,.lab-section h2 {font-size:21px;line-height:1.25;margin:0 0 18px;color:#172f46;}
.lab-panel h3 {font-size:15px;margin:19px 0 7px;font-weight:700;color:#52667c;}
.lab-panel p {font-size:14px;line-height:1.6;color:#334b60;margin:0 0 12px;}
.lab-input {max-height:315px;overflow-y:auto;white-space:pre-wrap;line-height:1.65;font-size:14px;}
.lab-question {font-size:18px;line-height:1.45;margin:0 0 16px;color:#172f46;font-weight:600;}
.lab-caption {font-size:12px!important;color:#64748b!important;line-height:1.5!important;}
.lab-note {padding:13px 0;font-size:13px;color:#546579;line-height:1.6;max-width:1000px;}
.lab-large {font-size:42px;line-height:1.15;font-weight:700;letter-spacing:-1px;color:#087e83;}
.lab-confidence {display:flex;align-items:baseline;gap:10px;margin-bottom:17px;}
.lab-prob {margin:12px 0;} .lab-prob-label {display:flex;justify-content:space-between;gap:9px;font-size:13px;margin-bottom:5px;}
.lab-track {position:relative;height:9px;background:#e8eef4;border-radius:2px;}
.lab-track span {display:block;background:#0c8588;height:100%;border-radius:2px;}
.lab-gate {border-left:3px solid #172f46;padding:8px 12px;margin:20px 0 14px;background:#f3f6fa;font-size:13px;}
.lab-route {border-left:3px solid #087e83;padding:10px 14px;margin:14px 0;background:#eff8f7;}
.lab-route.deferred {border-color:#b35b16;background:#fff6ec;}
.lab-route strong {display:block;font-size:16px;line-height:1.4;margin-bottom:4px;}
.lab-route small {color:#52667c;line-height:1.5;}
.lab-result {font-size:24px;font-weight:700;line-height:1.3;margin:8px 0 14px;overflow-wrap:anywhere;}
.lab-verdict {display:inline-block;padding:4px 8px;font-size:12px;border-radius:3px;background:#e9f5ef;color:#1f6847;}
.lab-verdict.wrong {background:#fcefeb;color:#9b392b;}
.lab-section {padding:22px 24px;background:white;border:1px solid #d8e1eb;border-radius:7px;}
.lab-metrics {display:grid;grid-template-columns:repeat(4,minmax(100px,1fr));gap:20px;margin:22px 0;}
.lab-metrics strong {display:block;font-size:29px;font-weight:650;line-height:1.2;color:#172f46;}
.lab-metrics span {display:block;font-size:13px;color:#64748b;margin-top:7px;}
.lab-balance {display:flex;flex-wrap:wrap;gap:12px;font-size:14px;padding-top:13px;border-top:1px solid #dce5ee;}
.lab-table {width:100%;border-collapse:collapse;font-size:13px;margin:15px 0;}
.lab-table th {text-align:left;font-weight:600;color:#52667c;background:#f3f6fa;}
.lab-table td,.lab-table th {padding:12px 10px;border-bottom:1px solid #e2e9f0;}
.lab-table td:not(:first-child),.lab-table th:not(:first-child) {text-align:right;font-variant-numeric:tabular-nums;}
.lab-perf-row {display:grid;grid-template-columns:160px 1fr 70px;align-items:center;gap:12px;margin:18px 0;font-size:14px;}
.lab-perf-track {height:22px;background:#edf2f6;}
.lab-perf-track span {height:100%;display:block;background:#587897;}
.lab-perf-track span.startlux {background:#087e83;}.lab-perf-track span.cascade {background:#91b8bd;}
.lab-perf-value {text-align:right;font-weight:600;font-variant-numeric:tabular-nums;}
.lab-svg {width:100%;max-width:600px;height:auto;display:block;}
.lab-footer {font-size:12px;line-height:1.55;color:#64748b;padding:22px 0;}
button:focus-visible,a:focus-visible,input:focus-visible {outline:3px solid #0c8588!important;outline-offset:3px;}
@media(max-width:720px){
 .column,.row,.form,.block,.tabs,.tabitem{min-width:0!important;max-width:100%!important;}
 .row,.form{flex-direction:column!important;}
 .row>*,.form>*{min-width:0!important;width:100%!important;flex-basis:auto!important;}
 .tab-container,.tab-wrapper{min-width:0!important;max-width:100%;overflow-x:auto;}
 .lab-panel,.lab-section{overflow-wrap:anywhere;}
 .lab-panel{min-height:0;padding:18px;}.lab-metrics{grid-template-columns:repeat(2,1fr);}
 .lab-perf-row{grid-template-columns:125px 1fr 50px;font-size:12px;}.lab-table{font-size:11px;}
 .lab-table td,.lab-table th{padding:9px 5px;}.lab-hero{padding-top:12px;}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;}}
"""


def _text(value) -> str:
    return escape(str(value), quote=True)


def _label(value) -> str:
    return _text(value).replace("_", " ")


def _percent(value, decimals=2) -> str:
    return "Not defined" if value is None else f"{100 * value:.{decimals}f}%"


def render_case_panels(result: dict) -> tuple[str, str, str]:
    """Escape every evidence string before interpolating HTML."""
    state = result["state"]
    if state.startswith("Passage:\n") and "\n\nQuestion:\n" in state:
        passage, question = state[len("Passage:\n"):].rsplit("\n\nQuestion:\n", 1)
        input_body = f'<div class="lab-question">{_text(question)}</div><h3>Provided passage</h3><div class="lab-input">{_text(passage)}</div>'
    else:
        input_body = f'<div class="lab-question">{_text(state)}</div>'
    input_panel = f'<section class="lab-panel"><h2>The request</h2>{input_body}<h3>Task</h3><p>{"Given-passage answerability" if result["task"] == "squad" else "Banking intent or out of scope"}</p><p class="lab-caption">Saved {_text(result["task"])} example</p></section>'
    probabilities = result["probabilities"]
    if probabilities:
        ranked = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
        bars = "".join(f'<div class="lab-prob"><div class="lab-prob-label"><span>{_label(label)}</span><strong>{100*p:.2f}%</strong></div><div class="lab-track"><span style="width:{100*p:.5f}%"></span></div></div>' for label, p in ranked[:6])
        extra = f'<p class="lab-caption">Top 6 of {len(ranked)} labels. Full distribution in the evidence below.</p>' if len(ranked) > 6 else ""
        confidence = f'<div class="lab-confidence"><strong class="lab-large">{_percent(result["confidence"], 1)}</strong><span>highest probability</span></div>'
    else:
        bars, extra, confidence = "", "", '<p>The archived local request failed. It must be deferred.</p>'
    gate = result["gate"]
    gate_text = "Accept none" if gate.get("accept_none") else f'Accept when probability ≥ {gate["threshold"]:.6f}'
    prob_panel = f'<section class="lab-panel"><h2>{_text(MODEL_NAMES[result["model"]])} probabilities</h2>{confidence}{bars}{extra}<div class="lab-gate">{gate_text}</div><p class="lab-caption">{_text(result["condition"].capitalize())} probabilities · temperature {result["temperature"]:.4f}<br>Frozen validation gate; no future-risk guarantee.</p></section>'
    route = "Keep the local decision" if result["accepted"] else "Hand off to Qwen"
    detail = "Qwen was not used by this policy." if result["accepted"] else "Qwen supplies the final label for this policy."
    class_name = "" if result["accepted"] else " deferred"
    correct_class = "" if result["correct"] else " wrong"
    verdict = "Matches the reference label" if result["correct"] else "Does not match the reference label"
    decision = f'<section class="lab-panel"><h2>The decision</h2><div class="lab-route{class_name}"><strong>{route}</strong><small>{detail}</small></div><h3>Final label</h3><div class="lab-result">{_label(result["final_prediction"])}</div><span class="lab-verdict{correct_class}">{verdict}</span><p class="lab-caption">Reference: {_label(result["gold"])}</p><h3>Independent Qwen baseline</h3><p>{_label(result["qwen_prediction"])}</p><p class="lab-caption">Qwen probabilities are not provided by this interface. This is saved evidence, not a live call.</p></section>'
    return input_panel, prob_panel, decision


def render_sandbox_summary(result: dict) -> str:
    gate = result["gate"]
    label = "Saved frozen gate" if result["matches_frozen_gate"] else "Exploratory threshold"
    threshold = "accept none" if gate.get("accept_none") else f'{gate["threshold"]:.6f}'
    values = [(f'{result["n"]:,}', "unique inputs"), (_percent(result["coverage"]), "kept locally"),
              (_percent(result["selective_risk"]), "error among accepted"), (_percent(result["cascade_accuracy"]), "final accuracy")]
    metric_html = "".join(f'<div><strong>{value}</strong><span>{title}</span></div>' for value, title in values)
    stress_note = "CLINC stress reuses 450 main-test banking inputs and adds 850 OOS inputs; it is not independent of main." if result["workload"] == "stress" else ""
    return f'<section class="lab-section"><h2>{label}</h2><p class="lab-caption">Threshold {threshold} · {_text(result["condition"])} probabilities · temperature {result["temperature"]:.4f}</p><div class="lab-metrics">{metric_html}</div><div class="lab-balance"><span>{result["deferral_count"]:,} handed off</span><span>+{result["rescued"]} rescued</span><span>−{result["harmed"]} harmed</span><strong>{result["net_correct"]:+d} net correct</strong></div><p class="lab-note">Direct {_text(MODEL_NAMES[result["model"]])}: {_percent(result["local_accuracy"])}. Direct Qwen: {_percent(result["qwen_accuracy"])}.</p><p class="lab-caption">{stress_note} Previously viewed data. Exploring a threshold does not change the frozen study. No new throughput or latency is inferred.</p></section>'


def render_risk_curve(data: dict, model: str, task: str, workload: str, condition: str, current: dict) -> str:
    # A finite grid is explicitly an illustrative curve, not an optimization or
    # a claim to enumerate every confidence tie. Current and frozen points are exact.
    values = [sandbox(data, model, task, workload, i / 50, condition) for i in range(51)]
    frozen = sandbox(data, model, task, workload, condition=condition)
    values += [frozen, current]
    points = [r for r in values if r["selective_risk"] is not None]
    ymax = min(1.0, max(0.12, max((r["selective_risk"] for r in points), default=0) * 1.15))
    x = lambda coverage: 54 + 470 * coverage
    y = lambda risk: 238 - 185 * risk / ymax
    ordered = sorted(points, key=lambda r: (r["coverage"], r["selective_risk"]))
    polyline = " ".join(f'{x(r["coverage"]):.2f},{y(r["selective_risk"]):.2f}' for r in ordered)
    axes = '<path d="M54 42V238H530" stroke="#9baebe" fill="none"/>'
    labels = ""
    for fraction in (0, 0.5, 1):
        labels += f'<text x="{x(fraction)}" y="260" text-anchor="middle">{100*fraction:.0f}%</text>'
    for fraction in (0, 0.5, 1):
        value = ymax * fraction
        labels += f'<text x="45" y="{y(value)+4}" text-anchor="end">{100*value:.1f}%</text>'
    target = f'<path d="M54 {y(.05):.2f}H530" stroke="#c49250" stroke-dasharray="5 5"/><text x="530" y="{y(.05)-6:.2f}" text-anchor="end" fill="#9a6420">5% reference</text>'
    marks = ""
    for row, color, name, radius in ((frozen, "#172f46", "Frozen", 5), (current, "#087e83", "Selected", 7)):
        if row["selective_risk"] is not None:
            marks += f'<circle cx="{x(row["coverage"]):.2f}" cy="{y(row["selective_risk"]):.2f}" r="{radius}" fill="{color}" stroke="white" stroke-width="2"><title>{name} gate</title></circle>'
    svg = f'<svg class="lab-svg" viewBox="0 0 590 298" role="img" aria-label="Accepted error rate versus local coverage"><g font-family="Arial,sans-serif" font-size="11" fill="#52667c">{axes}{labels}{target}<polyline points="{polyline}" fill="none" stroke="#9cbfc5" stroke-width="2.5"/>{marks}<text x="290" y="290" text-anchor="middle">Inputs kept locally</text><text x="54" y="25">Error among accepted inputs</text></g></svg>'
    return f'<section class="lab-section"><h2>Coverage and accepted risk</h2>{svg}<p class="lab-caption">Teal: selected policy. Navy: frozen gate. Curve samples 51 thresholds; markers use exact rules. With no accepted inputs, risk is undefined and no marker is drawn. This is an empirical view, not a risk guarantee.</p></section>'


def render_performance(data: dict, task: str, concurrency: int) -> tuple[str, list[dict]]:
    rows = measured_performance(data, task, int(concurrency))
    maximum = max((r["throughput"] for r in rows), default=1)
    bars, table = "", ""
    for row in rows:
        deployment = row["deployment"]
        style = "cascade" if "_to_" in deployment else "startlux" if deployment == "startlux_direct" else ""
        name = _text(DEPLOYMENT_NAMES.get(deployment, deployment))
        bars += f'<div class="lab-perf-row"><span>{name}</span><div class="lab-perf-track"><span class="{style}" style="width:{100*row["throughput"]/maximum:.5f}%"></span></div><span class="lab-perf-value">{row["throughput"]:.2f}</span></div>'
        table += f'<tr><td>{name}</td><td>{row["throughput"]:.2f}</td><td>{row["p50_ms"]:.2f}</td><td>{row["p95_ms"]:.2f}</td><td>{_percent(row["fallback_fraction"])}</td><td>{_percent(row["failure_rate"])}</td></tr>'
    html = f'<section class="lab-section"><h2>Measured successful requests per second</h2>{bars}<div style="overflow-x:auto"><table class="lab-table"><thead><tr><th>Deployment</th><th>Requests/s</th><th>p50 ms</th><th>p95 ms</th><th>Handed off</th><th>Failures</th></tr></thead><tbody>{table}</tbody></table></div><p class="lab-note">Saved v1 benchmark on one L40S. Each task repeats the same 200 unique inputs. Values are arithmetic means of three rounds; mean p95 is not a pooled percentile or a confidence interval.</p><p class="lab-caption">Closed-loop client concurrency {_text(concurrency)}. Request timing starts inside the controller worker, excluding its prior queue, service startup and warmup. Measured cascades use the frozen refitted gate. Sandbox thresholds have no measured performance here.</p></section>'
    return html, rows


def create_app(bundle_dir: Path | str):
    """Return a configured Blocks instance; loading does not launch a server."""
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the optional Lab dependency: pip install 'decision-study[lab]'") from exc
    data = load_bundle(bundle_dir)
    exports = tempfile.TemporaryDirectory(prefix="decision-lab-public-")

    def export_trace(result):
        body = json.dumps({**result, "provenance": data["provenance"]}, ensure_ascii=False, indent=2, allow_nan=False)
        path = Path(exports.name) / f'decision-{hashlib.sha256(body.encode()).hexdigest()[:20]}.json'
        if not path.exists():
            path.write_text(body + "\n", encoding="utf-8")
        return str(path)

    def show_case(sample_id, model, condition):
        result = sample_replay(data, model, sample_id, condition)
        return (*render_case_panels(result), result, export_trace(result))

    def show_sandbox(model, workload_name, condition, use_frozen, threshold, reject_all):
        task, workload = workload_name.split("/")
        result = sandbox(data, model, task, workload, None if use_frozen else float(threshold), condition, False if use_frozen else reject_all)
        return render_sandbox_summary(result), render_risk_curve(data, model, task, workload, condition, result), result

    cases = [(case["label"], case["sample_id"]) for case in data["cases"]]
    default_sid = cases[0][1] if cases else next(iter(data["samples"]))
    first = show_case(default_sid, "startlux", "refitted")
    initial_sandbox = show_sandbox("startlux", "squad/main", "refitted", True, .6561214813327713, False)
    initial_performance, performance_rows = render_performance(data, "squad", 8)
    with gr.Blocks(title="Decision Lab | Frozen model evidence", analytics_enabled=False, delete_cache=(3600, 3600)) as app:
        gr.HTML('<header class="lab-hero"><div class="lab-title">Decision Lab</div><h1>When should a model hand off?</h1><p>Follow a saved decision, explore the quality tradeoff, and compare the service performance that was actually measured.</p><span class="lab-mode">Recorded v1 evidence · CPU only · No model calls</span></header>')
        with gr.Tabs():
            with gr.Tab("Cases", id="cases"):
                with gr.Row():
                    story = gr.Dropdown(choices=cases, value=default_sid, label="Four illustrative paths", scale=2)
                    model = gr.Dropdown(choices=[("StartLux", "startlux"), ("Kev", "kev")], value="startlux", label="Local model")
                    condition = gr.Dropdown(choices=[("Refitted temperature", "refitted"), ("Released probabilities", "released")], value="refitted", label="Saved calibration")
                gr.HTML('<div class="lab-note">These four cases were selected by known outcomes under StartLux’s frozen refitted gate. Switching the model or calibration can change their paths. They illustrate behavior; the full workloads establish accuracy.</div>')
                with gr.Row(equal_height=True):
                    request_html = gr.HTML(first[0], scale=5, min_width=300)
                    probabilities_html = gr.HTML(first[1], scale=4, min_width=270)
                    decision_html = gr.HTML(first[2], scale=4, min_width=270)
                with gr.Accordion("Browse all saved test and stress inputs", open=False):
                    all_choices = [(f'{sample["task"]} · {sid} · {sample["state"][-90:].replace(chr(10), " ")}', sid) for sid, sample in sorted(data["samples"].items())]
                    browser = gr.Dropdown(choices=all_choices, value=default_sid, label="Search by ID or input text", filterable=True)
                    gr.Markdown("Only saved inputs with all three model records are listed. Calibration and validation are excluded because their Qwen predictions were not collected in v1.")
                with gr.Accordion("Evidence for this decision", open=False):
                    trace = gr.JSON(first[3], label="Normalized saved records and routing rule")
                    download = gr.DownloadButton("Download this decision as JSON", value=first[4], size="sm")
                outputs = [request_html, probabilities_html, decision_html, trace, download]
                active_sid = gr.State(default_sid)

                def select_case(sid, local_model, probability_condition):
                    return (*show_case(sid, local_model, probability_condition), sid)

                story.change(select_case, [story, model, condition], outputs + [active_sid], api_name="replay_case")
                browser.change(select_case, [browser, model, condition], outputs + [active_sid], api_name="browse_case")
                model.change(show_case, [active_sid, model, condition], outputs, api_name="replay_model")
                condition.change(show_case, [active_sid, model, condition], outputs, api_name="replay_calibration")
            with gr.Tab("Policy sandbox", id="sandbox"):
                gr.HTML('<div class="lab-note">Explore previously viewed data. The saved gate remains a reference, and none of these controls changes the formal study.</div>')
                with gr.Row():
                    smodel = gr.Dropdown([("StartLux", "startlux"), ("Kev", "kev")], value="startlux", label="Local model")
                    workload = gr.Dropdown([("SQuAD main · 2,000 inputs", "squad/main"), ("CLINC main · 600 inputs", "clinc/main"), ("CLINC OOS stress · 1,300 inputs", "clinc/stress")], value="squad/main", label="Saved workload")
                    scondition = gr.Dropdown([("Refitted temperature", "refitted"), ("Released probabilities", "released")], value="refitted", label="Probability scale")
                with gr.Row():
                    frozen = gr.Checkbox(value=True, label="Use saved frozen gate")
                    reject = gr.Checkbox(value=False, label="Defer every request (exploration only)", interactive=False)
                    threshold = gr.Slider(0, 1, value=.6561214813327713, step=.001, label="Exploratory acceptance threshold", interactive=False, scale=2)
                with gr.Row(equal_height=True):
                    sandbox_html = gr.HTML(initial_sandbox[0], scale=1, min_width=360)
                    risk_html = gr.HTML(initial_sandbox[1], scale=1, min_width=360)
                with gr.Accordion("Exact policy and counts", open=False):
                    sandbox_json = gr.JSON(initial_sandbox[2])
                inputs = [smodel, workload, scondition, frozen, threshold, reject]
                soutputs = [sandbox_html, risk_html, sandbox_json]
                for index, control in enumerate((smodel, workload, scondition, frozen, reject)):
                    control.change(show_sandbox, inputs, soutputs, api_name="explore_policy" if index == 0 else False)
                threshold.release(show_sandbox, inputs, soutputs, api_name="explore_threshold")
                frozen.change(lambda yes: (gr.Slider(interactive=not yes), gr.Checkbox(value=False, interactive=not yes)), frozen, [threshold, reject], api_name=False)
            with gr.Tab("Measured performance", id="performance"):
                with gr.Row():
                    ptask = gr.Dropdown([("SQuAD answerability", "squad"), ("CLINC banking", "clinc")], value="squad", label="Benchmark workload")
                    concurrency = gr.Radio([1, 8], value=8, label="Measured client concurrency")
                perf_html = gr.HTML(initial_performance)
                with gr.Accordion("Measured values and source identity", open=False):
                    perf_json = gr.JSON(performance_rows, label="Three-round descriptive means")
                    gr.JSON(data["provenance"], label="Source manifest and configuration hashes")
                ptask.change(lambda task, count: render_performance(data, task, int(count)), [ptask, concurrency], [perf_html, perf_json], api_name="measured_performance")
                concurrency.change(lambda task, count: render_performance(data, task, int(count)), [ptask, concurrency], [perf_html, perf_json], api_name=False)
        total = sum(data["metrics"]["prediction_counts"].values())
        gr.HTML(f'<footer class="lab-footer">Source study: {total:,} formal predictions. This portable viewer contains {len(data["samples"]):,} unique test/stress inputs. Qwen probabilities are unavailable. No new v2 experimental results are represented.</footer>')
    app.queue(max_size=16, default_concurrency_limit=1)
    app._decision_lab_exports = exports  # Retain public temporary exports for this app's lifetime.
    return app


def launch(bundle_dir: Path | str, host: str = "127.0.0.1", port: int = 7860):
    """Serve only the CPU viewer. Public sharing and model endpoints are disabled."""
    app = create_app(bundle_dir)
    import gradio as gr
    theme = gr.themes.Base(primary_hue="teal", neutral_hue="slate", radius_size="sm",
                           font=("Aptos", "Trebuchet MS", "Arial", "sans-serif"))
    return app.launch(server_name=host, server_port=port, share=False, inbrowser=False,
                      css=CSS, theme=theme, footer_links=[], show_error=False,
                      enable_monitoring=False, mcp_server=False, pwa=False)
