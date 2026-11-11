"""CPU viewer for immutable V1–V3 evidence and the accepted follow-ups; no fitting or model requests."""
from __future__ import annotations

from html import escape
import math
import os
from pathlib import Path

from ..v2.lab.app import CSS as V1_CSS, create_app as create_v1_app
from ..v2.lab.bundle import load_bundle as load_v1_bundle


CSS = V1_CSS + """
.ds-hero {padding:20px 0 24px;max-width:900px;}
.ds-hero h1 {font-size:clamp(30px,4.5vw,48px);letter-spacing:-1.5px;line-height:1.12;
  margin:8px 0 16px;color:#172f46;font-weight:700;}
.ds-hero p,.ds-section p {max-width:78ch;line-height:1.65;color:#506276;}
.ds-brand {font-size:17px;font-weight:700;color:#087e83;}
.ds-status {font-size:13px;color:#546579;margin-top:15px;}
.ds-section {background:#fff;padding:25px 28px;margin:10px 0 20px;border:1px solid #d8e1eb;
  border-radius:7px;min-width:0;}
.ds-section h2 {font-size:23px;line-height:1.3;margin:0 0 14px;color:#172f46;}
.ds-section h3 {font-size:17px;margin:22px 0 10px;}
.ds-caption {font-size:12px;color:#64748b;line-height:1.6;max-width:90ch;}
.ds-ci {width:100%;height:auto;display:block;max-height:290px;}
.ds-ci-mobile,.ds-scroll-hint {display:none;}
.ds-ci-mobile-item {padding:10px 0 15px;}
.ds-ci-mobile-label {font-size:14px;color:#172f46;line-height:1.6;margin:0;}
.ds-ci-mobile-label strong {margin-right:12px;}
.ds-ci-mobile-interval {font-size:12px;color:#52667c;line-height:1.6;}
.ds-ci-mobile svg {width:100%;height:auto;display:block;}
.ds-zero {stroke:#7a8da3;stroke-width:1;stroke-dasharray:5 5;}
.ds-table-wrap {overflow-x:auto;max-width:100%;}
.ds-table {width:100%;border-collapse:collapse;font-size:13px;margin:14px 0;font-variant-numeric:tabular-nums;}
.ds-table th {background:#f0f4f9;font-weight:600;text-align:left;color:#52667c;}
.ds-table th,.ds-table td {padding:11px 10px;border-bottom:1px solid #e0e7ef;vertical-align:top;}
.ds-table td.ds-number,.ds-table th.ds-number {text-align:right;white-space:nowrap;}
.ds-primary td:first-child {border-left:3px solid #087e83;}
.ds-policy-role {display:block;font-size:11px;color:#52667c;margin-top:4px;}
.ds-callout {border-left:4px solid #087e83;background:#eef7f7;padding:13px 17px;margin:18px 0;line-height:1.6;}
.ds-callout.ds-uncertain {border-color:#4467a1;background:#eff3f9;}
.ds-case-grid {display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);gap:22px;}
.ds-input {white-space:pre-wrap;overflow-wrap:anywhere;max-height:380px;overflow:auto;
  line-height:1.65;font-size:14px;color:#334b60;}
.ds-case-meta {font-size:12px;color:#64748b;overflow-wrap:anywhere;line-height:1.6;margin:10px 0;}
.ds-route {background:#f0f4f9;border-left:4px solid #4467a1;padding:15px 17px;margin:15px 0;}
.ds-route strong {display:block;font-size:20px;margin-bottom:5px;}
.ds-route.ds-correct {border-color:#087e83;background:#eef7f7;}
.ds-route.ds-wrong {border-color:#9b466a;background:#fbf0f4;}
.ds-prob {margin:11px 0;font-size:13px;}
.ds-prob-label {display:flex;justify-content:space-between;gap:15px;margin-bottom:4px;}
.ds-prob-track {height:7px;background:#e8eef4;border-radius:2px;overflow:hidden;}
.ds-prob-track span {display:block;height:100%;background:#087e83;}
.ds-perf-row {display:grid;grid-template-columns:195px minmax(80px,1fr) 75px;gap:12px;
  align-items:center;margin:15px 0;font-size:13px;}
.ds-perf-track {background:#edf2f6;height:22px;}
.ds-perf-track span {display:block;background:#4467a1;height:100%;}
.ds-perf-number {text-align:right;font-variant-numeric:tabular-nums;font-weight:600;}
.ds-footer {color:#64748b;font-size:12px;line-height:1.65;padding:20px 0;}
.ds-dot-chart {width:100%;min-width:620px;height:auto;display:block;margin:6px 0 4px;}
@media(max-width:720px){
 .ds-section{padding:18px 15px;}.ds-case-grid{grid-template-columns:1fr;}
 .ds-perf-row{grid-template-columns:125px minmax(50px,1fr) 58px;font-size:12px;gap:8px;}
 .ds-ci-wrap{display:none;}.ds-ci-mobile{display:block;}
 .ds-scroll-hint{display:block;font-size:12px;color:#64748b;margin:8px 0 2px;line-height:1.5;}
 .ds-table{font-size:12px;}.ds-table td,.ds-table th{padding:9px 7px;}
 .ds-hero h1{letter-spacing:-.7px;}.ds-hero{padding-top:12px;}
}
"""

CONDITION_NAMES = {
    "local_only": "StartLux alone",
    "resident_idle": "Both resident; StartLux only",
    "selected_cascade": "Frozen gain cascade",
    "qwen_only": "Qwen alone",
}

FALLBACK_NAMES = {"qwen": "Qwen 4B", "qwen4b": "Qwen 4B", "qwen9b": "Qwen 9B"}


def policy_label(row: dict) -> str:
    """Name both the routing family and the saved fallback configuration."""
    return f'{row["label"]} · {FALLBACK_NAMES[row["fallback"]]}'


def _text(value) -> str:
    return escape(str(value), quote=True)


def _number(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Cannot display a non-finite evidence value")
    return number


def _percent(value, digits=2) -> str:
    return "Not available" if value is None else f"{100 * _number(value):.{digits}f}%"


def _signed(value, digits=3) -> str:
    return f"{_number(value):+.{digits}f}"


def _table(headers: list[str], rows: list[list[str]], *, classes: list[str] | None = None) -> str:
    """Cells are already escaped/formatted by callers; never pass raw evidence."""
    head = "".join(f'<th scope="col">{_text(h)}</th>' for h in headers)
    body = "".join(
        f'<tr class="{classes[i] if classes else ""}">'
        + "".join(f'<td{(" class=" + chr(34) + "ds-number" + chr(34)) if j else ""}>{v}</td>' for j, v in enumerate(row))
        + "</tr>"
        for i, row in enumerate(rows)
    )
    return f'<p class="ds-scroll-hint">Tables scroll horizontally.</p><div class="ds-table-wrap"><table class="ds-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_primary_comparison(experiments: list[dict]) -> str:
    """Render paired confidence intervals on a common pp axis, always including zero."""
    if not experiments:
        raise ValueError("No primary comparisons available")
    limit = max(0.5, max(abs(_number(exp["primary"][bound])) for exp in experiments for bound in ("lower_pp", "upper_pp", "estimate_pp")))
    limit = math.ceil(limit * 2) / 2
    x = lambda value: 505 + _number(value) / limit * 265
    height = 68 + len(experiments) * 75
    svg = [f'<svg class="ds-ci" viewBox="0 0 850 {height}" role="img" aria-label="Accuracy differences in percentage points with paired 95 percent confidence intervals; zero means no difference">',
           f'<line class="ds-zero" x1="505" x2="505" y1="12" y2="{height-48}"/>']
    rows, mobile = [], []
    for i, exp in enumerate(experiments):
        primary = exp["primary"]
        y = 32 + i * 75
        color = "#4467a1" if i % 2 == 0 else "#087e83"
        lo, hi, point = (x(primary[k]) for k in ("lower_pp", "upper_pp", "estimate_pp"))
        svg.extend([
            f'<text x="12" y="{y+4}" fill="#172f46" font-size="16" font-weight="600">{_text(exp["id"].upper())}</text>',
            f'<text x="12" y="{y+23}" fill="#52667c" font-size="12">{_signed(primary["estimate_pp"])} pp</text>',
            f'<line x1="{lo:.2f}" x2="{hi:.2f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="4"/>',
            f'<line x1="{lo:.2f}" x2="{lo:.2f}" y1="{y-7}" y2="{y+7}" stroke="{color}" stroke-width="2"/>',
            f'<line x1="{hi:.2f}" x2="{hi:.2f}" y1="{y-7}" y2="{y+7}" stroke="{color}" stroke-width="2"/>',
            f'<circle cx="{point:.2f}" cy="{y}" r="6" fill="{color}"/>',
        ])
        rows.append([_text(exp["id"].upper()), _text(primary["label"]), _signed(primary["estimate_pp"]) + " pp",
                     f'[{_signed(primary["lower_pp"])}, {_signed(primary["upper_pp"])}] pp'])
        # Use the identical common scale in a compact, fully visible mobile plot.
        mx = lambda value: 150 + _number(value) / limit * 128
        mlo, mhi, mpoint = (mx(primary[k]) for k in ("lower_pp", "upper_pp", "estimate_pp"))
        mobile.append(
            '<div class="ds-ci-mobile-item">'
            + f'<div class="ds-ci-mobile-label"><strong>{_text(exp["id"].upper())}</strong>{_signed(primary["estimate_pp"])} pp</div>'
            + f'<div class="ds-ci-mobile-interval">95% CI [{_signed(primary["lower_pp"])}, {_signed(primary["upper_pp"])}] pp</div>'
            + f'<svg viewBox="0 0 300 70" role="img" aria-label="{_text(exp["id"].upper())} primary accuracy difference with paired 95 percent confidence interval; zero means no difference">'
            + '<line class="ds-zero" x1="150" x2="150" y1="8" y2="43"/>'
            + f'<line x1="{mlo:.2f}" x2="{mhi:.2f}" y1="26" y2="26" stroke="{color}" stroke-width="4"/>'
            + f'<line x1="{mlo:.2f}" x2="{mlo:.2f}" y1="19" y2="33" stroke="{color}" stroke-width="2"/>'
            + f'<line x1="{mhi:.2f}" x2="{mhi:.2f}" y1="19" y2="33" stroke="{color}" stroke-width="2"/>'
            + f'<circle cx="{mpoint:.2f}" cy="26" r="5" fill="{color}"/>'
            + ''.join(f'<text x="{mx(value):.2f}" y="60" text-anchor="middle" fill="#52667c" font-size="13">{value:+g} pp</text>' for value in (-limit, 0, limit))
            + '</svg></div>')
    for value in (-limit, 0, limit):
        svg.append(f'<text x="{x(value):.2f}" y="{height-30}" text-anchor="middle" fill="#52667c" font-size="13">{value:+g} pp</text>')
    svg.append(f'<text x="505" y="{height-8}" text-anchor="middle" fill="#52667c" font-size="12">Candidate accuracy minus frozen confidence policy</text></svg>')
    crosses = all(_number(e["primary"]["lower_pp"]) <= 0 <= _number(e["primary"]["upper_pp"]) for e in experiments)
    interpretation = "Both intervals include zero; neither primary comparison establishes an improvement." if len(experiments) == 2 and crosses else "An interval containing zero does not establish improvement or equivalence."
    return '<section class="ds-section"><h2>Does learned routing improve accuracy?</h2>' + f'<p>{interpretation}</p><div class="ds-ci-wrap">' + "".join(svg) + '</div><div class="ds-ci-mobile">' + ''.join(mobile) + '</div>' + _table(["Study", "Prespecified comparison", "Difference", "95% paired interval"], rows) + '<p class="ds-caption">These are separate tasks and holdouts, not pooled estimates. Positive values favor the candidate. Intervals use the saved paired cluster bootstrap.</p></section>'


def render_overview(data: dict, v1: dict, followups: dict | None = None) -> str:
    experiments = [data["experiments"][key] for key in ("v2", "v3")]
    v1_count = sum(int(v) for v in v1["metrics"]["prediction_counts"].values())
    rows = [["V1", "CLINC and SQuAD answerability", f"{v1_count:,}", "Frozen model quality, calibration and measured service performance"]]
    rows += [[_text(e["id"].upper()), _text(e["title"]), f'{int(e["formal_predictions"]):,}', f'{int(e["n"]):,} held-out questions; saved policies and evaluation'] for e in experiments]
    total = v1_count + sum(int(e["formal_predictions"]) for e in experiments)
    intro = '<section class="ds-section"><h2>Three completed experiments</h2>' + _table(["Study", "Question", "Formal predictions", "Evidence"], rows)
    intro += f'<p class="ds-caption">{total:,} formal model predictions in total. Smoke checks and performance requests are counted separately. This viewer reads recorded evidence and makes no model calls.</p></section>'
    if followups is not None:
        from .followup_view import render_followup_overview
        intro += render_followup_overview(followups)
    return render_primary_comparison(experiments) + intro


def render_experiment(exp: dict) -> str:
    direct = [[_text(r["label"]), f'{int(r["correct"]):,} / {int(r["n"]):,}', _percent(r["accuracy"])] for r in exp["direct"]]
    policy_rows, classes = [], []
    for row in exp["policies"]:
        role = "Primary candidate" if row["is_primary_candidate"] else "Primary reference" if row["is_primary_reference"] else "Secondary description"
        policy_rows.append([
            _text(policy_label(row)) + f'<span class="ds-policy-role">{role}</span>',
            _percent(row["accuracy"]),
            f'{int(row["deferrals"]):,} ({_percent(row["deferral_rate"])})',
            str(int(row["rescue"])), str(int(row["harm"])),
        ])
        classes.append("ds-primary" if row["is_primary_candidate"] else "")
    primary = exp["primary"]
    candidate = next(r for r in exp["policies"] if r["id"] == primary["candidate_id"])
    reference = next(r for r in exp["policies"] if r["id"] == primary["reference_id"])
    notes = '<ul>' + ''.join(f'<li>{_text(n)}</li>' for n in exp.get("notes", [])) + '</ul>'
    zero = ''
    if candidate["deferrals"] == 0:
        zero = '<div class="ds-callout"><strong>The primary candidate chose no handoffs.</strong><br>Its saved decisions equal the local model. Fallback predictions were still collected independently during the experiment.</div>'
    budget = '<section class="ds-section"><h2>Validation budget and actual holdout routing</h2>'
    budget += '<p>The frozen policy makes a per-question decision. Its validation budget is not a guaranteed holdout cap.</p>'
    budget += _table(["Frozen policy", "Validation handoffs", "Holdout handoffs"], [
        [_text(policy_label(row)), _percent(row["validation_deferral_rate"], 3), _percent(row["deferral_rate"], 3)]
        for row in (candidate, reference)]) + '</section>'
    return (f'<section class="ds-section"><h2>{_text(exp["title"])}</h2><p>{int(exp["n"]):,} held-out questions. '
            'Policy accuracy below uses the same saved holdout; rescue and harm count changes relative to the local prediction.</p>'
            + zero + '<h3>Direct models</h3>' + _table(["Model", "Correct", "Accuracy"], direct)
            + '<h3>Frozen policies</h3>' + _table(["Policy", "Accuracy", "Handoffs", "Rescued", "Harmed"], policy_rows, classes=classes)
            + '<p class="ds-caption">The gain/confidence/class-conditional comparison uses the main 10% validation target. A saved original V1 gate, where shown, is a separate historical reference. Secondary rows do not replace the prespecified primary comparison.</p>'
            + notes + '</section>' + budget)


def render_performance(exp: dict, offered_rate: float | None = None, followup_hint: bool = False) -> str:
    perf = exp["performance"]
    title = '<section class="ds-section"><h2>Measured service performance</h2>'
    if not perf["measured"]:
        hint = ('<p>A separate follow-up on 2026-11-09 measured StartLux-9B and a StartLux-4B → Qwen3.5-9B cascade under load on the same ANLI test; '
                'see the Follow-ups tab. It does not change this study’s frozen evidence.</p>') if followup_hint else ''
        return title + '<div class="ds-callout ds-uncertain"><strong>No co-resident performance measurement in this study.</strong><br>' + _text(perf["note"]) + '</div><p>Sequential collection latency is not a measured cascade goodput.</p>' + hint + '</section>'
    rows = perf["rows"]
    if offered_rate is None:
        offered_rate = min(_number(r["offered_rate"]) for r in rows)
    selected = [r for r in rows if _number(r["offered_rate"]) == _number(offered_rate)]
    if not selected:
        raise ValueError("No saved performance measurement at this offered rate")
    maximum = max(1.0, *(_number(r["goodput_mean"]) for r in selected), _number(offered_rate))
    bars = []
    table = []
    for row in selected:
        label = CONDITION_NAMES.get(row["condition"], row["condition"].replace("_", " "))
        val = _number(row["goodput_mean"])
        bars.append(f'<div class="ds-perf-row"><span>{_text(label)}</span><div class="ds-perf-track" aria-hidden="true"><span style="width:{max(0,min(100,val/maximum*100)):.3f}%"></span></div><span class="ds-perf-number">{val:.3f}/s</span></div>')
        table.append([_text(label), f'{val:.3f}', f'{_number(row["goodput_min"]):.3f}–{_number(row["goodput_max"]):.3f}', str(int(row["repetitions"]))])
    return title + f'<p>Offered load: {_number(offered_rate):g} requests/s. Goodput counts requests completed within the saved 500 ms SLO, measured from scheduled arrival.</p>' + ''.join(bars) + _table(["Deployment", "Mean goodput /s", "Repeat range /s", "Repeats"], table) + f'<p class="ds-caption">{_text(perf["note"])} Ranges describe repeats; they are not confidence intervals. The performance input pool differs from the quality holdout.</p></section>'


def replay_case(exp: dict, sample_id: str, policy_id: str) -> dict:
    """Select recorded case and route; never fit or reevaluate a threshold."""
    case = next((c for c in exp["cases"] if c["sample_id"] == sample_id), None)
    policy = next((p for p in exp["policies"] if p["id"] == policy_id), None)
    if case is None or policy is None:
        raise ValueError("Select a case and policy from the saved experiment")
    if policy_id not in case["routes"]:
        raise ValueError("No saved route for this case and policy")
    fallback = case["fallbacks"].get(policy["fallback"])
    if fallback is None:
        raise ValueError("The selected fallback prediction is missing")
    return {"experiment": exp["id"], "case": case, "policy": policy, "route": case["routes"][policy_id], "fallback": fallback}


def render_case(result: dict) -> str:
    case, policy, route, fallback = (result[k] for k in ("case", "policy", "route", "fallback"))
    local = case["local"]
    probabilities = ''.join(
        f'<div class="ds-prob"><div class="ds-prob-label"><span>{_text(label)}</span><span>{_percent(value)}</span></div><div class="ds-prob-track"><span style="width:{max(0,min(100,_number(value)*100)):.3f}%"></span></div></div>'
        for label, value in (local.get("probabilities") or {}).items())
    path = "Use the fallback prediction" if route["deferred"] else "Keep the local prediction"
    correct = "Correct" if route["correct"] else "Incorrect"
    result_class = "ds-correct" if route["correct"] else "ds-wrong"
    return ('<section class="ds-section"><div class="ds-case-grid"><div><h2>Saved input</h2>'
            + f'<div class="ds-input">{_text(case["state"])}</div><p class="ds-case-meta">Sample {_text(case["sample_id"])}<br>Reference label: {_text(case["gold"])}</p></div>'
            + '<div><h2>Frozen decision path</h2>'
            + f'<p>{_text(local["label"])} predicts <strong>{_text(local["prediction"])}</strong>.<br>Saved confidence: {_percent(local.get("confidence"))}</p>'
            + probabilities + f'<div class="ds-route {result_class}"><strong>{path}</strong>{_text(policy_label(policy))}<br>Final label: <b>{_text(route["final_prediction"])}</b><br>{correct}; {_text(route["outcome"]).replace("_", " ")}</div>'
            + f'<p>{_text(fallback["label"])} independently predicts <strong>{_text(fallback["prediction"])}</strong>. '
            + ('This saved policy used that output.' if route["deferred"] else 'This saved policy did not use that output.')
            + '</p><p class="ds-caption">Qwen probabilities are unavailable. These recorded outputs do not represent a fresh model request.</p></div></div></section>')


def show_case(exp: dict, sample_id: str, policy_id: str) -> tuple[str, dict]:
    result = replay_case(exp, sample_id, policy_id)
    return render_case(result), result


def _experiment_tab(gr, exp: dict, followup_hint: bool = False):
    gr.HTML(render_experiment(exp))
    if exp["performance"]["measured"]:
        rates = sorted({_number(row["offered_rate"]) for row in exp["performance"]["rows"]})
        rate = gr.Radio(choices=rates, value=rates[-1], label="Measured offered load (requests/s)")
        perf = gr.HTML(render_performance(exp, rates[-1]))
        rate.change(lambda value: render_performance(exp, float(value)), rate, perf, api_name=f'{exp["id"]}_performance')
    else:
        gr.HTML(render_performance(exp, followup_hint=followup_hint))
    gr.Markdown("### Follow a saved handoff")
    gr.Markdown(exp["case_note"])
    initial_case = exp["cases"][0]["sample_id"]
    initial_policy = exp["primary"]["candidate_id"]
    with gr.Row():
        cases = gr.Dropdown(choices=[(f'{c["selection_reason"]} | {c["sample_id"]}', c["sample_id"]) for c in exp["cases"]], value=initial_case, label="Illustrative case", scale=3)
        policy = gr.Dropdown(choices=[(policy_label(p), p["id"]) for p in exp["policies"]], value=initial_policy, label="Saved policy", scale=2)
    first = show_case(exp, initial_case, initial_policy)
    panels = gr.HTML(first[0])
    with gr.Accordion("Saved case and evidence identities", open=False):
        trace = gr.JSON(first[1], label="Normalized recorded decision")
        gr.JSON(exp["provenance"], label="Source hashes")
    callback = lambda sample, chosen: show_case(exp, sample, chosen)
    cases.change(callback, [cases, policy], [panels, trace], api_name=f'{exp["id"]}_case')
    policy.change(callback, [cases, policy], [panels, trace], api_name=f'{exp["id"]}_policy')


def create_app(bundle_dir: Path | str, v1_bundle_dir: Path | str, followup_bundle_dir: Path | str | None = None):
    """Create the viewer without serving, fitting, opening a browser or calling models."""
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the Lab dependency: pip install 'gradio==6.31.0'") from exc
    from .evidence import load_bundle
    data = load_bundle(bundle_dir)
    v1 = load_v1_bundle(v1_bundle_dir)
    followups = None
    if followup_bundle_dir is not None:
        from .followups import load_bundle as load_followups
        from .followup_view import followup_tab
        followups = load_followups(followup_bundle_dir)
    original = create_v1_app(v1_bundle_dir)
    scope = "V1–V3 and follow-up evidence" if followups is not None else "V1–V3 evidence"
    with gr.Blocks(title=f"Decision Lab | {scope}", analytics_enabled=False, delete_cache=(3600, 3600)) as app:
        gr.HTML('<header class="ds-hero"><div class="ds-brand">Decision Lab</div><h1>When is a handoff worth it?</h1><p>Compare the quality gained, the errors introduced, and the service cost across three completed experiments'
                + (' and their follow-ups' if followups is not None else '') + f'.</p><div class="ds-status">Recorded {scope}. CPU only. No model calls.</div></header>')
        with gr.Tabs():
            with gr.Tab("Overview", id="overview"):
                gr.HTML(render_overview(data, v1, followups))
            with gr.Tab("V2 · SQuAD", id="v2"):
                _experiment_tab(gr, data["experiments"]["v2"])
            with gr.Tab("V3 · ANLI", id="v3"):
                _experiment_tab(gr, data["experiments"]["v3"], followup_hint=followups is not None)
            if followups is not None:
                with gr.Tab("Follow-ups", id="followups"):
                    followup_tab(gr, followups)
            with gr.Tab("V1 replay", id="v1"):
                gr.Markdown("The original V1 viewer below retains its saved cases, exploratory threshold controls and measured performance. Its descriptions apply to V1 only.")
                original.render()
        gr.HTML('<footer class="ds-footer">All displayed policies and primary intervals come from saved evaluations, and serving values from accepted measurements at the tested loads only. Illustrative cases are selected by known outcomes and do not estimate their frequency. Exploring evidence does not change a frozen study.</footer>')
    app._decision_lab_original = original  # Keep original callback export files alive.
    app.queue(max_size=16, default_concurrency_limit=1)
    return app


def launch(bundle_dir: Path | str, v1_bundle_dir: Path | str, host: str = "127.0.0.1", port: int = 7860,
           followup_bundle_dir: Path | str | None = None):
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The evidence viewer binds to localhost only")
    app = create_app(bundle_dir, v1_bundle_dir, followup_bundle_dir)
    import gradio as gr
    theme = gr.themes.Base(primary_hue="teal", neutral_hue="slate", radius_size="sm",
                           font=("Aptos", "Trebuchet MS", "Arial", "sans-serif"))
    return app.launch(server_name=host, server_port=port, share=False, inbrowser=False,
                      css=CSS, theme=theme, footer_links=[], show_error=False,
                      enable_monitoring=False, mcp_server=False, pwa=False)
