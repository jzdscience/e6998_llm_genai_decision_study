"""Follow-up page of the CPU Decision Lab: measured serving and baseline evidence only.

Every rendered number comes from the validated portable follow-up bundle. Selectors choose among
measured rates and SLOs; an unmeasured value raises instead of being interpolated. No model calls.
"""
from __future__ import annotations

from .app import _number, _percent, _table, _text

# Validated categorical slots (blue, green, yellow pass the all-pairs check on white); every
# deployment also has its own marker shape and a text label, so colour never carries identity alone.
STYLE = {"startlux4b": ("#2a78d6", "circle"), "startlux9b": ("#008300", "square"), "cascade": ("#eda100", "diamond")}
CLINC_COLOR = {"minilm400": "#1baf7a", "minilm1600": "#1baf7a", "startlux4b": "#2a78d6", "qwen4b": "#eb6834"}
INK, INK2, MUTED, GRID = "#172f46", "#52667c", "#7a8da3", "#e0e7ef"


def _choice(value, allowed, what: str) -> int:
    """Accept only a measured value; never snap or interpolate to a neighbour."""
    if isinstance(value, bool):
        raise ValueError(f"No measured {what} for this selection")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"No measured {what} for this selection") from exc
    if not number.is_integer() or int(number) not in allowed:
        raise ValueError(f"No measured {what} for this selection")
    return int(number)


def _deployments(data: dict) -> dict:
    return {d["id"]: d for d in data["serving9b"]["deployments"]}


def _row(data: dict, condition: str, rate: int, slo: int) -> dict:
    return next(r for r in data["serving9b"]["summary"]
                if r["condition"] == condition and r["rate"] == rate and r["slo_ms"] == slo)


def _rate_value(value: float) -> str:
    return f"{_number(value):.3f}"


def _marker(shape: str, x: float, y: float, size: float, color: str, filled: bool) -> str:
    fill, stroke = (color, color) if filled else ("#ffffff", color)
    if shape == "square":
        return f'<rect x="{x - size:.2f}" y="{y - size:.2f}" width="{2 * size:.2f}" height="{2 * size:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    if shape == "diamond":
        d = size * 1.25
        return f'<polygon points="{x:.2f},{y - d:.2f} {x + d:.2f},{y:.2f} {x:.2f},{y + d:.2f} {x - d:.2f},{y:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


def _serving_chart(data: dict, rate: int, slo: int) -> str:
    """One row per deployment: the three repeat values, their min-max range and the median marker."""
    deployments = _deployments(data)
    rows = [_row(data, d, rate, slo) for d in deployments]
    top = max(float(rate), *(_number(v) for r in rows for v in r["correct_goodput"]["repeats"]))
    x0, x1 = 230.0, 690.0
    x = lambda value: x0 + _number(value) / top * (x1 - x0)
    height = 58 + 56 * len(rows)
    svg = [f'<svg class="ds-dot-chart" viewBox="0 0 860 {height}" role="img" aria-label="Correct answers within {slo} ms per second at {rate} requests per second; each row shows the three repeats, their range and the median">',
           f'<line x1="{x(rate):.2f}" x2="{x(rate):.2f}" y1="14" y2="{height - 40}" stroke="{GRID}" stroke-width="2"/>',
           f'<text x="{x(rate):.2f}" y="11" text-anchor="middle" fill="{MUTED}" font-size="12">offered {rate}/s</text>']
    for i, row in enumerate(rows):
        y = 40 + 56 * i
        color, shape = STYLE[row["condition"]]
        g = row["correct_goodput"]
        svg.append(f'<text x="12" y="{y + 5}" fill="{INK}" font-size="14" font-weight="600">{_text(deployments[row["condition"]]["short"])}</text>')
        svg.append(f'<line x1="{x(g["min"]):.2f}" x2="{x(g["max"]):.2f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        for value in g["repeats"]:
            svg.append(f'<circle cx="{x(value):.2f}" cy="{y}" r="3.5" fill="{color}" stroke="#ffffff" stroke-width="1"/>')
        svg.append(_marker(shape, x(g["median"]), y, 7, color, row["passes_rule"]))
        svg.append(f'<text x="{x1 + 26}" y="{y + 5}" fill="{INK}" font-size="13">{_rate_value(g["median"])}/s</text>')
    axis_y = height - 26
    svg.append(f'<line x1="{x0}" x2="{x1}" y1="{axis_y - 12}" y2="{axis_y - 12}" stroke="{GRID}" stroke-width="1"/>')
    for tick in (0, top / 2, top):
        svg.append(f'<text x="{x(tick):.2f}" y="{axis_y + 4}" text-anchor="middle" fill="{INK2}" font-size="12">{tick:g}</text>')
    svg.append(f'<text x="{(x0 + x1) / 2:.2f}" y="{height - 4}" text-anchor="middle" fill="{INK2}" font-size="12">Correct answers within {slo:,} ms per second</text></svg>')
    return "".join(svg)


def render_serving(data: dict, rate, slo) -> str:
    """The selected measured cell group: three deployments at one rate and one SLO."""
    s = data["serving9b"]
    rate, slo = _choice(rate, s["rates"], "offered rate"), _choice(slo, s["slos_ms"], "SLO")
    deployments = _deployments(data)
    rows = [_row(data, d, rate, slo) for d in deployments]
    best = max(rows, key=lambda r: _number(r["correct_goodput"]["median"]))
    name = lambda r: deployments[r["condition"]]["label"]
    sentence = (f'At {rate} requests/s with a {slo:,} ms SLO, {_text(name(best))} returned the most correct answers on time, '
                f'a median of {_rate_value(best["correct_goodput"]["median"])} per second over three 120-second repeats.')
    passing = [r for r in rows if r["passes_rule"]]
    if not best["passes_rule"] and passing:
        sentence += (f' It missed the 99% on-time rule in at least one repeat (worst {_percent(best["timely_fraction"]["min"])}), '
                     f'while {_text(" and ".join(name(r) for r in passing))} met it in every repeat.')
    elif not passing:
        sentence += " No deployment met the 99% on-time rule at this load."
    table = []
    for r in rows:
        d, g, t = deployments[r["condition"]], r["correct_goodput"], r["timely_fraction"]
        table.append([
            _text(d["label"]), _text(d["gpu_share"]), _rate_value(g["median"]),
            " / ".join(_rate_value(v) for v in g["repeats"]),
            " / ".join(_percent(v) for v in t["repeats"]),
            "Yes" if r["passes_rule"] else "No",
            f'{_number(r["successful_per_second_median"]):.2f}',
            _percent(r["client_rejected"] / r["offered"]),
            f'{_number(r["latency_p95_ms_median"]):,.1f} ms',
        ])
    notes = []
    anomaly = s["anomaly"]
    if rate == anomaly["rate"]:
        lo, hi = anomaly["arrival_window_seconds"]
        cell = next(c for c in s["cells"] if c["cell_id"] == anomaly["cell_id"])
        late = {slo_key: cell["successful"] - cell["slo"][slo_key]["met"] for slo_key in ("500", "1000")}
        notes.append(f'StartLux-4B repeat r0 at {rate}/s is kept as measured. In that repeat {late["500"]} responses arrived after '
                     f'500 ms ({late["1000"]} after 1,000 ms), for requests scheduled between {_number(lo):.1f} s and {_number(hi):.1f} s; '
                     f'their median client-queue wait was {_number(anomaly["client_queue_median_ms"]):,.0f} ms against a median model '
                     f'call of {_number(anomaly["model_call_median_ms"]):.0f} ms. The cause was not isolated.')
    cascade = next(r for r in rows if r["condition"] == "cascade")
    if cascade["deferral_rate"]:
        dr = cascade["deferral_rate"]
        notes.append(f'The cascade handed {_percent(dr["min"], 1)}–{_percent(dr["max"], 1)} of its answered requests to Qwen3.5-9B in these repeats.')
    note_html = "".join(f'<p class="ds-caption">{n}</p>' for n in notes)
    return ('<section class="ds-section"><h2>Measured cells</h2>' + f'<p>{sentence}</p>'
            + '<div class="ds-table-wrap">' + _serving_chart(data, rate, slo) + '</div>'
            + '<p class="ds-caption">Small dots are the three repeats and the line spans their range; the large marker is the median, filled when every repeat answered at least 99% of offered requests within the SLO.</p>'
            + _table(["Deployment", "GPU share", "Correct on time /s, median", "Repeats r0 / r1 / r2", "On time, r0 / r1 / r2",
                      "All repeats ≥ 99%", "Completed /s incl. late", "Client rejected", "p95 latency"], table)
            + note_html + '</section>')


def render_serving_overview(data: dict, slo, selected_rate=None) -> str:
    """All measured rates at one SLO; a tick marks cells where every repeat met the 99% rule."""
    s = data["serving9b"]
    slo = _choice(slo, s["slos_ms"], "SLO")
    selected = None if selected_rate is None else _choice(selected_rate, s["rates"], "offered rate")
    deployments = _deployments(data)
    rows = []
    for d in deployments:
        cells = [_text(deployments[d]["label"]), _text(deployments[d]["gpu_share"])]
        for rate in s["rates"]:
            r = _row(data, d, rate, slo)
            cells.append(_rate_value(r["correct_goodput"]["median"]) + (" ✓" if r["passes_rule"] else ""))
        rows.append(cells)
    headers = ["Deployment", "GPU share"] + [f"{r}/s" + (" (selected)" if r == selected else "") for r in s["rates"]]
    passing = s["passing_rates"][str(slo)]
    pass_text = "; ".join(f'{_text(deployments[d]["short"])}: {", ".join(str(r) for r in passing[d]) or "none"}' for d in deployments)
    return (f'<section class="ds-section"><h2>Every measured rate at {slo:,} ms</h2>'
            + _table(headers, rows)
            + f'<p>Rates where every repeat met the 99% rule: {pass_text}.</p>'
            + '<p class="ds-caption">Median correct answers on time per second over three repeats; ✓ marks a rate where every repeat '
              'answered at least 99% of offered requests within the SLO. Only these four rates were measured, and a passing rate says '
              'nothing about the rates around it. GPU memory shares differ by deployment.</p></section>')


def render_serving_context(data: dict) -> str:
    """Quality, memory and preflight context that does not depend on the selectors."""
    s = data["serving9b"]
    deployments = _deployments(data)
    quality = [[_text(deployments[q["condition"]]["label"]), f'{int(q["successful_responses"]):,}',
                _percent(q["pooled_live_accuracy"], 3), _percent(q["pooled_matched_historical_accuracy"], 3),
                f'{_percent(q["live_accuracy_cell_range"][0])}–{_percent(q["live_accuracy_cell_range"][1])}',
                _percent(q["historical_full_pool_accuracy"]), str(int(q["inputs_with_more_than_one_live_label"]))]
               for q in s["quality"]]
    role = {"startlux4b": "StartLux-4B", "startlux9b": "StartLux-9B", "qwen9b": "Qwen3.5-9B"}
    peaks = {}
    for cell in s["cells"]:
        peaks.setdefault(cell["condition"], []).append(_number(cell["sampled_gpu_peak_gib"]))
    memory = [[_text(deployments[r["condition"]]["label"]), _text(role[r["role"]]), _text(r["gpu_share"]),
               " / ".join(f"{_number(v):.2f}" for v in r["formal_kv_gib"]) + " GiB",
               f'{_number(r["preflight_kv_gib"]):.2f} GiB',
               f'{min(peaks[r["condition"]]):.2f}–{max(peaks[r["condition"]]):.2f} GiB'] for r in s["resources"]]
    pre = s["preflight"]
    t = s["totals"]
    hist = s["historical_deferral"]
    return ('<section class="ds-section"><h2>Quality and memory behind these numbers</h2>'
            '<p>Live accuracy is measured on the requests that were answered, so it shifts with load. The matched column scores the saved '
            'offline predictions for exactly those requests, counting repeated inputs as often as they were answered.</p>'
            + _table(["Deployment", "Answered requests", "Live accuracy", "Saved predictions, same requests", "Live accuracy per cell",
                      "Saved, all 3,200 items", "Inputs with more than one live label"], quality)
            + f'<p class="ds-caption">{int(t["offered"]):,} offered requests in 36 cells: {int(t["successful"]):,} answered and '
              f'{int(t["client_rejected"]):,} rejected by the client limit; no model errors or timeouts. The frozen router defers '
              f'{int(hist["deferred"])} of {int(hist["n"]):,} saved test items ({_percent(hist["deferred"] / hist["n"])}). '
              'Close agreement between live and saved accuracy is not an equivalence test.</p>'
            + _table(["Deployment", "Model", "GPU share", "KV cache, formal cells", "KV cache, earlier preflight", "Sampled GPU memory peak"], memory)
            + '<p class="ds-caption">KV capacity comes from each formal engine start (12 per role). The preflight ran about eight hours '
              'earlier with different KV for the 9B models. Memory peaks are whole-card samples per cell, so short spikes between samples '
              'are missed; there is no dollar cost.</p>'
            + f'<p class="ds-caption">The preflight forced every cascade request through both models: {int(pre["timely_500"]["cascade"])} of '
              f'{int(pre["concurrent_offers"])} answered within 500 ms. Its pairwise HTTP-interval overlap sums to '
              f'{_number(pre["pairwise_http_stage_overlap_seconds"]):.1f} s, while both models had requests in flight for '
              f'{_number(pre["simultaneous_http_union_seconds"]):.1f} s of wall-clock time; neither figure measures GPU kernel overlap.</p>'
            + '<ul>' + "".join(f'<li>{_text(n)}</li>' for n in s["notes"]) + '</ul></section>')


def render_clinc_service(data: dict, rate) -> str:
    service = data["clinc_service"]
    rate = _choice(rate, service["rates"], "offered rate")
    rows = [r for r in service["rows"] if r["rate"] == rate]
    top = max(float(rate), *(_number(r["correct_goodput"]["max"]) for r in rows))
    bars, table = [], []
    for r in rows:
        label = service["labels"][r["condition"]]
        g = r["correct_goodput"]
        width = max(0.0, min(100.0, _number(g["median"]) / top * 100))
        bars.append(f'<div class="ds-perf-row"><span>{_text(label)}</span><div class="ds-perf-track" aria-hidden="true">'
                    f'<span style="width:{width:.3f}%;background:{CLINC_COLOR[r["condition"]]}"></span></div>'
                    f'<span class="ds-perf-number">{_number(g["median"]):.2f}/s</span></div>')
        resource = (f'{_number(r["cpu_core_equivalents_median"]):.2f} CPU cores' if r["sampled_gpu_peak_gib_max"] is None
                    else f'{_number(r["sampled_gpu_peak_gib_max"]):.1f} GiB GPU peak')
        table.append([_text(label), f'{_number(g["median"]):.2f}', f'{_number(g["min"]):.2f}–{_number(g["max"]):.2f}',
                      f'{_number(r["successful_per_second_median"]):.2f}', _percent(r["rejected"] / r["offered"]),
                      f'{_number(r["latency_p95_ms_median"]):,.1f} ms', _percent(r["accuracy_successful_median"]), resource])
    return (f'<section class="ds-section"><h2>{_text(service["title"])}</h2>'
            f'<p>Offered load {rate} requests/s, 500 ms SLO. Bars show the median correct answers on time per second over five repeats.</p>'
            + "".join(bars)
            + _table(["Service", "Correct on time /s", "Repeat range", "Completed /s incl. late", "Client rejected", "p95 latency",
                      "Accuracy of answers", "Resource"], table)
            + f'<p class="ds-caption">{_text(service["note"])} CPU cores are observed core-equivalents for the service process; GPU '
              'memory is the highest per-cell sample.</p></section>')


def render_baselines(data: dict) -> str:
    b, size = data["cpu_baselines"], data["size_control"]
    rows = [[_text(r["label"]), f'{int(r["main_correct"])} / {int(r["main_count"])}', _percent(r["main_correct"] / r["main_count"]),
             _percent(r["banking_correct"] / r["banking_count"]), _percent(r["oos_correct"] / r["oos_count"]),
             _percent(r["mixed_accuracy"])] for r in b["rows"]]
    d, direct = size["difference_pp"], size["direct"]
    return (f'<section class="ds-section"><h2>{_text(b["title"])}</h2>'
            + _table(["Model", "Main test correct", "Main accuracy", "Banking intents", "Out-of-scope recall", "Stress workload"], rows)
            + f'<p class="ds-caption">{_text(b["note"])}</p>'
            + '<h3>ANLI size control</h3>'
            + f'<p>On the same {int(size["n"]):,} ANLI test items, StartLux-9B answered {_percent(direct["startlux9b"]["accuracy"])} correctly '
              f'and StartLux-4B {_percent(direct["startlux4b"]["accuracy"])}, a paired difference of {_number(d["estimate"]):+.2f} points '
              f'(95% interval {_number(d["lower"]):.2f} to {_number(d["upper"]):.2f}). The 9B model is right on {int(size["only_9b_correct"])} '
              f'items the 4B model misses and wrong on {int(size["only_4b_correct"])} it gets right.</p>'
            + f'<p class="ds-caption">{_text(size["note"])}</p></section>')


def render_followup_overview(data: dict) -> str:
    """Short pointer from the Overview tab; the follow-ups are retrospective and kept apart from V1-V3."""
    s, service, size = data["serving9b"], data["clinc_service"], data["size_control"]
    minilm = next(r for r in service["rows"] if r["condition"] == "minilm1600" and r["rate"] == 100)
    gpu = next(r for r in service["rows"] if r["condition"] == "startlux4b" and r["rate"] == 100)
    at = lambda c, rate: _rate_value(_row(data, c, rate, 500)["correct_goodput"]["median"])
    rows = [
        ["CLINC CPU baselines", "TF-IDF and MiniLM classifiers trained on 400 or 1,600 labels",
         f'MiniLM-1600 {next(r for r in data["cpu_baselines"]["rows"] if r["model"] == "minilm1600")["main_correct"]} of 600 correct'],
        ["CLINC CPU/GPU serving", "Four direct services, 5–100 requests/s",
         f'At 100/s: MiniLM-1600 {_number(minilm["correct_goodput"]["median"]):.1f}, StartLux-4B {_number(gpu["correct_goodput"]["median"]):.2f} correct on time /s'],
        ["ANLI size control", f'{int(size["n"]):,} test items, StartLux-9B vs 4B', f'{_number(size["difference_pp"]["estimate"]):+.2f} points'],
        ["ANLI 9B serving", "StartLux-4B, StartLux-9B and a 4B → Qwen3.5-9B cascade, 10–80 requests/s",
         f'At 40/s: 9B {at("startlux9b", 40)}, 4B {at("startlux4b", 40)}, cascade {at("cascade", 40)}; at 80/s 4B leads with {at("startlux4b", 80)}'],
    ]
    return ('<section class="ds-section"><h2>Follow-up studies after V3</h2>'
            + _table(["Study", "Scope", "Headline (500 ms SLO where serving)"], [[_text(c) for c in r] for r in rows])
            + '<p class="ds-caption">Each follow-up reuses a test set scored earlier, so it is retrospective and does not change the V1–V3 '
              'primary comparisons. Serving requests repeat the same inputs and are counted apart from formal predictions. Details are in the '
              'Follow-ups tab.</p></section>')


def replay_followup_case(data: dict, case_id: str) -> dict:
    case = next((c for c in data["cases"] if c["id"] == case_id), None)
    if case is None:
        raise ValueError("Select one of the saved follow-up cases")
    ids = {k: case[k] for k in ("sample_id", "round", "cell_id", "offer_id", "condition", "rate", "repeat") if k in case}
    return {"case": case, "source_ids": {**ids, "selection": case["kind"],
                                         "pattern_count": case["pattern_count"], "pattern_base": case["pattern_base"],
                                         "pattern_unit": case["pattern_unit"]}}


def _probability_line(label: str, saved: dict) -> str:
    probabilities = saved.get("probabilities") or {}
    confidence = f' (probability {_percent(probabilities[saved["prediction"]])})' if saved["prediction"] in probabilities else ""
    return f'<p>{_text(label)} predicts <strong>{_text(saved["prediction"])}</strong>{confidence}.</p>'


def render_followup_case(result: dict) -> str:
    case = result["case"]
    gold = case["gold"]
    verdict = lambda label: "correct" if label == gold else "incorrect"
    left = (f'<div><h2>Saved input</h2><div class="ds-input">{_text(case["state"])}</div>'
            f'<p class="ds-case-meta">ANLI {_text(case["round"])} test item<br>Reference label: {_text(gold)}</p></div>')
    if case["kind"] == "late_correct":
        t = case["trace"]
        right = ('<div><h2>One live request</h2>'
                 f'<p>StartLux-9B answered <strong>{_text(t["prediction"])}</strong> ({verdict(t["prediction"])}) '
                 f'{_number(t["arrival_to_completion_ms"]):,.0f} ms after the scheduled arrival: {_number(t["client_queue_ms"]):,.0f} ms waiting '
                 f'in the client queue and {_number(t["model_call_ms"]):,.0f} ms in the model call.</p>'
                 '<div class="ds-route ds-wrong"><strong>Counted as completed, adds nothing to correct goodput</strong>'
                 'It missed both the 500 ms and the 1,000 ms SLO.</div>'
                 f'<p class="ds-caption">Offered load {int(case["rate"])} requests/s, first repeat. In that cell {int(case["pattern_count"]):,} of '
                 f'{int(case["pattern_base"]):,} {_text(case["pattern_unit"])} were correct but later than 1,000 ms; inputs repeat '
                 'across live requests.</p></div>')
    else:
        s = case["saved"]
        route = "hands the input to Qwen3.5-9B" if s["deferred"] else "keeps the StartLux-4B answer"
        final_class = "ds-correct" if s["cascade_final"] == gold else "ds-wrong"
        big_class = "ds-correct" if s["startlux9b"]["prediction"] == gold else "ds-wrong"
        live_rows = [[_text({"startlux4b": "StartLux-4B direct", "startlux9b": "StartLux-9B direct", "cascade": "Cascade"}[c]),
                      str(int(v["successful"])), _text(", ".join(f"{k} × {n}" for k, n in v["labels"].items())), str(int(v["timely_500"]))]
                     for c, v in ((c, case["live"][c]) for c in ("startlux4b", "startlux9b", "cascade") if c in case["live"])]
        right = ('<div><h2>Saved decisions</h2>'
                 + _probability_line("StartLux-4B", s["startlux4b"])
                 + f'<p>The frozen router {route}. Qwen3.5-9B independently predicts <strong>{_text(s["qwen9b"]["prediction"])}</strong>.</p>'
                 + f'<div class="ds-route {final_class}"><strong>Cascade answer: {_text(s["cascade_final"])}</strong>{verdict(s["cascade_final"]).capitalize()}</div>'
                 + f'<div class="ds-route {big_class}"><strong>StartLux-9B answer: {_text(s["startlux9b"]["prediction"])}</strong>'
                 + f'{verdict(s["startlux9b"]["prediction"]).capitalize()}</div>'
                 + f'<p class="ds-caption">{int(case["pattern_count"])} of {int(case["pattern_base"]):,} {_text(case["pattern_unit"])} share this '
                 + f'pattern, within {int(case["context"]["count"])} {_text(case["context"]["label"])}.</p>'
                 + '<h3>The same input during the serving runs</h3>'
                 + _table(["Deployment", "Answered", "Labels returned", "Within 500 ms"], live_rows) + '</div>')
    return (f'<section class="ds-section"><h2>{_text(case["title"])}</h2><div class="ds-case-grid">{left}{right}</div>'
            '<p class="ds-caption">Recorded outputs only; nothing here sends a new request to a model.</p></section>')


def show_followup_case(data: dict, case_id: str) -> tuple[str, dict]:
    result = replay_followup_case(data, case_id)
    return render_followup_case(result), result["source_ids"]


def followup_tab(gr, data: dict):
    """Build the Follow-ups tab body inside an open gr.Tab context."""
    s = data["serving9b"]
    default_rate, default_slo = 40, 500
    gr.HTML('<section class="ds-section"><h2>' + _text(s["title"]) + '</h2><p>Three fixed deployments served the 3,200 ANLI test '
            'inputs on one L40S at four offered loads, three 120-second repeats each. Choose a latency SLO and a measured load to see '
            'how many correct answers each deployment returned on time.</p></section>')
    with gr.Row():
        slo = gr.Radio(choices=[("500 ms (primary)", 500), ("1,000 ms (secondary)", 1000)], value=default_slo,
                       label="Latency SLO, from scheduled arrival")
        rate = gr.Radio(choices=[(f"{r} requests/s", r) for r in s["rates"]], value=default_rate, label="Measured offered load")
    serving = gr.HTML(render_serving(data, default_rate, default_slo))
    overview = gr.HTML(render_serving_overview(data, default_slo, default_rate))
    update = lambda slo_value, rate_value: (render_serving(data, rate_value, slo_value),
                                            render_serving_overview(data, slo_value, rate_value))
    slo.change(update, [slo, rate], [serving, overview], api_name="followup_serving_slo")
    rate.change(update, [slo, rate], [serving, overview], api_name="followup_serving_rate")
    gr.HTML(render_serving_context(data))
    clinc_rates = data["clinc_service"]["rates"]
    clinc_rate = gr.Radio(choices=[(f"{r} requests/s", r) for r in clinc_rates], value=clinc_rates[-1],
                          label="CLINC measured offered load")
    clinc = gr.HTML(render_clinc_service(data, clinc_rates[-1]))
    clinc_rate.change(lambda value: render_clinc_service(data, value), clinc_rate, clinc, api_name="followup_clinc_rate")
    gr.HTML(render_baselines(data))
    gr.Markdown("### Follow a saved ANLI case")
    gr.Markdown(data["case_note"])
    first_id = data["cases"][0]["id"]
    case = gr.Dropdown(choices=[(c["title"], c["id"]) for c in data["cases"]], value=first_id, label="Illustrative case")
    first = show_followup_case(data, first_id)
    panel = gr.HTML(first[0])
    with gr.Accordion("Follow-up case and evidence identities", open=False):
        trace = gr.JSON(first[1], label="Case source IDs")
        gr.JSON(data["provenance"], label="Source hashes")
    case.change(lambda value: show_followup_case(data, value), case, [panel, trace], api_name="followup_case")
