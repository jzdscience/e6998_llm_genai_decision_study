"""Deterministic, source-verified data preparation for the solo protocol.

Only public source data are downloaded. Model inputs and gold labels are written
separately. No text is truncated, and no split is silently resized or replenished
from another split.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request


SEED = "6998-solo-v1"
PROTOCOL = "solo-v1"
SOURCES = {
    "data_full.json": {
        "url": "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
        "sha256": "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
    },
    "domains.json": {
        "url": "https://raw.githubusercontent.com/clinc/oos-eval/master/data/domains.json",
        "sha256": "b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a",
    },
    "train-v2.0.json": {
        "url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
        "sha256": "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002",
    },
    "dev-v2.0.json": {
        "url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "sha256": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    },
}

BANKING_CRITERIA = {
    "freeze_account": "Temporarily freeze or suspend a bank account.",
    "routing": "Find or ask about a bank routing number.",
    "pin_change": "Change or reset a banking card PIN.",
    "bill_due": "Find when a bill payment is due.",
    "pay_bill": "Make or arrange a bill payment.",
    "account_blocked": "Ask about an account that is blocked or inaccessible.",
    "interest_rate": "Ask about a banking interest rate.",
    "min_payment": "Find the minimum required payment.",
    "bill_balance": "Find the outstanding balance of a bill.",
    "transfer": "Move money between accounts or to another recipient.",
    "order_checks": "Order a supply of paper checks.",
    "balance": "Check the available balance of a bank account.",
    "spending_history": "Review historical spending or spending totals.",
    "transactions": "Review or ask about individual account transactions.",
    "report_fraud": "Report suspected fraud or unauthorized account activity.",
    "oos": "The request does not match any of the fifteen listed banking intents.",
}
BANKING_LABELS = sorted(k for k in BANKING_CRITERIA if k != "oos") + ["oos"]
SQUAD_LABELS = ["answerable", "unanswerable"]
SQUAD_CRITERIA = {
    "answerable": "The given passage contains evidence sufficient to answer the question.",
    "unanswerable": "The given passage does not contain evidence sufficient to answer the question.",
}
CLINC_INSTRUCTIONS = (
    "Classify the user's request into exactly one of the listed banking intents. "
    "Select oos when none of the fifteen banking intents applies."
)
SQUAD_INSTRUCTIONS = (
    "Decide whether the question can be answered using only the given passage. "
    "Use answerable only if the passage supports an answer; otherwise use "
    "unanswerable. Do not use outside knowledge and do not generate an answer."
)
OOS_FIXTURES = [
    "What will the weather be like tomorrow?",
    "Give me a recipe for tomato soup.",
    "Find a flight from Boston to Seattle.",
    "Translate good morning into Japanese.",
]


class DataError(ValueError):
    """A source or protocol invariant failed; do not run formal experiments."""


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rank(role: str, identifier: str) -> str:
    return _sha(f"{SEED}|{role}|{identifier}".encode("utf-8"))


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True,
                                allow_nan=False) + "\n").encode("utf-8") for r in rows)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _source_bytes(cache_dir: Path, name: str, download: bool) -> bytes:
    spec = SOURCES[name]
    path = cache_dir / name
    if path.exists():
        content = path.read_bytes()
    else:
        if not download:
            raise DataError(f"Missing cached source {path}; enable downloads or populate the cache.")
        last_error = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "COMS6998-decision-study/1"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    content = response.read()
                break
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(attempt + 1)
        else:
            raise DataError(f"Could not download {name}: {last_error}") from last_error
        if _sha(content) != spec["sha256"]:
            raise DataError(f"Source SHA256 mismatch for {name}: got {_sha(content)}. No unverified file saved.")
        _atomic_write(path, content)
    actual = _sha(content)
    if actual != spec["sha256"]:
        raise DataError(f"Source SHA256 mismatch for cached {name}: got {actual}; expected {spec['sha256']}.")
    return content


def _sample(rows: list[dict], count: int, role: str, id_key: str = "sample_id") -> list[dict]:
    if len(rows) < count:
        raise DataError(f"Insufficient {role} pool: need {count}, found {len(rows)}.")
    return sorted(rows, key=lambda r: (_rank(role, r[id_key]), r[id_key]))[:count]


def _clinc_example(row: dict, split: str) -> dict:
    return {
        "sample_id": row["sample_id"], "task": "clinc", "split": split,
        "group_id": row["sample_id"], "state": row["text"],
        "instructions": CLINC_INSTRUCTIONS, "labels": BANKING_LABELS.copy(),
        "criteria": {k: BANKING_CRITERIA[k] for k in BANKING_LABELS},
        "source_split": row["source_split"],
    }


def _prepare_clinc(source: dict, domains: dict) -> tuple[list[dict], dict[str, str], dict]:
    banking = BANKING_LABELS[:-1]
    if sorted(domains.get("banking", [])) != banking:
        raise DataError("CLINC banking label set differs from the frozen fifteen-intent protocol.")
    expected_source = {"train": 15000, "val": 3000, "test": 4500,
                       "oos_train": 100, "oos_val": 100, "oos_test": 1000}
    pools = {}
    seen = {}
    for split, total in expected_source.items():
        if len(source.get(split, [])) != total:
            raise DataError(f"Unexpected CLINC {split} count.")
        rows = []
        for index, item in enumerate(source[split]):
            text, label = item
            if split.startswith("oos_") and label != "oos":
                raise DataError(f"Unexpected label in {split}: {label}")
            if label not in BANKING_LABELS:
                continue
            sample_id = f"clinc:{split}:{index}:{_sha(text.encode('utf-8'))[:16]}"
            key = normalized(text)
            if key in seen:
                raise DataError(f"Duplicate CLINC input across target pools: {seen[key]} and {sample_id}.")
            seen[key] = sample_id
            rows.append({"sample_id": sample_id, "text": text, "label": label, "source_split": split})
        pools[split] = rows
    for split, each in (("train", 100), ("val", 20), ("test", 30)):
        if Counter(r["label"] for r in pools[split]) != Counter({k: each for k in banking}):
            raise DataError(f"Unexpected CLINC per-label counts in {split}.")
    cal = []
    development = []
    for label in banking:
        rows = [r for r in pools["train"] if r["label"] == label]
        selected = _sample(rows, 20, "cal")
        cal.extend(selected)
        used = {r["sample_id"] for r in selected}
        development.extend(_sample([r for r in rows if r["sample_id"] not in used], 4, "development"))
    cal.extend(pools["oos_train"])
    test_oos = _sample(pools["oos_test"], 150, "test")
    test_oos_ids = {r["sample_id"] for r in test_oos}
    partition = {
        "calibration": cal,
        "validation": pools["val"] + pools["oos_val"],
        "test": pools["test"] + test_oos,
        "stress": [r for r in pools["oos_test"] if r["sample_id"] not in test_oos_ids],
        "development": development + [
            {"sample_id": f"clinc:authored-oos:{i}", "text": text,
             "label": "oos", "source_split": "authored_development_fixture"}
            for i, text in enumerate(OOS_FIXTURES)
        ],
    }
    examples, gold = [], {}
    for split, rows in partition.items():
        for row in rows:
            examples.append(_clinc_example(row, split))
            gold[row["sample_id"]] = row["label"]
    return examples, gold, {"source_counts": expected_source, "development_authored_oos": 4}


def _flatten_squad(source: dict, source_split: str) -> list[dict]:
    rows = []
    for article in source["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                rows.append({
                    "qid": qa["id"], "title": article["title"],
                    "context": paragraph["context"], "question": qa["question"],
                    "label": "unanswerable" if qa["is_impossible"] else "answerable",
                    "source_split": source_split,
                })
    return rows


def _deduplicate_squad(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    ids = [r["qid"] for r in rows]
    if len(set(ids)) != len(ids):
        raise DataError("Duplicate SQuAD question IDs.")
    groups = defaultdict(list)
    for row in rows:
        groups[(normalized(row["context"]), normalized(row["question"]))].append(row)
    retained, exclusions = [], []
    for group in groups.values():
        group.sort(key=lambda r: r["qid"])
        if len({r["label"] for r in group}) > 1:
            for row in group:
                exclusions.append({"sample_id": f"squad:{row['qid']}",
                                   "source_split": row["source_split"],
                                   "reason": "conflicting_labels_for_same_normalized_input",
                                   "group_question_ids": [r["qid"] for r in group]})
        else:
            retained.append(group[0])
            for row in group[1:]:
                exclusions.append({"sample_id": f"squad:{row['qid']}",
                                   "source_split": row["source_split"],
                                   "reason": "same_label_duplicate_input",
                                   "canonical_question_id": group[0]["qid"]})
    return retained, exclusions


def _check_squad_grouping(train: list[dict], dev: list[dict]) -> None:
    for name, rows in (("train", train), ("dev", dev)):
        owner = {}
        for row in rows:
            key = normalized(row["context"])
            if key in owner and owner[key] != row["title"]:
                raise DataError(f"SQuAD {name} context shared across article titles; article split would leak.")
            owner[key] = row["title"]
    fields = {
        "title": lambda r: normalized(r["title"]),
        "context": lambda r: normalized(r["context"]),
        "question_id": lambda r: r["qid"],
    }
    for name, getter in fields.items():
        if {getter(r) for r in train} & {getter(r) for r in dev}:
            raise DataError(f"SQuAD train/dev overlap in {name}.")


def _balanced_squad(rows: list[dict], each: int, role: str) -> list[dict]:
    return [r for label in SQUAD_LABELS for r in
            _sample([r for r in rows if r["label"] == label], each, role, "qid")]


def _squad_example(row: dict, split: str) -> dict:
    return {
        "sample_id": f"squad:{row['qid']}", "task": "squad", "split": split,
        "group_id": f"squad:article:{row['title']}",
        "state": f"Passage:\n{row['context']}\n\nQuestion:\n{row['question']}",
        "instructions": SQUAD_INSTRUCTIONS, "labels": SQUAD_LABELS.copy(),
        "criteria": SQUAD_CRITERIA.copy(), "source_split": row["source_split"],
        "context_hash": _sha(normalized(row["context"]).encode("utf-8")),
        "input_hash": _sha((normalized(row["context"]) + "\0" + normalized(row["question"])).encode("utf-8")),
    }


def _prepare_squad(train_source: dict, dev_source: dict) -> tuple[list[dict], dict[str, str], list[dict], dict]:
    train = _flatten_squad(train_source, "train")
    dev = _flatten_squad(dev_source, "dev")
    if len(train) != 130319 or len(dev) != 11873:
        raise DataError("Unexpected raw SQuAD QA counts.")
    _check_squad_grouping(train, dev)
    clean_train, exclusions_train = _deduplicate_squad(train)
    clean_dev, exclusions_dev = _deduplicate_squad(dev)
    if len(clean_train) != 130183 or len(clean_dev) != 11852:
        raise DataError("SQuAD duplicate/conflict audit differs from the frozen protocol.")
    titles = sorted({r["title"] for r in clean_train}, key=lambda title: (_rank("article", title), title))
    if len(titles) != 442 or len({r["title"] for r in clean_dev}) != 35:
        raise DataError("Unexpected SQuAD article counts.")
    n_dev, n_cal = math.floor(len(titles) / 5), math.floor(2 * len(titles) / 5)
    title_roles = {title: ("development" if i < n_dev else "calibration" if i < n_dev + n_cal else "validation")
                   for i, title in enumerate(titles)}
    pools = {role: [r for r in clean_train if title_roles[r["title"]] == role]
             for role in ("development", "calibration", "validation")}
    development = []
    for label in SQUAD_LABELS:
        rows = [r for r in pools["development"] if r["label"] == label]
        longest = sorted(rows, key=lambda r: (-(len(r["context"]) + len(r["question"])), r["qid"]))[:2]
        selected_ids = {r["qid"] for r in longest}
        development.extend(longest + _sample([r for r in rows if r["qid"] not in selected_ids], 30, "development", "qid"))
    partition = {
        "development": development,
        "calibration": _balanced_squad(pools["calibration"], 500, "calibration"),
        "validation": _balanced_squad(pools["validation"], 500, "validation"),
        "test": _balanced_squad(clean_dev, 1000, "test"),
    }
    examples, gold = [], {}
    for split, rows in partition.items():
        for row in rows:
            examples.append(_squad_example(row, split))
            gold[f"squad:{row['qid']}"] = row["label"]
    audit = {
        "raw_qa_counts": {"train": len(train), "dev": len(dev)},
        "retained_qa_counts": {"train": len(clean_train), "dev": len(clean_dev)},
        "article_pool_counts": dict(Counter(title_roles.values())),
        "selected_article_counts": {role: len({r["title"] for r in rows}) for role, rows in partition.items()},
        "article_assignment": title_roles,
        "duplicate_key": "NFKC + casefold + collapse whitespace of (context, question); input text is unchanged",
    }
    return examples, gold, exclusions_train + exclusions_dev, audit


def _validate_examples(examples: list[dict], gold: dict[str, str]) -> dict:
    ids = [row["sample_id"] for row in examples]
    if len(ids) != len(set(ids)) or set(ids) != set(gold):
        raise DataError("Sample IDs are duplicated or gold does not exactly cover the input manifest.")
    expected = {
        "clinc": {"development": 64, "calibration": 400, "validation": 400, "test": 600, "stress": 850},
        "squad": {"development": 64, "calibration": 1000, "validation": 1000, "test": 2000},
    }
    counts = {task: dict(Counter(row["split"] for row in examples if row["task"] == task)) for task in expected}
    if counts != expected:
        raise DataError(f"Final split counts differ from protocol: {counts}.")
    owner = {}
    for row in examples:
        if "gold" in row or "label" in row or "answers" in row or "is_impossible" in row:
            raise DataError("Gold fields leaked into model input records.")
        if gold[row["sample_id"]] not in row["labels"] or set(row["labels"]) != set(row["criteria"]):
            raise DataError("Label/criteria mismatch.")
        if row["task"] == "squad":
            for kind in ("group_id", "context_hash", "input_hash"):
                key = (kind, row[kind])
                if key in owner and owner[key] != row["split"]:
                    raise DataError(f"Cross-role SQuAD leakage: {kind}.")
                owner[key] = row["split"]
    for task in expected:
        for split in expected[task]:
            distribution = Counter(gold[r["sample_id"]] for r in examples if r["task"] == task and r["split"] == split)
            if task == "squad" and len(set(distribution.values())) != 1:
                raise DataError("SQuAD split is not label balanced.")
            if task == "clinc" and split in ("calibration", "validation", "test"):
                if distribution["oos"] * 4 != expected[task][split]:
                    raise DataError("CLINC OOS prevalence differs from 25%.")
    return counts


def prepare_data(output_dir: Path, cache_dir: Path | None = None, download: bool = True) -> dict:
    """Build immutable source-checked manifests and return their metadata.

    Existing identical manifests are accepted. A different or corrupted manifest
    is never silently overwritten. Failures before publication leave no usable
    manifest.json; source downloads may remain in the verified cache.
    """
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else output_dir / "raw"
    sources = {name: json.loads(_source_bytes(cache_dir, name, download)) for name in SOURCES}
    clinc, gold_clinc, audit_clinc = _prepare_clinc(sources["data_full.json"], sources["domains.json"])
    squad, gold_squad, exclusions, audit_squad = _prepare_squad(sources["train-v2.0.json"], sources["dev-v2.0.json"])
    examples = sorted(clinc + squad, key=lambda r: (r["task"], r["split"], r["sample_id"]))
    gold = {**gold_clinc, **gold_squad}
    counts = _validate_examples(examples, gold)
    exclusions.sort(key=lambda r: r["sample_id"])
    payloads = {"examples.jsonl": _jsonl_bytes(examples), "gold.json": _json_bytes(gold),
                "exclusions.jsonl": _jsonl_bytes(exclusions)}
    hashes = {name: _sha(value) for name, value in payloads.items()}
    fingerprint = _sha(_json_bytes({"protocol": PROTOCOL, "seed": SEED, "sources": SOURCES, "files": hashes}))
    manifest = {
        "protocol": PROTOCOL, "seed": SEED, "fingerprint": fingerprint,
        "sources": SOURCES, "files": hashes, "counts": counts,
        "formal_unique_inputs": 6250, "development_inputs": 128,
        "expected_predictions": {"kev": 6250, "startlux": 6250, "qwen": 3450},
        "label_order": {"clinc": BANKING_LABELS, "squad": SQUAD_LABELS},
        "audit": {"clinc": audit_clinc, "squad": audit_squad,
                  "exclusions_by_reason": dict(Counter(r["reason"] for r in exclusions))},
        "stress_definition": "450 banking main-test inputs reused with the 850 stress OOS inputs; not an independent replicate",
        "token_preflight_required": True, "truncation_allowed": False,
        "upstream_exposure": "Public model training/development exposure is incompletely known; these splits prevent project-level leakage only.",
    }
    payloads["manifest.json"] = _json_bytes(manifest)
    destination = output_dir / "data"
    if (destination / "manifest.json").exists():
        for name, content in payloads.items():
            if not (destination / name).exists() or (destination / name).read_bytes() != content:
                raise DataError(f"Existing data artifact differs: {destination / name}. Use a new output directory.")
        return manifest
    # Publish the manifest last: it is the commit marker for the dataset.
    for name, content in payloads.items():
        _atomic_write(destination / name, content)
    return manifest


def verify_data(output_dir: Path) -> dict:
    directory = Path(output_dir) / "data"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise DataError("Unsupported dataset protocol.")
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise DataError("Unsafe data manifest filename.")
        if _sha((directory / name).read_bytes()) != expected:
            raise DataError(f"Prepared data checksum mismatch: {name}.")
    expected_fingerprint = _sha(_json_bytes({"protocol": manifest["protocol"], "seed": manifest["seed"],
                                           "sources": manifest["sources"], "files": manifest["files"]}))
    if expected_fingerprint != manifest["fingerprint"]:
        raise DataError("Dataset fingerprint mismatch.")
    return manifest


def load_examples(output_dir: Path, task: str | None = None, split: str | None = None) -> list[dict]:
    verify_data(output_dir)
    path = Path(output_dir) / "data" / "examples.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if (task is None or row["task"] == task) and (split is None or row["split"] == split)]


def load_gold(output_dir: Path) -> dict[str, str]:
    verify_data(output_dir)
    return json.loads((Path(output_dir) / "data" / "gold.json").read_text(encoding="utf-8"))


def select_benchmark(examples: list[dict], gold: dict[str, str], task: str) -> list[dict]:
    """Select 200 primary-test inputs without looking at model predictions."""
    if task not in ("clinc", "squad"):
        raise DataError(f"Unknown task {task}.")
    rows = [r for r in examples if r["task"] == task and r["split"] == "test"]
    quotas = ({label: (50 if label == "oos" else 10) for label in BANKING_LABELS}
              if task == "clinc" else {label: 100 for label in SQUAD_LABELS})
    selected = []
    for label, count in quotas.items():
        selected.extend(_sample([r for r in rows if gold[r["sample_id"]] == label], count, "benchmark"))
    return sorted(selected, key=lambda r: (_rank("benchmark-order", r["sample_id"]), r["sample_id"]))
