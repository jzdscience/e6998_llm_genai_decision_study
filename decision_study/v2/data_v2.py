"""Independent v2 inputs and sealed labels; never modifies a v1 artifact.

Preparation uses the source-verified SQuAD training file and excludes every
article represented in any v1 split. No model output is used for selection.
The shared inputs contain no gold. Fitting loaders never open sealed gold.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path

from .. import data as v1_data
from ..adapters import decision_request, qwen_request
from ..common import file_hash, fingerprint


PROTOCOL = "solo-v2"
SEED = "6998-solo-v2"
V1_DATA_FINGERPRINT = "839927242dd73b0e399e4a87c99ea81bca0965ccd3a71258f4e89886aa03ae57"
INPUTS = "data/examples.jsonl"
FIT_GOLD = "data/fit_gold.json"
SEALED_GOLD = "sealed/holdout_gold.json"
MANIFEST = "data/manifest.json"
FILES = (INPUTS, FIT_GOLD, SEALED_GOLD)
ROLES = ("qwen_fit", "holdout")


class DataV2Error(ValueError):
    """Invalid source, split, provenance, or immutable output."""


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _role(example: dict) -> str:
    split = example.get("split")
    if split in {"calibration", "validation"}:
        return "qwen_fit"
    if split == "holdout":
        return "holdout"
    raise DataV2Error(f"Unsupported v2 split: {split!r}")


def _rank(sample_id: str) -> tuple[str, str]:
    return _sha(f"{SEED}|holdout|{sample_id}".encode()), sample_id


def request_hashes(examples: list[dict]) -> dict[str, str]:
    """Fingerprint exact request-body bytes serialized by the frozen client."""
    result = {}
    ordered = sorted(examples, key=lambda row: row["sample_id"])
    for model in ("kev", "startlux", "qwen"):
        requests = []
        for example in ordered:
            body = qwen_request(example) if model == "qwen" else decision_request(example, model)
            wire = json.dumps(body, ensure_ascii=False).encode()
            requests.append({"sample_id": example["sample_id"], "sha256": _sha(wire)})
        result[model] = fingerprint(requests)
    return result


def _roles(examples: list[dict]) -> dict:
    result = {}
    for role in ROLES:
        rows = [row for row in examples if _role(row) == role]
        result[role] = {
            "count": len(rows),
            "sample_ids_hash": fingerprint(sorted(row["sample_id"] for row in rows)),
            "group_ids_hash": fingerprint(sorted({row["group_id"] for row in rows})),
            "task_split_counts": dict(sorted(Counter(f"{row['task']}/{row['split']}" for row in rows).items())),
            "request_hashes": request_hashes(rows),
        }
    return result


def _build_holdout(train_rows: list[dict], v1_examples: list[dict], per_class: int) -> tuple[list[dict], dict, dict]:
    """Pure selection helper, also exercised with clearly artificial fixtures."""
    if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class <= 0:
        raise DataV2Error("holdout_per_class must be a positive integer")
    if len({row["qid"] for row in train_rows}) != len(train_rows):
        raise DataV2Error("Duplicate question ID in deduplicated SQuAD source")
    all_squad = [row for row in v1_examples if row["task"] == "squad"]
    used_titles = {row["group_id"].removeprefix("squad:article:") for row in all_squad}
    used_ids = {row["sample_id"] for row in v1_examples}
    used_inputs = {row["input_hash"] for row in all_squad}
    used_contexts = {row["context_hash"] for row in all_squad}
    unused_titles = {row["title"] for row in train_rows} - used_titles
    candidates, gold = [], {}
    rejected = Counter()
    for raw in train_rows:
        if raw["title"] not in unused_titles:
            continue
        example = v1_data._squad_example(raw, "holdout")
        if example["sample_id"] in used_ids or example["input_hash"] in used_inputs or example["context_hash"] in used_contexts:
            rejected["v1_id_input_or_context_overlap"] += 1
            continue
        candidates.append(example)
        gold[example["sample_id"]] = raw["label"]
    if len({row["input_hash"] for row in candidates}) != len(candidates):
        raise DataV2Error("Repeated normalized input in deduplicated holdout candidates")
    selected = []
    for label in v1_data.SQUAD_LABELS:
        pool = [row for row in candidates if gold[row["sample_id"]] == label]
        if len(pool) < per_class:
            raise DataV2Error(f"Insufficient unused {label} questions: {len(pool)} < {per_class}")
        selected.extend(sorted(pool, key=lambda row: _rank(row["sample_id"]))[:per_class])
    selected.sort(key=lambda row: row["sample_id"])
    selected_gold = {row["sample_id"]: gold[row["sample_id"]] for row in selected}
    audit = {
        "unused_article_count": len(unused_titles),
        "unused_article_ids_hash": fingerprint(sorted(unused_titles)),
        "candidate_count": len(candidates),
        "candidate_label_counts": dict(Counter(gold.values())),
        "candidate_articles_by_label": {label: len({row["group_id"] for row in candidates if gold[row["sample_id"]] == label}) for label in v1_data.SQUAD_LABELS},
        "selected_article_count": len({row["group_id"] for row in selected}),
        "selected_label_counts": dict(Counter(selected_gold.values())),
        "overlap_exclusions": dict(rejected),
        "selection": "sha256(seed|holdout|sample_id), separately by gold class; original text unchanged",
        "independence": "Article-disjoint from all project v1 inputs; public train-origin data, not a pretraining-unseen claim.",
        "token_preflight": "Required before GPU inference; no truncation or silent sample replacement is permitted.",
    }
    return selected, selected_gold, audit


def _payloads(examples: list[dict], fit_gold: dict, holdout_gold: dict) -> dict[str, bytes]:
    return {INPUTS: v1_data._jsonl_bytes(examples), FIT_GOLD: v1_data._json_bytes(fit_gold),
            SEALED_GOLD: v1_data._json_bytes(holdout_gold)}


def _manifest(examples: list[dict], payloads: dict[str, bytes], *, provenance: dict,
              protocol_configuration: dict, protocol_file_hash: str, audit: dict,
              evidence_type: str = "source_dataset") -> dict:
    manifest = {
        "schema_version": 1, "protocol": PROTOCOL, "seed": SEED, "evidence_type": evidence_type,
        "files": {name: _sha(payloads[name]) for name in FILES},
        "counts": {role: sum(_role(row) == role for row in examples) for role in ROLES},
        "roles": _roles(examples), "v1": provenance,
        "protocol_identity": {"canonical_sha256": fingerprint(protocol_configuration),
                              "file_sha256": protocol_file_hash, "configuration": protocol_configuration},
        "source_hashes": {"train-v2.0.json": v1_data.SOURCES["train-v2.0.json"]["sha256"]},
        "preparation_source_hashes": {"data_v2.py": file_hash(Path(__file__)),
                                      "data.py": file_hash(Path(v1_data.__file__))},
        "audit": audit,
        "gold_access": "Fitting uses data/fit_gold.json only. Scoring sealed/holdout_gold.json requires the study policy-freeze gate.",
    }
    manifest["manifest_hash"] = fingerprint(manifest)
    return manifest


@contextmanager
def _preparation_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".prepare.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DataV2Error("Another data preparation is using this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _publish(output_dir: Path, payloads: dict[str, bytes], manifest: dict) -> None:
    """Publish the manifest last; refuse any differing complete or partial data."""
    output_dir = Path(output_dir).resolve()
    items = {**payloads, MANIFEST: v1_data._json_bytes(manifest)}
    with _preparation_lock(output_dir):
        for name, content in items.items():
            path = output_dir / name
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != output_dir.parent):
                raise DataV2Error(f"Refusing symlink in output data path: {name}")
            if path.exists() and path.read_bytes() != content:
                raise DataV2Error(f"Existing v2 artifact differs: {name}; use a new output directory")
        for name, content in items.items():
            if not (output_dir / name).exists():
                v1_data._atomic_write(output_dir / name, content)


def prepare(v1_dir: Path, raw_dir: Path, output_dir: Path, holdout_per_class: int = 2000,
            *, protocol_path: Path | None = None) -> dict:
    v1_dir, raw_dir, output_dir = Path(v1_dir).resolve(), Path(raw_dir).resolve(), Path(output_dir).resolve()
    if output_dir == v1_dir or v1_dir in output_dir.parents:
        raise DataV2Error("v2 output must be outside the immutable v1 directory")
    protocol_path = Path(protocol_path) if protocol_path else Path(__file__).resolve().parents[2] / "configs/v2_protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    configuration = json.loads(protocol_bytes)
    if configuration.get("data", {}).get("holdout_per_class") != holdout_per_class:
        raise DataV2Error("Requested holdout size differs from the frozen v2 protocol")
    manifest = v1_data.verify_data(v1_dir)
    if manifest["fingerprint"] != V1_DATA_FINGERPRINT:
        raise DataV2Error("v1 dataset differs from the completed frozen experiment")
    examples = v1_data.load_examples(v1_dir)
    gold = v1_data.load_gold(v1_dir)
    v1_data._validate_examples(examples, gold)
    fit = [row for row in examples if row["split"] in {"calibration", "validation"}]
    if len(fit) != 2800:
        raise DataV2Error("Expected exactly 2,800 original calibration/validation inputs")
    raw = v1_data._source_bytes(raw_dir, "train-v2.0.json", download=False)
    train = v1_data._flatten_squad(json.loads(raw), "train")
    v1_data._check_squad_grouping(train, [])
    clean, _ = v1_data._deduplicate_squad(train)
    holdout, holdout_gold, audit = _build_holdout(clean, examples, holdout_per_class)
    if audit["unused_article_count"] != 57:
        raise DataV2Error("Expected exactly 57 project-unused SQuAD training articles")
    assigned = manifest["audit"]["squad"]["article_assignment"]
    used = {row["group_id"].removeprefix("squad:article:") for row in examples if row["task"] == "squad" and row["source_split"] == "train"}
    audit["unused_v1_article_pool_counts"] = dict(Counter(role for title, role in assigned.items() if title not in used))
    fit_gold = {row["sample_id"]: gold[row["sample_id"]] for row in fit}
    combined = fit + holdout
    payloads = _payloads(combined, fit_gold, holdout_gold)
    provenance = {"data_fingerprint": manifest["fingerprint"], "manifest_sha256": file_hash(v1_dir / "data/manifest.json"),
                  "files": manifest["files"], "fit_request_hashes": request_hashes(fit),
                  "fit_rows_copied_without_changes": True}
    output_manifest = _manifest(combined, payloads, provenance=provenance,
                                protocol_configuration=configuration, protocol_file_hash=_sha(protocol_bytes), audit=audit)
    _publish(output_dir, payloads, output_manifest)
    verify(output_dir)
    return output_manifest


def _read_manifest(output_dir: Path) -> dict:
    manifest = json.loads((output_dir / MANIFEST).read_text(encoding="utf-8"))
    checksum = manifest.get("manifest_hash")
    if checksum != fingerprint({key: value for key, value in manifest.items() if key != "manifest_hash"}):
        raise DataV2Error("v2 manifest checksum mismatch")
    if manifest.get("schema_version") != 1 or manifest.get("protocol") != PROTOCOL or manifest.get("seed") != SEED:
        raise DataV2Error("Unsupported v2 data protocol")
    if set(manifest.get("files", {})) != set(FILES):
        raise DataV2Error("Unexpected v2 data file set")
    identity = manifest["protocol_identity"]
    if fingerprint(identity["configuration"]) != identity["canonical_sha256"]:
        raise DataV2Error("v2 protocol identity mismatch")
    return manifest


def _verify(output_dir: Path, *, sealed: bool) -> dict:
    output_dir = Path(output_dir).resolve()
    manifest = _read_manifest(output_dir)
    for name in FILES if sealed else (INPUTS, FIT_GOLD):
        path = output_dir / name
        if path.is_symlink() or not path.resolve().is_relative_to(output_dir):
            raise DataV2Error(f"Unsafe v2 artifact path: {name}")
        if file_hash(path) != manifest["files"][name]:
            raise DataV2Error(f"v2 data checksum mismatch: {name}")
    rows = [json.loads(line) for line in (output_dir / INPUTS).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise DataV2Error("Duplicate v2 sample ID")
    if any(set(row) & {"gold", "label", "answers", "is_impossible"} for row in rows):
        raise DataV2Error("Gold leaked into v2 model inputs")
    if _roles(rows) != manifest["roles"]:
        raise DataV2Error("v2 role or request identity mismatch")
    counts = {role: sum(_role(row) == role for row in rows) for role in ROLES}
    if counts != manifest["counts"]:
        raise DataV2Error("v2 counts mismatch")
    if counts["holdout"] != 2 * manifest["protocol_identity"]["configuration"]["data"]["holdout_per_class"]:
        raise DataV2Error("v2 holdout count differs from the protocol")
    fit = [row for row in rows if _role(row) == "qwen_fit"]
    holdout = [row for row in rows if _role(row) == "holdout"]
    if {row["group_id"] for row in fit if row["task"] == "squad"} & {row["group_id"] for row in holdout}:
        raise DataV2Error("Article leakage between v2 fitting and holdout")
    fit_gold = json.loads((output_dir / FIT_GOLD).read_text(encoding="utf-8"))
    if set(fit_gold) != {row["sample_id"] for row in fit} or any(fit_gold[row["sample_id"]] not in row["labels"] for row in fit):
        raise DataV2Error("Fit gold must cover fitting inputs only")
    if sealed:
        holdout_gold = json.loads((output_dir / SEALED_GOLD).read_text(encoding="utf-8"))
        if set(holdout_gold) != {row["sample_id"] for row in holdout}:
            raise DataV2Error("Sealed gold must cover holdout inputs only")
        expected = manifest["protocol_identity"]["configuration"]["data"]["holdout_per_class"]
        if dict(Counter(holdout_gold.values())) != {label: expected for label in v1_data.SQUAD_LABELS}:
            raise DataV2Error("Sealed holdout labels are not balanced")
    return manifest


def verify(output_dir: Path) -> dict:
    """Verify all data identities; does not predict or score any holdout item."""
    return _verify(output_dir, sealed=True)


def verify_fit(output_dir: Path) -> dict:
    """Verify fitting access without opening sealed holdout labels."""
    return _verify(output_dir, sealed=False)


def load_examples(output_dir: Path, role: str | None = None) -> list[dict]:
    if role is not None and role not in ROLES:
        raise DataV2Error(f"Unknown v2 role: {role!r}")
    verify_fit(output_dir)
    rows = [json.loads(line) for line in (Path(output_dir) / INPUTS).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if role is None or _role(row) == role]


def load_fit_gold(output_dir: Path) -> dict[str, str]:
    verify_fit(output_dir)
    return json.loads((Path(output_dir) / FIT_GOLD).read_text(encoding="utf-8"))
