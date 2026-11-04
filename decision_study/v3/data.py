"""Pinned ANLI preparation with premise-disjoint fitting and explicit gold access."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import unicodedata
import zipfile

from ..common import canonical, file_hash, fingerprint, read_json

PROTOCOL = "solo-v3-anli"
PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT / "configs/v3_protocol.json"
SOURCE_URL = "https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip"
SOURCE_SHA256 = "e5c058f2bb4e6190b0651badca2c590e45db95248f8b28cde674615ee40820bf"
DUPLICATE_RULE = "exclude contradictory training input groups; same-label duplicates retain lexicographically first uid; preserve all official test"
ROUNDS = ("R1", "R2", "R3")
LABELS = ("entailment", "neutral", "contradiction")
LABEL_MAP = {"e": "entailment", "n": "neutral", "c": "contradiction"}
SOURCE_COUNTS = {"R1": {"train": 16946, "dev": 1000, "test": 1000},
                 "R2": {"train": 45460, "dev": 1000, "test": 1000},
                 "R3": {"train": 100459, "dev": 1200, "test": 1200}}
FIT_ROLES = ("calibration", "validation", "development")
SPLITS = (*FIT_ROLES, "holdout")
INPUTS, FIT_GOLD, HOLDOUT_GOLD, MANIFEST = (
    "data/examples.jsonl", "data/fit_gold.json", "sealed/holdout_gold.json", "data/manifest.json")
DELIMITER = "\n\nHypothesis:\n"
INSTRUCTIONS = ("Classify the relationship between the premise and hypothesis. "
                "Use only the given premise. Select entailment when the hypothesis must be true, "
                "contradiction when it must be false, and neutral when neither follows. "
                "Return exactly one listed label; do not use outside knowledge.")
CRITERIA = {"entailment": "The premise supports the hypothesis as necessarily true.",
            "neutral": "The premise neither entails nor contradicts the hypothesis.",
            "contradiction": "The premise supports the hypothesis as necessarily false."}
EXAMPLE_KEYS = {"sample_id", "task", "split", "group_id", "round", "source_split",
                "state", "instructions", "labels", "criteria", "premise_hash", "input_hash"}


class DataError(ValueError):
    pass


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _text_hash(text):
    return _sha(normalized(text).encode("utf-8"))


def _input_hash(premise, hypothesis):
    return _sha((normalized(premise) + "\0" + normalized(hypothesis)).encode("utf-8"))


def _rank(seed, role, value):
    return _sha(f"{seed}|{PROTOCOL}|{role}|{value}".encode("utf-8"))


def _protocol(configuration):
    data = configuration["data"]
    if (configuration.get("protocol") != PROTOCOL or data.get("task") != "anli"
            or data.get("duplicates") != DUPLICATE_RULE or data.get("source_sha256") != SOURCE_SHA256
            or data.get("source_url") != SOURCE_URL or data.get("rounds") != list(ROUNDS)
            or data.get("labels") != list(LABELS)
            or data.get("training_role_proportions") != {"calibration": .6, "validation": .3, "development": .1}):
        raise DataError("Unsupported ANLI source, labels, rounds or premise-role proportions")
    for role in FIT_ROLES:
        count = data.get(f"{role}_per_round_class")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise DataError(f"Invalid {role} sample count")
    return configuration


def _read_source(path):
    path = Path(path)
    if file_hash(path) != SOURCE_SHA256:
        raise DataError("ANLI archive SHA256 differs from the pinned official source")
    rows, members = [], {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):
            raise DataError("Unsafe or duplicate ANLI archive member")
        for round_name in ROUNDS:
            for split, expected in SOURCE_COUNTS[round_name].items():
                name = f"anli_v1.0/{round_name}/{split}.jsonl"
                payload = archive.read(name)
                records = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
                if len(records) != expected:
                    raise DataError(f"Unexpected official source count: {name}")
                members[name] = {"sha256": _sha(payload), "count": len(records)}
                for record in records:
                    if (record.get("label") not in LABEL_MAP
                            or any(not isinstance(record.get(k), str) or not record[k].strip()
                                   for k in ("uid", "context", "hypothesis"))):
                        raise DataError(f"Invalid ANLI source record: {name}")
                    # Whitelist source fields: reason/model_label/annotator fields
                    # never enter public examples, fitting features or prompts.
                    rows.append({"uid": record["uid"], "premise": record["context"],
                                 "hypothesis": record["hypothesis"], "label": LABEL_MAP[record["label"]],
                                 "round": round_name, "source_split": split})
    return rows, members


def _role(seed, premise_hash):
    bucket = int(_rank(seed, "premise-role", premise_hash)[:16], 16) % 1000000
    return "calibration" if bucket < 600000 else "validation" if bucket < 900000 else "development"


def _example(row, split):
    if DELIMITER in row["premise"] or DELIMITER in row["hypothesis"]:
        raise DataError("Ambiguous ANLI input delimiter; no silent input transformation")
    premise_hash = _text_hash(row["premise"])
    return {"sample_id": "anli:" + row["uid"], "task": "anli", "split": split,
            "group_id": "anli:premise:" + premise_hash, "premise_hash": premise_hash,
            "input_hash": _input_hash(row["premise"], row["hypothesis"]),
            "round": row["round"], "source_split": row["source_split"],
            "state": "Premise:\n" + row["premise"] + DELIMITER + row["hypothesis"],
            "instructions": INSTRUCTIONS, "labels": list(LABELS), "criteria": CRITERIA.copy()}


def _build(rows, configuration):
    """Pure selection helper, also exercised with explicitly artificial test rows."""
    _protocol(configuration)
    ids = [row["uid"] for row in rows]
    if len(ids) != len(set(ids)):
        raise DataError("Duplicate UID across official ANLI source files")
    by_input, by_premise = defaultdict(list), defaultdict(list)
    for row in rows:
        by_input[_input_hash(row["premise"], row["hypothesis"])].append(row)
        by_premise[_text_hash(row["premise"])].append(row)
    holdout = [row for row in rows if row["source_split"] == "test"]
    test_premises = {_text_hash(row["premise"]) for row in holdout}
    test_inputs = {_input_hash(row["premise"], row["hypothesis"]) for row in holdout}
    if len(test_inputs) != len(holdout):
        raise DataError("Unexpected duplicate official test input; preserve source and review, never drop test")
    excluded = defaultdict(list)
    eligible = []
    for row in rows:
        if row["source_split"] != "train":
            continue
        if _text_hash(row["premise"]) in test_premises or _input_hash(row["premise"], row["hypothesis"]) in test_inputs:
            excluded["test_premise_or_input_overlap"].append(row["uid"])
        else:
            eligible.append(row)
    train_inputs = defaultdict(list)
    for row in eligible:
        train_inputs[_input_hash(row["premise"], row["hypothesis"])].append(row)
    clean = []
    for group in train_inputs.values():
        ordered = sorted(group, key=lambda row: row["uid"])
        if len({row["label"] for row in ordered}) > 1:
            excluded["conflicting_training_input"].extend(row["uid"] for row in ordered)
        else:
            clean.append(ordered[0])
            excluded["duplicate_training_input"].extend(row["uid"] for row in ordered[1:])
    seed = configuration["seed"]
    pools = defaultdict(list)
    for row in clean:
        pools[(_role(seed, _text_hash(row["premise"])), row["round"], row["label"])].append(row)
    selected = []
    for split in FIT_ROLES:
        count = configuration["data"][f"{split}_per_round_class"]
        for round_name in ROUNDS:
            for label in LABELS:
                candidates = pools[(split, round_name, label)]
                if len(candidates) < count:
                    raise DataError(f"Insufficient {split}/{round_name}/{label}: {len(candidates)} < {count}; no pool replenishment")
                chosen = sorted(candidates, key=lambda row: (_rank(seed, f"{split}|{round_name}|{label}", row["uid"]), row["uid"]))[:count]
                selected.extend((row, split) for row in chosen)
    selected.extend((row, "holdout") for row in holdout)
    examples = sorted((_example(row, split) for row, split in selected), key=lambda row: (row["split"], row["round"], row["sample_id"]))
    fit_gold = {"anli:" + row["uid"]: row["label"] for row, split in selected if split != "holdout"}
    holdout_gold = {"anli:" + row["uid"]: row["label"] for row in holdout}
    duplicate_groups = [group for group in by_input.values() if len(group) > 1]
    audit = {
        "official_source_counts": dict(sorted(Counter(row["round"] + "/" + row["source_split"] for row in rows).items())),
        "source_uid_count": len(ids), "duplicate_uid_count": 0,
        "source_normalized_input_count": len(by_input), "source_duplicate_input_groups": len(duplicate_groups),
        "source_conflicting_input_groups": sum(len({row["label"] for row in group}) > 1 for group in duplicate_groups),
        "source_cross_file_input_groups": sum(len({(row["round"], row["source_split"]) for row in group}) > 1 for group in duplicate_groups),
        "source_premise_count": len(by_premise), "source_cross_file_premise_groups": sum(len({(row["round"], row["source_split"]) for row in group}) > 1 for group in by_premise.values()),
        "official_test_preserved_count": len(holdout), "official_test_ids_hash": fingerprint(sorted(holdout_gold)),
        "official_test_unique_inputs": len(test_inputs), "official_test_premise_count": len(test_premises),
        "train_after_test_overlap_exclusion": len(eligible), "train_after_duplicate_resolution": len(clean),
        "training_exclusion_counts": {key: len(value) for key, value in sorted(excluded.items())},
        "training_excluded_ids_hashes": {key: fingerprint(sorted(value)) for key, value in sorted(excluded.items())},
        "training_pool_counts": {"/".join(key): len(value) for key, value in sorted(pools.items())},
        "premise_role_rule": "sha256(seed|solo-v3-anli|premise-role|premise_hash) first 16 hex digits modulo 1000000: <600000 calibration, <900000 validation, otherwise development",
        "official_dev_use": "Audited only; never selected for fitting, development, validation or holdout",
        "test_sampling": "All official R1/R2/R3 test UIDs preserved without subsampling or balancing",
        "label_distributions": {split: dict(sorted(Counter(f"{row['round']}/{row['label']}" for row, role in selected if role == split).items())) for split in SPLITS},
    }
    return examples, fit_gold, holdout_gold, audit


def _payloads(examples, fit_gold, holdout_gold):
    return {INPUTS: b"".join(canonical(row) + b"\n" for row in examples),
            FIT_GOLD: canonical(fit_gold) + b"\n", HOLDOUT_GOLD: canonical(holdout_gold) + b"\n"}


def _manifest(examples, payloads, configuration, protocol_file_hash, source_members, audit, *, evidence_type="source_dataset"):
    roles = {}
    for role in (*SPLITS, "fit", "non_holdout"):
        selected = _select_role(examples, role)
        roles[role] = {"count": len(selected), "sample_ids_hash": fingerprint([x["sample_id"] for x in selected]),
                       "group_ids_hash": fingerprint(sorted({x["group_id"] for x in selected})),
                       "inputs_hash": fingerprint(selected)}
    result = {"schema_version": 1, "protocol": PROTOCOL, "evidence_type": evidence_type,
              "source": {"archive_name": "anli_v1.0.zip", "archive_sha256": SOURCE_SHA256,
                         "url": SOURCE_URL, "members": source_members},
              "protocol_identity": {"canonical_sha256": fingerprint(configuration), "file_sha256": protocol_file_hash,
                                    "configuration": configuration},
              "preparation_source_sha256": file_hash(Path(__file__)),
              "files": {name: _sha(payload) for name, payload in payloads.items()},
              "counts": {split: sum(x["split"] == split for x in examples) for split in SPLITS},
              "roles": roles, "audit": audit,
              "gold_access": "verify_fit/load_examples/load_fit_gold never open sealed gold; evaluation requires an existing matching freeze and first-access receipt"}
    result["manifest_hash"] = fingerprint(result)
    return result


def _select_role(examples, role):
    if role is None:
        return examples
    if role == "fit":
        return [row for row in examples if row["split"] in ("calibration", "validation")]
    if role == "non_holdout":
        return [row for row in examples if row["split"] != "holdout"]
    if role not in SPLITS:
        raise DataError(f"Unknown ANLI data role: {role}")
    return [row for row in examples if row["split"] == role]


@contextmanager
def _prepare_lock(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".prepare.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DataError("Another ANLI preparation owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _publish(output_dir, payloads, manifest):
    output_dir = Path(output_dir)
    all_payloads = {**payloads, MANIFEST: canonical(manifest) + b"\n"}
    with _prepare_lock(output_dir):
        for name, payload in all_payloads.items():
            path = _regular(output_dir, name, require_exists=False)
            if path.exists() and path.read_bytes() != payload:
                raise DataError(f"Existing ANLI data differs: {name}; preserve it and use a new output")
        for name, payload in all_payloads.items():
            path = output_dir / name
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            temporary.replace(path)


def prepare(raw_zip, output_dir, *, protocol_path=None):
    path = Path(protocol_path or PROTOCOL_PATH)
    configuration = _protocol(read_json(path))
    rows, source_members = _read_source(raw_zip)
    examples, fit_gold, holdout_gold, audit = _build(rows, configuration)
    payloads = _payloads(examples, fit_gold, holdout_gold)
    manifest = _manifest(examples, payloads, configuration, file_hash(path), source_members, audit)
    _publish(Path(output_dir), payloads, manifest)
    verify_fit(output_dir)
    return manifest


def _regular(output_dir, name, *, require_exists=True):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise DataError(f"Unsafe ANLI member: {name}")
    path = Path(output_dir)
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise DataError(f"Symlink ANLI member: {name}")
    if require_exists and not path.is_file():
        raise DataError(f"Missing ANLI member: {name}")
    return path


def _validate_rows(examples, fit_gold, manifest):
    ids, owners, inputs = set(), {}, {}
    for row in examples:
        if (set(row) != EXAMPLE_KEYS or row["task"] != "anli" or row["split"] not in SPLITS
                or row["round"] not in ROUNDS or row["labels"] != list(LABELS)
                or row["criteria"] != CRITERIA or row["instructions"] != INSTRUCTIONS):
            raise DataError("Invalid or gold-leaking ANLI input schema")
        if row["source_split"] != ("test" if row["split"] == "holdout" else "train"):
            raise DataError("Official ANLI source split differs from selected role")
        if row["sample_id"] in ids or not row["sample_id"].startswith("anli:"):
            raise DataError("Duplicate or invalid ANLI sample ID")
        ids.add(row["sample_id"])
        if not row["state"].startswith("Premise:\n") or row["state"].count(DELIMITER) != 1:
            raise DataError("Invalid ANLI premise/hypothesis rendering")
        premise, hypothesis = row["state"][len("Premise:\n"):].split(DELIMITER)
        context = _text_hash(premise)
        if (row["premise_hash"] != context or row["group_id"] != "anli:premise:" + context
                or row["input_hash"] != _input_hash(premise, hypothesis)):
            raise DataError("ANLI text/input/group identity mismatch")
        if context in owners and owners[context] != row["split"]:
            raise DataError("Premise overlap between ANLI roles")
        owners[context] = row["split"]
        if row["input_hash"] in inputs:
            raise DataError("Duplicate normalized ANLI model input")
        inputs[row["input_hash"]] = row["sample_id"]
    non_holdout = _select_role(examples, "non_holdout")
    if set(fit_gold) != {row["sample_id"] for row in non_holdout} or not set(fit_gold.values()) <= set(LABELS):
        raise DataError("Fitting gold must cover exactly non-holdout input IDs")
    config = manifest["protocol_identity"]["configuration"]
    for role in FIT_ROLES:
        selected = _select_role(examples, role)
        counts = Counter((row["round"], fit_gold[row["sample_id"]]) for row in selected)
        expected = {(round_name, label): config["data"][f"{role}_per_round_class"] for round_name in ROUNDS for label in LABELS}
        if counts != expected:
            raise DataError(f"ANLI {role} per-round/class counts changed")
        if any(_role(config["seed"], row["premise_hash"]) != role for row in selected):
            raise DataError("ANLI premise role assignment changed")
    holdout = _select_role(examples, "holdout")
    if fingerprint(sorted(row["sample_id"] for row in holdout)) != manifest["audit"]["official_test_ids_hash"]:
        raise DataError("Official test membership changed")
    if len(holdout) != manifest["audit"]["official_test_preserved_count"]:
        raise DataError("Official test count changed")
    if manifest["evidence_type"] == "source_dataset":
        if Counter(row["round"] for row in holdout) != {r: SOURCE_COUNTS[r]["test"] for r in ROUNDS}:
            raise DataError("Complete official ANLI test rounds were not preserved")
        if manifest["audit"]["official_source_counts"] != {f"{r}/{s}": n for r, splits in SOURCE_COUNTS.items() for s, n in splits.items()}:
            raise DataError("Pinned ANLI source counts changed")
    if set(manifest["roles"]) != {*SPLITS, "fit", "non_holdout"}:
        raise DataError("ANLI role inventory changed")
    for role, expected in manifest["roles"].items():
        selected = _select_role(examples, role)
        actual = {"count": len(selected), "sample_ids_hash": fingerprint([x["sample_id"] for x in selected]),
                  "group_ids_hash": fingerprint(sorted({x["group_id"] for x in selected})), "inputs_hash": fingerprint(selected)}
        if actual != expected:
            raise DataError(f"ANLI role identity changed: {role}")
    if manifest["counts"] != {split: sum(row["split"] == split for row in examples) for split in SPLITS}:
        raise DataError("ANLI split counts changed")


def verify_fit(output_dir):
    output_dir = Path(output_dir)
    manifest = read_json(_regular(output_dir, MANIFEST))
    if manifest.get("manifest_hash") != fingerprint({k: v for k, v in manifest.items() if k != "manifest_hash"}):
        raise DataError("ANLI manifest self hash changed")
    if (manifest.get("protocol") != PROTOCOL or manifest.get("schema_version") != 1
            or manifest.get("evidence_type") not in {"source_dataset", "software_test_fixture"}
            or manifest["source"]["archive_sha256"] != SOURCE_SHA256
            or manifest["source"]["url"] != SOURCE_URL
            or manifest.get("preparation_source_sha256") != file_hash(Path(__file__))):
        raise DataError("ANLI source/preparation identity changed")
    config = _protocol(manifest["protocol_identity"]["configuration"])
    if fingerprint(config) != manifest["protocol_identity"]["canonical_sha256"]:
        raise DataError("ANLI protocol identity changed")
    if set(manifest["files"]) != {INPUTS, FIT_GOLD, HOLDOUT_GOLD}:
        raise DataError("Unexpected ANLI data file inventory")
    for name in (INPUTS, FIT_GOLD):
        if file_hash(_regular(output_dir, name)) != manifest["files"][name]:
            raise DataError(f"ANLI file hash changed: {name}")
    examples = [json.loads(line) for line in _regular(output_dir, INPUTS).read_text().splitlines() if line.strip()]
    _validate_rows(examples, read_json(_regular(output_dir, FIT_GOLD)), manifest)
    return manifest


def load_examples(output_dir, role=None):
    verify_fit(output_dir)
    examples = [json.loads(line) for line in _regular(output_dir, INPUTS).read_text().splitlines() if line.strip()]
    return _select_role(examples, role)


def load_fit_gold(output_dir):
    verify_fit(output_dir)
    return read_json(_regular(output_dir, FIT_GOLD))


def _authorize_gold(output_dir, manifest, expected_freeze_hash):
    if not expected_freeze_hash:
        raise DataError("Explicit existing freeze identity is required for holdout access")
    freeze = read_json(_regular(output_dir, "policy_freeze.json"))
    if (freeze.get("freeze_hash") != expected_freeze_hash or expected_freeze_hash != fingerprint({k: v for k, v in freeze.items() if k != "freeze_hash"})
            or freeze.get("protocol") != PROTOCOL or freeze.get("schema_version") != 1
            or freeze.get("data_manifest_hash") != fingerprint(manifest)
            or freeze.get("protocol_hash") != manifest["protocol_identity"]["canonical_sha256"]):
        raise DataError("Policy freeze does not authorize this ANLI holdout")
    access = read_json(_regular(output_dir, "holdout_access.json"))
    if any(access.get(key) != freeze[key] for key in ("freeze_hash", "data_manifest_hash", "protocol_hash")):
        raise DataError("First-access receipt does not match policy/data/protocol")
    if "access_hash" in access and access["access_hash"] != fingerprint({k: v for k, v in access.items() if k != "access_hash"}):
        raise DataError("First-access receipt self hash changed")
    try:
        frozen_at = datetime.fromisoformat(freeze["created_at"])
        accessed_at = datetime.fromisoformat(access["created_at"])
        if frozen_at.tzinfo is None or accessed_at.tzinfo is None or accessed_at < frozen_at:
            raise ValueError("Invalid freeze/access ordering")
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError("Missing or invalid freeze/access timestamps") from exc


def load_holdout_gold_for_evaluation(output_dir, *, expected_freeze_hash):
    """Evaluator must check all policy/model sources before writing its access receipt."""
    manifest = verify_fit(output_dir)
    _authorize_gold(output_dir, manifest, expected_freeze_hash)
    path = _regular(output_dir, HOLDOUT_GOLD)
    if file_hash(path) != manifest["files"][HOLDOUT_GOLD]:
        raise DataError("ANLI sealed holdout hash changed")
    gold = read_json(path)
    examples = load_examples(output_dir, "holdout")
    if set(gold) != {row["sample_id"] for row in examples} or not set(gold.values()) <= set(LABELS):
        raise DataError("ANLI holdout labels/IDs differ from preserved official test")
    actual = dict(sorted(Counter(f"{row['round']}/{gold[row['sample_id']]}" for row in examples).items()))
    if actual != manifest["audit"]["label_distributions"]["holdout"]:
        raise DataError("ANLI official holdout label distribution changed")
    return gold


def verify(output_dir, *, include_sealed=False, expected_freeze_hash=None):
    manifest = verify_fit(output_dir)
    if include_sealed:
        load_holdout_gold_for_evaluation(output_dir, expected_freeze_hash=expected_freeze_hash)
    return manifest
