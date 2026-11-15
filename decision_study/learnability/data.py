"""Pinned, premise-disjoint ANLI allocation for the complete learnability run.

Preparation uses train labels for balanced sampling. Official dev labels are
projected away. Evaluation labels are obtained only by the freeze-gated caller
through evaluation_gold(); the source ZIP itself is not a security boundary.
"""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile

from ..v3.data import CRITERIA, INSTRUCTIONS, LABELS, LABEL_MAP, DELIMITER, SOURCE_COUNTS

SOURCE_URL = "https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip"
SOURCE_SHA256 = "e5c058f2bb4e6190b0651badca2c590e45db95248f8b28cde674615ee40820bf"
HISTORY_SHA256 = "a470ad0143224dd87906059651efc74b1c8342445dcfbffcea11768f37618544"
HISTORY_COUNT = 5990
DEV_MEMBERSHIP_SHA256 = "558a4842c6513b8224625c3ca8527b98ff2aaf872ea38ea4d10817354e62b5a4"
DEV_COUNT = 1737
SEED = 20261112
FIT_SIZES = (1800, 6000, 18000)
SELECTION_PER_STRATUM = 100
HOLDOUT_PER_STRATUM = 200
ROUNDS = ("R1", "R2", "R3")
STRATA = tuple((rd, label) for rd in ROUNDS for label in LABELS)
SPLITS = ("fitting", "selection", "eligible_dev", "train_holdout")
INPUT_FIELDS = {"sample_id", "task", "split", "group_id", "round", "source_split", "state",
                "instructions", "labels", "criteria", "premise_hash", "input_hash"}
MAX_SOURCE_BYTES = 100 * 1024 * 1024
FILES = ("data/examples.jsonl", "data/fit_gold.json", "data/nested_fit_ids.json")


class DataError(ValueError):
    pass


def sha(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def jsonl(rows):
    return b"".join((json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode() for r in rows)


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def hashes(premise, hypothesis):
    return sha(normalized(premise).encode()), sha((normalized(premise) + "\0" + normalized(hypothesis)).encode())


def safe(root, relative):
    root = Path(os.path.abspath(os.path.expanduser(str(root))))
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or not part.parts or "\\" in relative:
        raise DataError("Unsafe prepared-data path")
    result = root.joinpath(*part.parts)
    if any(p.is_symlink() for p in (result, *result.parents)):
        raise DataError("Symlink in prepared-data path")
    return result


def fixed_write(path, payload):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise DataError(f"Existing prepared evidence differs: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def configuration():
    return {"id": "anli-learnability-data-v1", "seed": SEED, "fit_sizes": list(FIT_SIZES),
            "selection_per_stratum": SELECTION_PER_STRATUM, "train_holdout_per_stratum": HOLDOUT_PER_STRATUM,
            "strata_order": [list(x) for x in STRATA],
            "allocation": "train_holdout then selection then fitting; SHA-ranked premise groups; skip groups violating future stratum capacity; choose needed SHA-ranked rows; unselected group members remain unused",
            "nested_remainder": "first strata in fixed order receive one additional row; each smaller size is a prefix per stratum",
            "duplicate_rule": "drop conflicting normalized train inputs; same-label duplicates retain lexicographically first sample_id",
            "external_usage": "unconfirmed; local historical-use audit only"}


def acquire_source(target, candidates=()):
    """Use a verified local archive or bounded download; never replace a file."""
    target = Path(target)
    target = safe(target.parent, target.name)
    if target.exists():
        if file_hash(target) != SOURCE_SHA256:
            raise DataError("Existing ANLI source SHA differs")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.part")
    try:
        candidate = next((Path(p) for p in candidates if Path(p).is_file() and file_hash(p) == SOURCE_SHA256), None)
        if candidate is not None:
            shutil.copyfile(candidate, temporary)
        else:
            download_source(temporary)
        if file_hash(temporary) != SOURCE_SHA256:
            raise DataError("Downloaded/copied ANLI source SHA differs")
        # Atomic creation with no replacement, including a concurrent creator.
        os.link(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)



def download_source(temporary):
    """The absolute subprocess timeout also bounds a slowly trickling HTTP read."""
    script = """import sys,urllib.request
size=0
with urllib.request.urlopen(sys.argv[1],timeout=30) as response, open(sys.argv[2],'xb') as stream:
 while True:
  block=response.read(1024*1024)
  if not block: break
  size+=len(block)
  if size>int(sys.argv[3]): raise ValueError('Source download exceeds byte bound')
  stream.write(block)
"""
    try:
        subprocess.run([sys.executable, "-I", "-c", script, SOURCE_URL, str(temporary), str(MAX_SOURCE_BYTES)],
                       check=True, timeout=330, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired as exc:
        raise DataError("ANLI source download exceeded its absolute watchdog") from exc
    except subprocess.CalledProcessError as exc:
        raise DataError("ANLI source download failed; existing source files were not changed") from exc

def input_projection(pairs):
    return {k: v for k, v in pairs if k in {"uid", "context", "hypothesis"}}


def train_projection(pairs):
    return {k: v for k, v in pairs if k in {"uid", "context", "hypothesis", "label"}}


def read_source(path):
    if file_hash(path) != SOURCE_SHA256:
        raise DataError("Pinned official source SHA differs")
    result = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in names):
            raise DataError("Unsafe/duplicate source ZIP members")
        for rd in ROUNDS:
            for split, expected in SOURCE_COUNTS[rd].items():
                member = f"anli_v1.0/{rd}/{split}.jsonl"
                if archive.getinfo(member).file_size > MAX_SOURCE_BYTES:
                    raise DataError("Source member exceeds bound")
                lines = [line for line in archive.read(member).splitlines() if line.strip()]
                if len(lines) != expected:
                    raise DataError("Official source count differs")
                for line in lines:
                    raw = json.loads(line, object_pairs_hook=train_projection if split == "train" else input_projection)
                    if any(not isinstance(raw.get(k), str) or not raw[k].strip() for k in ("uid", "context", "hypothesis")):
                        raise DataError("Malformed official input")
                    sid = "anli:" + raw["uid"]
                    if sid in result or DELIMITER in raw["context"] or DELIMITER in raw["hypothesis"]:
                        raise DataError("Duplicate UID or ambiguous input delimiter")
                    ph, ih = hashes(raw["context"], raw["hypothesis"])
                    row = {"sample_id": sid, "round": rd, "source_split": split, "premise_hash": ph,
                           "input_hash": ih, "premise": raw["context"], "hypothesis": raw["hypothesis"]}
                    if split == "train":
                        if raw.get("label") not in LABEL_MAP:
                            raise DataError("Invalid training label")
                        row["gold"] = LABEL_MAP[raw["label"]]
                    result[sid] = row
    return result


def text_pair(example):
    state = example.get("state")
    if not isinstance(state, str) or not state.startswith("Premise:\n") or state.count(DELIMITER) != 1:
        raise DataError("Malformed canonical ANLI state")
    return tuple(state[len("Premise:\n"):].split(DELIMITER))


def history(path, source):
    if file_hash(path) != HISTORY_SHA256:
        raise DataError("Pinned historical V3 inputs SHA differs")
    ids = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("sample_id")
        if sid not in source or sid in ids:
            raise DataError("Unknown or repeated historical input")
        ph, ih = hashes(*text_pair(row))
        if (ph, ih) != (source[sid]["premise_hash"], source[sid]["input_hash"]):
            raise DataError("Historical source input identity differs")
        if (row.get("premise_hash"), row.get("input_hash")) != (ph, ih):
            raise DataError("Historical stored input hash differs")
        ids.add(sid)
    if len(ids) != HISTORY_COUNT:
        raise DataError("Historical input count differs")
    return ids


def membership(rows):
    keys = ("sample_id", "round", "source_split", "premise_hash", "input_hash")
    return [{k: r[k] for k in keys} for r in sorted(rows, key=lambda r: r["sample_id"])]


def rank(role, value):
    return sha(f"{SEED}|{role}|{value}".encode())


def quotas(size):
    q, r = divmod(size, len(STRATA))
    return {s: q + int(i < r) for i, s in enumerate(STRATA)}


def build(source, used):
    historical_premises = {source[sid]["premise_hash"] for sid in used}
    dev = [r for r in source.values() if r["source_split"] == "dev" and r["premise_hash"] not in historical_premises]
    if len(dev) != DEV_COUNT or sha(jsonl(membership(dev))) != DEV_MEMBERSHIP_SHA256:
        raise DataError("Eligible official-dev membership differs from accepted input audit")
    dev_premises = {r["premise_hash"] for r in dev}
    test_premises = {r["premise_hash"] for r in source.values() if r["source_split"] == "test"}
    by_input = defaultdict(list)
    for row in source.values():
        if row["source_split"] == "train" and row["premise_hash"] not in test_premises | dev_premises:
            by_input[row["input_hash"]].append(row)
    clean, conflicts, duplicates = [], 0, 0
    for group in by_input.values():
        if len({r["gold"] for r in group}) != 1:
            conflicts += len(group)
        else:
            clean.append(min(group, key=lambda r: r["sample_id"]))
            duplicates += len(group)-1
    groups = defaultdict(list)
    for row in clean:
        groups[row["premise_hash"]].append(row)
    counts = {ph: Counter((r["round"], r["gold"]) for r in rows) for ph, rows in groups.items()}
    remaining = Counter((r["round"], r["gold"]) for r in clean)
    selected = {"eligible_dev": dev}
    assigned = {}
    fit_quota = quotas(max(FIT_SIZES))
    for role, each in (("train_holdout", HOLDOUT_PER_STRATUM), ("selection", SELECTION_PER_STRATUM)):
        need = {s: each for s in STRATA}
        reserve = {s: fit_quota[s] + (SELECTION_PER_STRATUM if role == "train_holdout" else 0) for s in STRATA}
        candidates = [ph for ph in groups if ph not in assigned and (role != "train_holdout" or ph not in historical_premises)]
        chosen = []
        # Fixed SHA order is independent of group size and all model outcomes.
        for ph in sorted(candidates, key=lambda value: (rank(role, value), value)):
            contribution = counts[ph]
            if not any(need[s] and contribution[s] for s in STRATA):
                continue
            if any(remaining[s]-contribution[s] < reserve[s] for s in STRATA):
                continue
            assigned[ph] = role
            remaining.subtract(contribution)
            for stratum in STRATA:
                eligible = [r for r in groups[ph] if (r["round"], r["gold"]) == stratum]
                eligible.sort(key=lambda row: (rank(role, row["sample_id"]), row["sample_id"]))
                picked = eligible[:need[stratum]]
                need[stratum] -= len(picked)
                chosen.extend(picked)
            if not any(need.values()):
                break
        if any(need.values()):
            raise DataError(f"Whole-premise {role} allocation infeasible; deficits {need}; no automatic recipe change")
        selected[role] = chosen
    fitting = defaultdict(list)
    for row in clean:
        if row["premise_hash"] not in assigned:
            fitting[row["round"], row["gold"]].append(row)
    for stratum in STRATA:
        fitting[stratum].sort(key=lambda row: (rank("fitting", row["sample_id"]), row["sample_id"]))
        if len(fitting[stratum]) < fit_quota[stratum]:
            raise DataError("Insufficient fitting stratum after group assignment")
    selected["fitting"] = [r for s in STRATA for r in fitting[s][:fit_quota[s]]]
    nested = {str(size): sorted(r["sample_id"] for s, count in quotas(size).items() for r in fitting[s][:count]) for size in FIT_SIZES}
    public = sorted((example(row, role) for role, rows in selected.items() for row in rows), key=lambda r: (SPLITS.index(r["split"]), r["sample_id"]))
    fitting_gold = {r["sample_id"]: r["gold"] for role in ("fitting", "selection") for r in selected[role]}
    stats = {role: {"count": len(rows), "premise_groups": len({r["premise_hash"] for r in rows}),
                    "round_counts": dict(sorted(Counter(r["round"] for r in rows).items())),
                    "membership_sha256": sha(jsonl(membership(rows)))} for role, rows in selected.items()}
    for role in ("fitting", "selection", "train_holdout"):
        stats[role]["allocation_class_counts"] = {"/".join(s): sum((r["round"], r["gold"]) == s for r in selected[role]) for s in STRATA}
    audit = {"roles": stats, "training_cleanup": {"conflicting_rows_excluded": conflicts, "same_label_duplicates_excluded": duplicates},
             "remaining_before_fitting": {"/".join(s): remaining[s] for s in STRATA},
             "historical_count": len(used), "historical_premise_groups": len(historical_premises),
             "official_test_premise_groups": len(test_premises), "dev_labels_used": False,
             "train_labels_used_for_allocation": True, "holdout_target_population": "fixed SHA/group/quota-selected retained subsets, not unbiased estimates of the original complete source pools"}
    return public, fitting_gold, nested, audit


def example(row, split):
    return {"sample_id": row["sample_id"], "task": "anli", "split": split,
            "group_id": "anli:premise:" + row["premise_hash"], "round": row["round"],
            "source_split": row["source_split"], "premise_hash": row["premise_hash"], "input_hash": row["input_hash"],
            "state": "Premise:\n" + row["premise"] + DELIMITER + row["hypothesis"],
            "instructions": INSTRUCTIONS, "labels": list(LABELS), "criteria": CRITERIA.copy()}


def prepare(source_archive, historical_v3_examples, output_dir):
    source = read_source(source_archive)
    used = history(historical_v3_examples, source)
    examples, gold, nested, audit = build(source, used)
    contents = {FILES[0]: jsonl(examples), FILES[1]: canonical(gold), FILES[2]: canonical(nested)}
    manifest = {"configuration": configuration(), "source_sha256": SOURCE_SHA256, "history_sha256": HISTORY_SHA256,
                "eligible_dev_membership_sha256": DEV_MEMBERSHIP_SHA256,
                "files": {name: {"sha256": sha(data), "bytes": len(data)} for name, data in contents.items()}, **audit}
    for name, payload in contents.items():
        fixed_write(safe(output_dir, name), payload)
    fixed_write(safe(output_dir, "data/manifest.json"), canonical(manifest))
    load_prepared(output_dir)
    return manifest


def validate(examples, gold, nested, manifest):
    if (manifest.get("configuration") != configuration() or manifest.get("source_sha256") != SOURCE_SHA256
            or manifest.get("history_sha256") != HISTORY_SHA256 or manifest.get("eligible_dev_membership_sha256") != DEV_MEMBERSHIP_SHA256):
        raise DataError("Prepared data source/configuration identity differs")
    if len({r.get("sample_id") for r in examples}) != len(examples):
        raise DataError("Repeated prepared ID")
    by_role = defaultdict(list)
    for row in examples:
        if (set(row) != INPUT_FIELDS or row["task"] != "anli" or row["split"] not in SPLITS or row["round"] not in ROUNDS
                or row["labels"] != list(LABELS) or row["criteria"] != CRITERIA or row["instructions"] != INSTRUCTIONS):
            raise DataError("Prepared input schema differs or contains unapproved fields")
        ph, ih = hashes(*text_pair(row))
        expected_split = "dev" if row["split"] == "eligible_dev" else "train"
        if ((row["premise_hash"], row["input_hash"]) != (ph, ih) or row["group_id"] != "anli:premise:"+ph
                or row["source_split"] != expected_split):
            raise DataError("Prepared input content/source identity differs")
        by_role[row["split"]].append(row)
    expected = {"fitting": max(FIT_SIZES), "selection": len(STRATA)*SELECTION_PER_STRATUM,
                "train_holdout": len(STRATA)*HOLDOUT_PER_STRATUM, "eligible_dev": DEV_COUNT}
    for role in SPLITS:
        if len(by_role[role]) != expected[role]:
            raise DataError("Prepared role count differs")
        observed = {"count": len(by_role[role]), "premise_groups": len({r["premise_hash"] for r in by_role[role]}),
                    "round_counts": dict(sorted(Counter(r["round"] for r in by_role[role]).items())),
                    "membership_sha256": sha(jsonl(membership(by_role[role])))}
        if any(manifest["roles"][role].get(k) != v for k, v in observed.items()):
            raise DataError("Prepared role statistics/membership differ")
        for other in SPLITS:
            if other != role and {r["premise_hash"] for r in by_role[role]} & {r["premise_hash"] for r in by_role[other]}:
                raise DataError("Prepared premise leaks across roles")
    if sha(jsonl(membership(by_role["eligible_dev"]))) != DEV_MEMBERSHIP_SHA256:
        raise DataError("Prepared candidate-dev membership differs")
    fitted = {r["sample_id"] for role in ("fitting", "selection") for r in by_role[role]}
    if set(gold) != fitted or any(label not in LABELS for label in gold.values()):
        raise DataError("Fitting gold contains evaluation IDs or incomplete labels")
    if set(nested) != {str(n) for n in FIT_SIZES}:
        raise DataError("Nested fitting sizes differ")
    fitting_ids = {r["sample_id"] for r in by_role["fitting"]}
    row_map = {r["sample_id"]: r for r in examples}
    previous = set()
    for size in FIT_SIZES:
        ids = nested[str(size)]
        if len(ids) != size or len(set(ids)) != size or not previous <= set(ids) <= fitting_ids:
            raise DataError("Fitting subsets are not exact nested distinct IDs")
        counts = Counter((row_map[sid]["round"], gold[sid]) for sid in ids)
        if dict(counts) != quotas(size):
            raise DataError("Fitting round/class quota differs")
        previous = set(ids)
    if previous != fitting_ids:
        raise DataError("Largest fitting subset does not cover fitting pool")
    selection_counts = Counter((r["round"], gold[r["sample_id"]]) for r in by_role["selection"])
    if dict(selection_counts) != {s: SELECTION_PER_STRATUM for s in STRATA}:
        raise DataError("Selection round/class quota differs")


def load_prepared(output_dir):
    manifest = json.loads(safe(output_dir, "data/manifest.json").read_text())
    if set(manifest.get("files", {})) != set(FILES):
        raise DataError("Prepared manifest file set differs")
    for name, identity in manifest["files"].items():
        path = safe(output_dir, name)
        if path.stat().st_size != identity.get("bytes") or file_hash(path) != identity.get("sha256"):
            raise DataError("Prepared file hash/size differs")
    examples = [json.loads(line) for line in safe(output_dir, FILES[0]).read_text().splitlines() if line.strip()]
    gold = json.loads(safe(output_dir, FILES[1]).read_text())
    nested = json.loads(safe(output_dir, FILES[2]).read_text())
    validate(examples, gold, nested, manifest)
    return {"examples": examples, "fit_gold": gold, "fit_ids_by_size": nested, "manifest": manifest}


def evaluation_gold(source_archive, manifest, examples):
    """Read final evaluation labels only after the orchestrator's model/policy freeze.

    This helper never writes labels. The orchestrator owns the immutable freeze and
    gold-access receipt, and must call this only after their prerequisite checks.
    """
    if file_hash(source_archive) != SOURCE_SHA256 or manifest.get("source_sha256") != SOURCE_SHA256:
        raise DataError("Evaluation source identity differs")
    if sha(jsonl(examples)) != manifest["files"][FILES[0]]["sha256"]:
        raise DataError("Evaluation input identity differs")
    selected = {r["sample_id"]: r for r in examples if r["split"] in {"eligible_dev", "train_holdout"}}
    result = {}
    with zipfile.ZipFile(source_archive) as archive:
        for rd in ROUNDS:
            for split in ("train", "dev"):
                for line in archive.read(f"anli_v1.0/{rd}/{split}.jsonl").splitlines():
                    raw = json.loads(line, object_pairs_hook=train_projection)
                    sid = "anli:"+raw["uid"]
                    if sid not in selected:
                        continue
                    row = selected[sid]
                    if (raw.get("label") not in LABEL_MAP or row["round"] != rd or row["source_split"] != split
                            or hashes(raw["context"], raw["hypothesis"]) != (row["premise_hash"], row["input_hash"])):
                        raise DataError("Evaluation gold does not match frozen inputs")
                    result[sid] = LABEL_MAP[raw["label"]]
    if set(result) != set(selected):
        raise DataError("Evaluation gold coverage differs")
    return result


evaluate_gold = evaluation_gold
