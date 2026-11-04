"""Artificial data tests; these fixtures are never experimental predictions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_study import data as v1_data
from decision_study.adapters import qwen_request
from decision_study.common import fingerprint
from decision_study.v2 import data_v2 as data


def raw(title, qid, label="answerable", *, context=None, question=None):
    return {"title": title, "qid": qid, "context": context or f"ARTIFICIAL context for {title}",
            "question": question or f"ARTIFICIAL question {qid}", "label": label, "source_split": "train"}


def fixture():
    selected = [v1_data._squad_example(raw("fit-cal", "cal"), "calibration"),
                v1_data._squad_example(raw("fit-val", "val", "unanswerable"), "validation"),
                v1_data._squad_example(raw("used-dev", "development"), "development")]
    source = [raw("fit-cal", "cal"), raw("fit-val", "val", "unanswerable"),
              raw("used-dev", "development"), raw("used-dev", "unselected-but-used-article", "unanswerable")]
    source.extend(raw(title, f"{title}-{i}", label) for title in ("unused-A", "unused-B", "unused-C")
                  for i, label in enumerate(v1_data.SQUAD_LABELS))
    return source, selected


def write_fixture(root):
    source, v1 = fixture()
    holdout, holdout_gold, audit = data._build_holdout(source, v1, 1)
    fit = [row for row in v1 if row["split"] in {"calibration", "validation"}]
    fit_gold = {fit[0]["sample_id"]: "answerable", fit[1]["sample_id"]: "unanswerable"}
    examples = fit + holdout
    payloads = data._payloads(examples, fit_gold, holdout_gold)
    protocol = {"data": {"holdout_per_class": 1}, "is_test_fixture": True}
    manifest = data._manifest(examples, payloads,
                              provenance={"is_test_fixture": True, "fit_request_hashes": data.request_hashes(fit)},
                              protocol_configuration=protocol, protocol_file_hash="fixture-only",
                              audit=audit, evidence_type="software_test_fixture")
    data._publish(root, payloads, manifest)
    return examples, fit_gold, holdout_gold, payloads, manifest


def replace_fixture_payload(root, name, payload):
    """Reseal a synthetic artifact to exercise semantic checks beyond checksums."""
    (root / name).write_bytes(payload)
    manifest = json.loads((root / data.MANIFEST).read_text())
    manifest["files"][name] = data._sha(payload)
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = fingerprint(manifest)
    (root / data.MANIFEST).write_bytes(v1_data._json_bytes(manifest))


class HoldoutSelectionTests(unittest.TestCase):
    def test_balanced_deterministic_article_exclusion_includes_development(self):
        source, v1 = fixture()
        left, labels, audit = data._build_holdout(source, v1, 2)
        right, right_labels, _ = data._build_holdout(source[::-1], v1[::-1], 2)
        self.assertEqual(left, right)
        self.assertEqual(labels, right_labels)
        self.assertEqual(len(left), 4)
        self.assertEqual(list(labels.values()).count("answerable"), 2)
        self.assertEqual(list(labels.values()).count("unanswerable"), 2)
        self.assertEqual(audit["unused_article_count"], 3)
        self.assertFalse({row["group_id"] for row in left} & {row["group_id"] for row in v1})
        self.assertTrue(all(row["split"] == "holdout" for row in left))

    def test_shortage_raises_instead_of_replenishing_from_used_article(self):
        source, v1 = fixture()
        with self.assertRaisesRegex(data.DataV2Error, "Insufficient unused"):
            data._build_holdout(source, v1, 4)

    def test_duplicate_ids_and_normalized_inputs_are_rejected(self):
        source, v1 = fixture()
        with self.assertRaisesRegex(data.DataV2Error, "Duplicate question ID"):
            data._build_holdout(source + [source[-1]], v1, 1)
        duplicate = {**source[-1], "qid": "another-id"}
        with self.assertRaisesRegex(data.DataV2Error, "Repeated normalized input"):
            data._build_holdout(source + [duplicate], v1, 1)

    def test_different_title_cannot_hide_v1_text_overlap(self):
        source, v1 = fixture()
        cloned = raw("renamed-article", "hidden-copy", context="ARTIFICIAL context for used-dev")
        chosen, _, audit = data._build_holdout(source + [cloned], v1, 1)
        self.assertNotIn("squad:hidden-copy", {row["sample_id"] for row in chosen})
        self.assertEqual(audit["overlap_exclusions"]["v1_id_input_or_context_overlap"], 1)


class DataArtifactTests(unittest.TestCase):
    def test_roundtrip_fit_requests_equal_original_client_serialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, _, manifest = write_fixture(root)
            self.assertEqual(data.verify(root), manifest)
            original_fit = fixture()[1][:2]
            loaded = data.load_examples(root, "qwen_fit")
            self.assertEqual(loaded, original_fit)
            for original, copied in zip(original_fit, loaded):
                self.assertEqual(json.dumps(qwen_request(original), ensure_ascii=False).encode(),
                                 json.dumps(qwen_request(copied), ensure_ascii=False).encode())
            self.assertEqual(manifest["roles"]["qwen_fit"]["request_hashes"], data.request_hashes(original_fit))
            self.assertEqual(manifest["evidence_type"], "software_test_fixture")

    def test_fitting_loaders_never_open_sealed_gold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, fit_gold, _, _, _ = write_fixture(root)
            original_open = Path.open
            sealed = (root / data.SEALED_GOLD).resolve()
            def guarded_open(path, *args, **kwargs):
                if path.resolve() == sealed:
                    raise AssertionError("Fitting attempted to open holdout gold")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", guarded_open):
                self.assertEqual(data.load_fit_gold(root), fit_gold)
                self.assertEqual(len(data.load_examples(root, "qwen_fit")), 2)
                self.assertEqual(len(data.load_examples(root, "holdout")), 2)

    def test_changed_sealed_gold_is_detected_only_by_full_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            (root / data.SEALED_GOLD).write_text("{}\n")
            data.verify_fit(root)
            with self.assertRaisesRegex(data.DataV2Error, "checksum mismatch"):
                data.verify(root)

    def test_gold_in_inputs_rejected_even_with_new_file_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            examples, _, _, _, _ = write_fixture(root)
            examples[0]["gold"] = "answerable"
            replace_fixture_payload(root, data.INPUTS, v1_data._jsonl_bytes(examples))
            with self.assertRaisesRegex(data.DataV2Error, "Gold leaked"):
                data.verify_fit(root)

    def test_fit_gold_cannot_include_holdout_label_even_when_resealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, fit_gold, holdout_gold, _, _ = write_fixture(root)
            replace_fixture_payload(root, data.FIT_GOLD, v1_data._json_bytes({**fit_gold, **holdout_gold}))
            with self.assertRaisesRegex(data.DataV2Error, "Fit gold must cover"):
                data.verify_fit(root)

    def test_manifest_tampering_and_foreign_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, _, manifest = write_fixture(root)
            changed = copy.deepcopy(manifest)
            changed["counts"]["holdout"] = 999
            (root / data.MANIFEST).write_bytes(v1_data._json_bytes(changed))
            with self.assertRaisesRegex(data.DataV2Error, "manifest checksum"):
                data.verify(root)
            changed = copy.deepcopy(manifest)
            changed["files"]["../outside.json"] = "unused"
            changed.pop("manifest_hash")
            changed["manifest_hash"] = fingerprint(changed)
            (root / data.MANIFEST).write_bytes(v1_data._json_bytes(changed))
            with self.assertRaisesRegex(data.DataV2Error, "file set"):
                data.verify(root)

    def test_immutable_publish_is_idempotent_and_rejects_partial_conflict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, payloads, manifest = write_fixture(root)
            before = {name: (root / name).read_bytes() for name in (*data.FILES, data.MANIFEST)}
            data._publish(root, payloads, manifest)
            self.assertEqual(before, {name: (root / name).read_bytes() for name in before})
            altered = {**payloads, data.FIT_GOLD: b"{}\n"}
            with self.assertRaisesRegex(data.DataV2Error, "Existing v2 artifact differs"):
                data._publish(root, altered, manifest)
            self.assertEqual(before, {name: (root / name).read_bytes() for name in before})
            partial = root / "partial"
            (partial / "data").mkdir(parents=True)
            (partial / data.FIT_GOLD).write_bytes(b"conflicting partial file")
            with self.assertRaisesRegex(data.DataV2Error, "Existing v2 artifact differs"):
                data._publish(partial, payloads, manifest)
            self.assertFalse((partial / data.MANIFEST).exists())
            self.assertFalse((partial / data.INPUTS).exists())

    def test_prepare_refuses_to_write_inside_v1_before_reading_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            v1 = Path(temporary)
            with self.assertRaisesRegex(data.DataV2Error, "outside the immutable v1"):
                data.prepare(v1, v1, v1 / "nested")


if __name__ == "__main__":
    unittest.main()
