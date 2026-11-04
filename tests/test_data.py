import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from decision_study import data


class DataProtocolTests(unittest.TestCase):
    def row(self, qid, question="Who?", label="answerable", context="A passage.", title="Article"):
        return {"qid": qid, "question": question, "label": label, "context": context,
                "title": title, "source_split": "train"}

    def test_conflicting_visible_inputs_all_quarantined(self):
        rows = [self.row("a"), self.row("b", label="unanswerable"), self.row("c", question="What?")]
        retained, excluded = data._deduplicate_squad(rows)
        self.assertEqual([r["qid"] for r in retained], ["c"])
        self.assertEqual({r["sample_id"] for r in excluded}, {"squad:a", "squad:b"})
        self.assertTrue(all(r["reason"].startswith("conflicting") for r in excluded))

    def test_same_label_duplicate_canonical_id_input_preserved(self):
        canonical = self.row("a", question="WHO?", context="Ａ  passage.")
        rows = [self.row("z"), canonical]
        retained, excluded = data._deduplicate_squad(rows)
        self.assertEqual(retained, [canonical])
        self.assertEqual(retained[0]["context"], "Ａ  passage.")
        self.assertEqual(excluded[0]["canonical_question_id"], "a")

    def test_duplicate_question_id_is_not_silently_accepted(self):
        with self.assertRaises(data.DataError):
            data._deduplicate_squad([self.row("a"), self.row("a", question="Different?")])

    def test_context_cross_article_leakage_rejected(self):
        with self.assertRaises(data.DataError):
            data._check_squad_grouping([self.row("a"), self.row("b", title="Other")], [])

    def test_source_split_overlap_rejected(self):
        with self.assertRaises(data.DataError):
            data._check_squad_grouping([self.row("a")], [self.row("b", title="Other")])

    def test_sampling_independent_of_source_order_and_never_replenishes(self):
        rows = [{"sample_id": str(i)} for i in range(20)]
        self.assertEqual(data._sample(rows, 5, "test"), data._sample(rows[::-1], 5, "test"))
        with self.assertRaises(data.DataError):
            data._sample(rows, 21, "test")

    def test_cached_source_checksum_enforced_before_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "domains.json"
            path.write_text("{}")
            with self.assertRaisesRegex(data.DataError, "SHA256 mismatch"):
                data._source_bytes(Path(directory), "domains.json", False)

    def test_missing_cache_never_implicitly_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("urllib.request.urlopen") as request:
                with self.assertRaisesRegex(data.DataError, "Missing cached"):
                    data._source_bytes(Path(directory), "domains.json", False)
                request.assert_not_called()

    def test_squad_input_contains_no_answer_or_gold(self):
        row = self.row("a")
        example = data._squad_example(row, "calibration")
        self.assertEqual(example["labels"], ["answerable", "unanswerable"])
        self.assertNotIn("label", example)
        self.assertNotIn("gold", example)
        self.assertEqual(example["state"], "Passage:\nA passage.\n\nQuestion:\nWho?")

    def test_long_context_is_never_silently_truncated(self):
        context = "Complete evidence. " * 2000
        example = data._squad_example(self.row("long", context=context), "test")
        self.assertIn(context, example["state"])

    def test_benchmark_preserves_protocol_mix(self):
        examples, gold = [], {}
        for label in data.BANKING_LABELS:
            for i in range(60):
                identifier = f"{label}-{i}"
                examples.append({"sample_id": identifier, "task": "clinc", "split": "test"})
                gold[identifier] = label
        chosen = data.select_benchmark(examples, gold, "clinc")
        self.assertEqual(len(chosen), 200)
        self.assertEqual(sum(gold[r["sample_id"]] == "oos" for r in chosen), 50)
        self.assertEqual(chosen, data.select_benchmark(examples[::-1], gold, "clinc"))

    @unittest.skipUnless(
        all((Path(__file__).resolve().parents[1] / ".validation" / "raw" / name).is_file()
            for name in data.SOURCES),
        "Optional integration test requires previously verified official-source cache; no network download in tests.",
    )
    def test_real_preparation_reproducible_and_immutable(self):
        cache = Path(__file__).resolve().parents[1] / ".validation" / "raw"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = data.prepare_data(output, cache, download=False)
            second = data.prepare_data(output, cache, download=False)
            self.assertEqual(first, second)
            self.assertEqual(first["formal_unique_inputs"], 6250)
            self.assertEqual(first["audit"]["squad"]["selected_article_counts"]["test"], 35)
            examples = data.load_examples(output)
            self.assertEqual(len(examples), 6378)
            self.assertEqual(sum(r["split"] in ("test", "stress") for r in examples), 3450)
            with (output / "data" / "examples.jsonl").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(data.DataError, "checksum mismatch"):
                data.load_examples(output)
            with self.assertRaisesRegex(data.DataError, "Existing data artifact differs"):
                data.prepare_data(output, cache, download=False)


if __name__ == "__main__":
    unittest.main()
