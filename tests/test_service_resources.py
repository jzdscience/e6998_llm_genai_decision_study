"""Artificial process inventories verify resource provenance without a GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_study.service_study import resources


def process(pid, ppid, start, user=5, system=2, rss=3):
    return {"pid": pid, "ppid": ppid, "start_time_ticks": start,
            "command": "artificial", "state": "S", "user_cpu_ticks": user,
            "system_cpu_ticks": system, "rss_pages": rss, "virtual_bytes": 10000}


class ResourceIdentityTests(unittest.TestCase):
    def test_proc_stat_parser_preserves_kernel_units_and_command_parentheses(self):
        # /proc fields3..24; command includes ')' and whitespace by construction.
        tail = ["S", "9"] + ["0"] * 9 + ["13", "7"] + ["0"] * 6 + ["1234", "99999", "11"]
        row = resources._parse_stat("42 (name with ) inside) " + " ".join(tail))
        self.assertEqual(row["pid"], 42)
        self.assertEqual(row["command"], "name with ) inside")
        self.assertEqual(row["ppid"], 9)
        self.assertEqual(row["user_cpu_ticks"], 13)
        self.assertEqual(row["system_cpu_ticks"], 7)
        self.assertEqual(row["start_time_ticks"], 1234)
        self.assertEqual(row["virtual_bytes"], 99999)
        self.assertEqual(row["rss_pages"], 11)

    def test_descendants_remain_bound_after_reparenting_but_reused_root_does_not(self):
        first = {100: process(100, 1, 10), 101: process(101, 100, 11),
                 102: process(102, 101, 12), 999: process(999, 1, 99)}
        second = {100: process(100, 1, 90), 101: process(101, 1, 11, user=8),
                  102: process(102, 101, 12, user=9), 103: process(103, 100, 91)}
        unavailable_gpu = {"status": "unavailable", "gpus": None, "error": "FileNotFoundError"}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(resources, "_scan_processes", side_effect=[(first, []), (second, []), ({}, ["unavailable"])]), \
                patch.object(resources, "_gpu_sample", return_value=unavailable_gpu):
            path = Path(temporary) / "resources.jsonl"
            with resources.ResourceMonitor([100], path) as monitor:
                monitor._sample()
                monitor._sample()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([p["pid"] for p in rows[0]["processes"]], [100, 101, 102])
        self.assertEqual([p["pid"] for p in rows[1]["processes"]], [101, 102])
        self.assertEqual(rows[1]["unavailable_roots"], [{"pid": 100, "reason": "pid_reused"}])
        self.assertEqual(rows[2]["processes"], [])
        self.assertEqual(rows[2]["unavailable_roots"], [{"pid": 100, "reason": "missing_or_unreadable"}])
        self.assertIsNone(rows[2]["gpu"]["gpus"])
        self.assertNotIn("cpu_seconds", rows[2])
        self.assertEqual([row["sample_index"] for row in rows], [0, 1, 2])

    def test_sampler_refuses_to_overwrite_old_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resources.jsonl"
            path.write_text("preserve\n")
            with self.assertRaises(FileExistsError):
                with resources.ResourceMonitor([100], path):
                    pass
            self.assertEqual(path.read_text(), "preserve\n")

    def test_hardware_reports_limit_claims_without_asserting_exclusivity(self):
        record = resources.hardware()
        self.assertIn("clock_ticks_per_second", record)
        self.assertIn("page_size_bytes", record)
        self.assertIn("controller_affinity", record)
        self.assertIn("slurm", record)
        self.assertIn("do not establish CPU exclusivity", record["note"])


if __name__ == "__main__":
    unittest.main()
