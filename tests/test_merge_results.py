"""Regression tests for the normal merge CLI (no training dependencies)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_results.py"
RESULT = "slice_eval_TO_result.json"


def entry(seed, score, epoch=None):
    metrics = {"ndcg@10": score, "recall@10": score}
    report = {"overall": metrics, "long_tail": {"recall@10": score, "recall@30": score},
              "coverage@10": score, "novelty_bits": score, "fit": {"best_epoch": epoch}}
    return {"run_signature": "v1", "run_context": {"split": "TO"},
            "seeds": {seed: report}, "agg": {"obsolete": True}}


class TestMerge(unittest.TestCase):
    def merge(self, existing, incoming):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "out"
            source = Path(folder) / "source"
            out.mkdir()
            source.mkdir()
            (out / RESULT).write_text(json.dumps(existing), encoding="utf-8")
            (source / RESULT).write_text(json.dumps(incoming), encoding="utf-8")
            proc = subprocess.run([sys.executable, str(SCRIPT), "--split", "TO", "--inputs",
                                   str(source), "--out", str(out)], capture_output=True)
            result = json.loads((out / RESULT).read_text(encoding="utf-8"))
            return proc.returncode, result

    def test_add_seed_and_refresh_untouched_model(self):
        code, result = self.merge({"A": entry("7", 0.25, 2), "B": entry("9", 0.1)},
                                  {"A": entry("8", 0.75, 5)})
        self.assertEqual(code, 0)
        self.assertEqual(result["A"]["agg"]["overall.ndcg@10"],
                         {"seeds": [0.25, 0.75], "mean": 0.5, "std": 0.25})
        self.assertEqual(result["A"]["agg"]["best_epoch"], {"seeds": [2, 5], "mean": 3.5})
        self.assertEqual(result["B"]["agg"]["overall.ndcg@10"]["seeds"], [0.1])

    def test_new_model(self):
        new = entry("3", 0.123456)
        code, result = self.merge({"A": entry("7", 0.25)}, {"B": new})
        self.assertEqual(code, 0)
        agg = result["B"].pop("agg")
        self.assertEqual(result["B"], {k: v for k, v in new.items() if k != "agg"})
        self.assertEqual(set(agg), {"overall.ndcg@10", "overall.recall@10", "long_tail.recall@10",
                                    "long_tail.recall@30", "coverage@10", "novelty_bits"})
        for metric in agg.values():
            self.assertEqual(metric, {"seeds": [0.123456], "mean": 0.1235, "std": 0.0})

    def test_duplicate_seed(self):
        first = entry("1", 0.2)
        for score, status in [(0.2, 0), (0.8, 1)]:
            with self.subTest(score=score):
                code, result = self.merge({"A": first}, {"A": entry("1", score)})
                self.assertEqual(code, status)
                self.assertEqual(result["A"]["seeds"], first["seeds"])
                self.assertEqual(result["A"]["agg"]["overall.ndcg@10"]["seeds"], [0.2])
