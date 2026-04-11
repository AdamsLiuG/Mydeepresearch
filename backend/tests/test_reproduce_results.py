import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = BACKEND_ROOT / "src"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from evals.reproduce_results import build_output_paths


class ReproduceResultsTests(unittest.TestCase):
    def test_build_output_paths_matches_expected_legacy_results_layout(self):
        paths = build_output_paths(
            results_dir=Path("/tmp/legacy-results"),
            perf_results_dir=Path("/tmp/perf-results"),
            profile="real_local",
            tag="20260329",
        )

        self.assertEqual(
            paths.http_results_json.name,
            "full_system_http_results_real_local_20260329.json",
        )
        self.assertEqual(
            paths.interview_summary_md.name,
            "full_system_interview_summary_real_local_20260329.md",
        )
        self.assertEqual(
            paths.metrics_report_json.name,
            "project_metrics_report_real_local_20260329.json",
        )
        self.assertEqual(paths.smoke_json.name, "smoke_real_local_20260329.json")
        self.assertEqual(paths.regression_json.name, "regression_real_local_20260329.json")
        self.assertEqual(paths.load_json.name, "load_real_local_20260329.json")


if __name__ == "__main__":
    unittest.main()
