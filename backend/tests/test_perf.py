import csv
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from perf.common import build_summary, compare_against_baseline, percentile
from perf.runtime import parse_locust_stats_csv


class PerfUtilityTests(unittest.TestCase):
    def test_percentile_interpolates_expected_values(self):
        values = [10.0, 20.0, 30.0, 40.0]
        self.assertEqual(percentile(values, 50), 25.0)
        self.assertEqual(percentile(values, 95), 38.5)
        self.assertEqual(percentile(values, 99), 39.7)

    def test_build_summary_reports_expected_metrics(self):
        request_records = [
            {"latency_ms": 100.0, "success": True},
            {"latency_ms": 200.0, "success": True},
            {"latency_ms": 500.0, "success": False},
        ]
        summary = build_summary(
            mode="smoke",
            profile="stub",
            endpoint="/research",
            concurrency=1,
            request_records=request_records,
            total_duration_s=3.0,
            cpu_samples=[10.0, 20.0, 30.0],
            rss_samples=[100.0, 110.0, 120.0],
            estimated_cost_total=0.003,
        )

        self.assertEqual(summary["requests_total"], 3)
        self.assertEqual(summary["success_rate"], 0.6667)
        self.assertEqual(summary["error_rate"], 0.3333)
        self.assertEqual(summary["avg_latency_ms"], 266.67)
        self.assertEqual(summary["p95_latency_ms"], 470.0)
        self.assertEqual(summary["cpu_percent_peak"], 30.0)
        self.assertEqual(summary["rss_mb_peak"], 120.0)
        self.assertEqual(summary["estimated_cost_per_request"], 0.001)

    def test_compare_against_baseline_detects_violations(self):
        summary = {
            "success_rate": 1.0,
            "error_rate": 0.0,
            "p95_latency_ms": 1200.0,
            "p99_latency_ms": 1800.0,
            "rps": 0.5,
            "rss_mb_peak": 800.0,
        }
        baseline = {
            "profile": "stub",
            "baseline_ready": True,
            "summary": {
                "p95_latency_ms": 400.0,
                "p99_latency_ms": 700.0,
                "rps": 1.0,
                "rss_mb_peak": 200.0,
            },
            "thresholds": {
                "smoke": {
                    "success_rate": {"min_absolute": 1.0},
                    "error_rate": {"max_absolute": 0.0},
                    "p95_latency_ms": {"max_absolute": 1000.0},
                    "rps": {"min_absolute": 0.1}
                },
                "regression": {
                    "rps": {"min_ratio": 0.8},
                    "p95_latency_ms": {"max_ratio": 2.0},
                    "rss_mb_peak": {"max_ratio": 2.0},
                },
            },
        }

        smoke = compare_against_baseline(summary=summary, baseline_payload=baseline, mode="smoke")
        regression = compare_against_baseline(summary=summary, baseline_payload=baseline, mode="regression")

        self.assertFalse(smoke["passed"])
        self.assertIn("p95_latency_ms", smoke["failed_metrics"])
        self.assertFalse(regression["passed"])
        self.assertIn("rps", regression["failed_metrics"])
        self.assertIn("rss_mb_peak", regression["failed_metrics"])

    def test_parse_locust_stats_csv_reads_aggregated_row(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "locust_stats.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Type",
                        "Name",
                        "Request Count",
                        "Failure Count",
                        "Median Response Time",
                        "Average Response Time",
                        "Requests/s",
                        "50%",
                        "95%",
                        "99%",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Type": "",
                        "Name": "Aggregated",
                        "Request Count": "12",
                        "Failure Count": "1",
                        "Median Response Time": "90",
                        "Average Response Time": "120",
                        "Requests/s": "3.5",
                        "50%": "100",
                        "95%": "220",
                        "99%": "350",
                    }
                )

            stats = parse_locust_stats_csv(path)

        self.assertEqual(stats["requests_total"], 12)
        self.assertEqual(stats["failure_total"], 1)
        self.assertEqual(stats["success_rate"], 0.9167)
        self.assertEqual(stats["p95_latency_ms"], 220.0)


if __name__ == "__main__":
    unittest.main()
