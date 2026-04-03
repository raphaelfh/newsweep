"""
Quality benchmarks for the newsweep completion pipeline.

Sends realistic editing scenarios to the server and measures how well
the model predicts what the developer would type next.

Metrics:
  - Exact match rate: completion == expected_completion
  - Prefix match rate: expected_completion starts with completion (or vice versa)
  - Longest common prefix length (chars)
  - Latency (p50, p95, p99) per scenario type
  - Token-level accuracy

Usage:
  python tests/bench_quality.py --host=localhost --port=8741
  python tests/bench_quality.py --scenarios=tests/fixtures/scenarios.json
  python tests/bench_quality.py --type=refactoring  # filter by scenario type
  python tests/bench_quality.py --json              # machine-readable output
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def longest_common_prefix(a: str, b: str) -> str:
    """Return the longest common prefix of two strings."""
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]


def load_scenarios(path: Path) -> list[dict]:
    """Load scenarios from a single JSON file or merge all files in a directory."""
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    elif path.is_dir():
        scenarios = []
        for p in sorted(path.glob("scenarios*.json")):
            with open(p) as f:
                scenarios.extend(json.load(f))
        return scenarios
    else:
        print(f"ERROR: {path} not found")
        sys.exit(1)


def validate_scenarios(scenarios: list[dict]) -> list[str]:
    """Quick validation that scenarios are internally consistent."""
    errors = []
    for s in scenarios:
        sid = s.get("id", "?")
        if s.get("prefix", "") + s.get("suffix", "") != s.get("file_content", ""):
            errors.append(f"Scenario {sid}: prefix+suffix != file_content")
        if s.get("cursor_position") != len(s.get("prefix", "")):
            errors.append(f"Scenario {sid}: cursor_position != len(prefix)")
        expected_after = s.get("prefix", "") + s.get("expected_completion", "") + s.get("suffix", "")
        if expected_after != s.get("file_content_after", ""):
            errors.append(f"Scenario {sid}: prefix+completion+suffix != file_content_after")
    return errors


def send_completion(client, scenario: dict, max_tokens: int = 128) -> tuple[str, float]:
    """Send a scenario to the server and return (completion_text, latency_seconds)."""
    payload = {
        "segments": {
            "prefix": scenario["prefix"],
            "suffix": scenario["suffix"],
            "filepath": scenario.get("filepath", "unknown.py"),
        },
        "file_content": scenario["file_content"],
        "cursor_position": scenario["cursor_position"],
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()
    r = client.post("/v1/completions", json=payload)
    latency = time.perf_counter() - t0

    r.raise_for_status()
    data = r.json()

    text = ""
    if data.get("choices"):
        text = data["choices"][0].get("text", "")

    return text, latency


def score_completion(actual: str, expected: str) -> dict:
    """Score a single completion against the expected text."""
    actual_stripped = actual.strip()
    expected_stripped = expected.strip()

    exact = actual_stripped == expected_stripped
    lcp = longest_common_prefix(actual_stripped, expected_stripped)
    lcp_ratio = len(lcp) / len(expected_stripped) if expected_stripped else 0.0

    # Prefix match: the model's output is a prefix of expected, or expected is a
    # prefix of the model's output (model may overshoot or undershoot).
    prefix_match = (
        expected_stripped.startswith(actual_stripped)
        or actual_stripped.startswith(expected_stripped)
    )

    return {
        "exact_match": exact,
        "prefix_match": prefix_match,
        "lcp_length": len(lcp),
        "lcp_ratio": lcp_ratio,
        "actual_length": len(actual_stripped),
        "expected_length": len(expected_stripped),
    }


def run_benchmark(
    scenarios: list[dict],
    host: str,
    port: int,
    max_tokens: int,
    type_filter: str | None = None,
) -> dict:
    """Run all scenarios and return structured results."""
    try:
        import httpx
    except ImportError:
        print("httpx required: pip install httpx")
        sys.exit(1)

    base_url = f"http://{host}:{port}"
    client = httpx.Client(base_url=base_url, timeout=60.0)

    # Check server
    try:
        r = client.get("/v1/health")
        r.raise_for_status()
        server_info = r.json()
    except Exception as e:
        print(f"ERROR: Cannot reach server at {base_url}: {e}")
        sys.exit(1)

    # Filter scenarios
    if type_filter:
        keywords = [k.strip().lower() for k in type_filter.split(",")]
        scenarios = [
            s for s in scenarios
            if s.get("scenario_type", "").lower() in keywords
            or any(k in s.get("scenario_type", "").lower() for k in keywords)
        ]

    if not scenarios:
        print("No scenarios to run (check --type filter)")
        sys.exit(1)

    results = []
    by_type: dict[str, list[dict]] = {}

    print(f"\nServer: {server_info.get('model', 'unknown')}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Max tokens: {max_tokens}")
    print()

    for i, scenario in enumerate(scenarios):
        sid = scenario.get("id", i)
        stype = scenario.get("scenario_type", "unknown")
        desc = scenario.get("description", "")

        try:
            actual, latency = send_completion(client, scenario, max_tokens)
            scores = score_completion(actual, scenario["expected_completion"])
            scores["latency_s"] = latency
            scores["id"] = sid
            scores["scenario_type"] = stype
            scores["description"] = desc
            scores["actual"] = actual.strip()
            scores["expected"] = scenario["expected_completion"].strip()
            results.append(scores)

            by_type.setdefault(stype, []).append(scores)

            # Progress
            marker = "OK" if scores["exact_match"] else f"lcp={scores['lcp_ratio']:.0%}"
            print(
                f"  [{i+1:3d}/{len(scenarios)}] {stype:<25s} "
                f"{latency*1000:6.0f}ms  {marker:<8s} {desc[:50]}"
            )

        except Exception as e:
            print(f"  [{i+1:3d}/{len(scenarios)}] {stype:<25s} ERROR: {e}")
            results.append({
                "id": sid, "scenario_type": stype, "description": desc,
                "error": str(e),
            })

    client.close()
    return _aggregate(results, by_type, server_info)


def _aggregate(
    results: list[dict],
    by_type: dict[str, list[dict]],
    server_info: dict,
) -> dict:
    """Aggregate individual results into a summary."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "All scenarios failed"}

    latencies = [r["latency_s"] for r in valid]
    exact_matches = [r for r in valid if r["exact_match"]]
    prefix_matches = [r for r in valid if r["prefix_match"]]
    lcp_ratios = [r["lcp_ratio"] for r in valid]

    summary = {
        "total_scenarios": len(results),
        "successful": len(valid),
        "errors": len(results) - len(valid),
        "exact_match_rate": len(exact_matches) / len(valid),
        "prefix_match_rate": len(prefix_matches) / len(valid),
        "lcp_ratio_mean": statistics.mean(lcp_ratios),
        "lcp_ratio_median": statistics.median(lcp_ratios),
        "latency_ms": {
            "p50": percentile(latencies, 50) * 1000,
            "p95": percentile(latencies, 95) * 1000,
            "p99": percentile(latencies, 99) * 1000,
            "mean": statistics.mean(latencies) * 1000,
        },
    }

    # Per scenario-type breakdown
    type_summary = {}
    for stype, items in sorted(by_type.items()):
        t_latencies = [r["latency_s"] for r in items]
        t_exact = [r for r in items if r["exact_match"]]
        t_prefix = [r for r in items if r["prefix_match"]]
        t_lcp = [r["lcp_ratio"] for r in items]

        type_summary[stype] = {
            "count": len(items),
            "exact_match_rate": len(t_exact) / len(items),
            "prefix_match_rate": len(t_prefix) / len(items),
            "lcp_ratio_mean": statistics.mean(t_lcp),
            "latency_ms_p50": percentile(t_latencies, 50) * 1000,
            "latency_ms_mean": statistics.mean(t_latencies) * 1000,
        }

    return {
        "server": server_info,
        "summary": summary,
        "by_type": type_summary,
        "results": results,
    }


def print_report(data: dict):
    """Print a human-readable report."""
    s = data["summary"]
    print(f"\n{'=' * 70}")
    print("Quality Benchmark Results")
    print(f"{'=' * 70}")
    print(f"  Scenarios:         {s['total_scenarios']} ({s['successful']} ok, {s['errors']} errors)")
    print(f"  Exact match:       {s['exact_match_rate']:.1%}")
    print(f"  Prefix match:      {s['prefix_match_rate']:.1%}")
    print(f"  LCP ratio (mean):  {s['lcp_ratio_mean']:.1%}")
    print(f"  LCP ratio (p50):   {s['lcp_ratio_median']:.1%}")
    lat = s["latency_ms"]
    print(f"  Latency:           p50={lat['p50']:.0f}ms  p95={lat['p95']:.0f}ms  p99={lat['p99']:.0f}ms  mean={lat['mean']:.0f}ms")

    print(f"\n{'─' * 70}")
    print(f"  {'Type':<25s} {'N':>4s} {'Exact':>7s} {'Prefix':>8s} {'LCP':>6s} {'p50ms':>7s} {'mean':>7s}")
    print(f"  {'─' * 25} {'─' * 4} {'─' * 7} {'─' * 8} {'─' * 6} {'─' * 7} {'─' * 7}")

    for stype, ts in sorted(data["by_type"].items()):
        print(
            f"  {stype:<25s} {ts['count']:4d} "
            f"{ts['exact_match_rate']:6.0%} "
            f"{ts['prefix_match_rate']:7.0%} "
            f"{ts['lcp_ratio_mean']:5.0%} "
            f"{ts['latency_ms_p50']:6.0f}ms "
            f"{ts['latency_ms_mean']:6.0f}ms"
        )

    # Show worst performers
    valid = [r for r in data["results"] if "error" not in r and not r["exact_match"]]
    if valid:
        worst = sorted(valid, key=lambda r: r["lcp_ratio"])[:5]
        print(f"\n{'─' * 70}")
        print("  Worst predictions (lowest LCP ratio):")
        for r in worst:
            print(f"    #{r['id']} [{r['scenario_type']}] lcp={r['lcp_ratio']:.0%} — {r['description'][:60]}")
            print(f"       expected: {r['expected'][:80]}...")
            print(f"       actual:   {r['actual'][:80]}...")

    print()


def main():
    parser = argparse.ArgumentParser(description="Newsweep quality benchmarks")
    parser.add_argument(
        "--scenarios",
        default="tests/fixtures",
        help="Path to scenarios JSON file or directory (default: tests/fixtures)",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8741)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--type", dest="type_filter", help="Filter by scenario type (comma-separated)")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Output JSON instead of table")
    parser.add_argument("--output", help="Save results to file")
    args = parser.parse_args()

    scenarios_path = Path(args.scenarios)
    scenarios = load_scenarios(scenarios_path)

    # Validate
    errors = validate_scenarios(scenarios)
    if errors:
        print(f"WARNING: {len(errors)} scenario validation errors:")
        for e in errors[:5]:
            print(f"  {e}")
        print()

    # Run
    data = run_benchmark(
        scenarios, args.host, args.port, args.max_tokens, args.type_filter
    )

    # Output
    if args.json_output:
        print(json.dumps(data, indent=2))
    else:
        print_report(data)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
