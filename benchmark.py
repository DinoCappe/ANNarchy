"""
Local benchmark script — mirrors what the competition harness measures.
Runs all scenarios from scenarios.yaml and reports all 5 prize metrics.

Usage:
    python benchmark.py datasets/yahoo-minilm-public.hdf5
"""

import sys
import time
import resource
import h5py
import numpy as np
import yaml

sys.path.insert(0, "app")
from algorithm import Algorithm

SCENARIOS_FILE = "app/scenarios.yaml"


def load_scenarios():
    with open(SCENARIOS_FILE) as f:
        return yaml.safe_load(f)["scenarios"]


def measure_peak_mb(fn):
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    fn()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (after - before) / 1024  # KB -> MB


def run_scenario(name, cfg, train, queries, ground, k=100):
    index_params = cfg.get("index_params", {})
    query_params = cfg.get("query_params", {})

    alg = Algorithm()

    # Build
    t0 = time.perf_counter()
    alg.fit(train, **index_params)
    build_time = time.perf_counter() - t0

    # Memory delta from fit
    alg2 = Algorithm()
    mem_mb = measure_peak_mb(lambda: alg2.fit(train, **index_params))

    # Queries
    n_queries = len(queries)
    results = []
    query_times = []
    for i in range(n_queries):
        t0 = time.perf_counter()
        r = alg.query(queries[i], k, **query_params)
        query_times.append(time.perf_counter() - t0)
        results.append(r)

    total_query_time = sum(query_times)
    n_distances = alg.get_n_distances()

    # Recall
    recalls = []
    for i in range(n_queries):
        gt = set(ground[i])
        hits = sum(1 for idx in results[i] if idx in gt)
        recalls.append(hits / k)
    mean_recall = np.mean(recalls)

    return {
        "scenario":        name,
        "index_params":    index_params,
        "query_params":    query_params,
        "recall":          mean_recall,
        "build_time_s":    build_time,
        "query_time_s":    total_query_time,
        "qps":             n_queries / total_query_time,
        "mem_mb":          mem_mb,
        "n_distances":     n_distances,
        "median_ms":       np.median(query_times) * 1e3,
        "p99_ms":          np.percentile(query_times, 99) * 1e3,
    }


def print_results(results):
    prizes = {
        "high_recall": ("Sherlock Holmes", 0.95),
        "fast":        ("Bianconiglio",    0.80),
        "memory":      ("Dory/Paperone",   0.95),
    }

    header = f"{'Scenario':<14} {'Recall':>7} {'Build(s)':>9} {'Query(s)':>9} {'QPS':>7} {'Mem(MB)':>9} {'Distances':>14} {'Median(ms)':>11} {'p99(ms)':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for r in results:
        name = r["scenario"]
        prize, threshold = prizes.get(name, ("?", 0.0))
        ok = "✓" if r["recall"] >= threshold else "✗"
        print(
            f"{name:<14} {r['recall']:>7.4f}{ok} {r['build_time_s']:>9.2f} "
            f"{r['query_time_s']:>9.2f} {r['qps']:>7.1f} {r['mem_mb']:>9.1f} "
            f"{r['n_distances']:>14,} {r['median_ms']:>11.2f} {r['p99_ms']:>9.2f}"
        )

    print("=" * len(header))
    print(f"  ✓/✗ = meets/misses recall threshold for that scenario's prize")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <dataset.hdf5>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    dataset_name = dataset_path.split("/")[-1].replace(".hdf5", "")

    print(f"Loading {dataset_path}...")
    with h5py.File(dataset_path) as f:
        train   = f["/train"][:]
        queries = f["/test"][:]
        ground  = f["/neighbors"][:]

    print(f"  {train.shape[0]:,} train | {queries.shape[0]} queries | dim={train.shape[1]}")

    scenarios = load_scenarios()
    results = []

    for scenario_name, scenario_cfg in scenarios.items():
        cfg = scenario_cfg.get(dataset_name) or scenario_cfg.get("default")
        if cfg is None:
            print(f"  [skip] no config for scenario '{scenario_name}'")
            continue
        print(f"\nRunning scenario: {scenario_name}  params={cfg}")
        r = run_scenario(scenario_name, cfg, train, queries, ground)
        results.append(r)

    print_results(results)
