# Benchmark Results

Date (UTC): 2026-02-22

Mode: `synthetic` (deterministic fallback mode)

Command used:

```bash
python3.11 scripts/benchmark.py --synthetic --requests 600 --concurrency 60 --stream --max-tokens 64 --synthetic-failure-rate 0 --out artifacts/benchmark/latest.json
```

Result summary:

| Metric | Value |
|---|---:|
| Requests total | 600 |
| Success / Failed | 600 / 0 |
| Elapsed seconds | 1.4454 |
| Throughput (req/s) | 415.1138 |
| Token throughput (tok/s) | 26567.2859 |
| Mean latency (s) | 0.144539 |
| p50 latency (s) | 0.138976 |
| p95 latency (s) | 0.206259 |
| p99 latency (s) | 0.342013 |
| Max latency (s) | 0.445071 |

Notes:
- This sandbox does not have Docker or Redis available, so these are synthetic benchmark results from the built-in deterministic fallback mode.
- For full end-to-end benchmark results, run `make run` and then `make benchmark` on a machine with Docker.
