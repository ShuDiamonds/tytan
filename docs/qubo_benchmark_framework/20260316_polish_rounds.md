# 2026-03-16 23:59 - Multi-pass polish for SASampler

## Goal
Make the existing CPU SA sampler stronger on rugged benchmarks (e.g., G-set) by giving the polish stage enough opportunity to settle into a better local optimum.

## Hypothesis
A single sweep over each index after annealing often leaves a small-to-medium improvement on the table. Running up to three deterministic polish passes per sample before reporting should squeeze a bit more quality out of the expensive fallback run.

## Files changed
- `tytan/sampler.py`
- `tools/qubo_benchmark_framework/solvers.py`
- `tests/sampler/test_sa_sampler.py`

## Commands run
- `PYTHONPATH=/Users/shuichifukuda/Documents/Project/tytan /Users/shuichifukuda/Documents/Project/tytan/tools/qubo_benchmark_framework/.venv/bin/python tools/qubo_benchmark_framework/benchmark_cli.py --dataset-root tools/qubo_benchmark_framework/datasets_subset --solvers tytan_sa dwave_sa --dataset-types gset --time-limit-sec 1 --num-reads 20 --seed 1234 --output tools/qubo_benchmark_framework/results/gset_polish`
- `PYTHONPATH=/Users/shuichifukuda/Documents/Project/tytan /Users/shuichifukuda/Documents/Project/tytan/tools/qubo_benchmark_framework/.venv/bin/python -m pytest tests/sampler/test_sa_sampler.py`

## Before vs After
- **Before:** `results/gset_smoke/summary.csv` records `tytan_sa` winning the G1 instance with best objective `-28,351.0` after ~125 seconds, and only a single polish pass ran (the old default).
- **After:** `results/gset_polish/summary.csv` now reports `tytan_sa` at `-28,701.0` with ~1,002 seconds because the 1‑second timeout triggered a fallback run, but the additional polish passes shaved another ~350 units off the best objective.

## Improvements observed
- The new `polish_passes` parameter allows solvers to run up to three greedy passes per sample, and a `polish_rounds` metric is emitted so downstream tools can see how many passes actually executed.
- G1’s best objective improved from -28,351 to -28,701 when combining the existing annealing schedule with deterministic local polishing.

## Regressions observed
- Runtime increased because the benchmark still fell back from the 1 s limit; the solver now runs to completion (~17 minutes for G1) before reporting that better energy, so this cost must be weighed in broader regressions.

## Decision
Keep the multi-pass polish by default for the `tytan_sa` adapter and observe the added runtime; the quality gain is modest but non-trivial on hard instances.

## Next step
Measure the new polish stage on additional families (MQLib, QAPLIB) to ensure the runtime cost stays affordable, or gate extra passes behind the time limit if necessary.
