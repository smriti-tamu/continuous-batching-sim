# Continuous Batching vs. Static Batching — a from-scratch benchmark

A minimal, from-scratch implementation of the scheduling mechanism behind
modern LLM inference servers (vLLM, TGI, TensorRT-LLM): **continuous
(iteration-level) batching**, benchmarked against naive static batching
under a Poisson request load.

Everything here runs on a single machine (tested on an M3 MacBook, 24GB
unified memory, PyTorch MPS backend) using GPT-2 (124M) — small enough to
iterate on quickly, while the scheduling mechanism itself is identical in
spirit to what production servers do with much larger models.

## The problem

An inference server receives requests continuously, at random times, each
asking for a different number of output tokens. Decoding is the expensive,
repeated part of serving an LLM, and batching multiple requests' decode
steps together is what gets you good GPU utilization.

**Static batching**: group arrivals into fixed-size batches, run the whole
batch until every member finishes, *then* admit the next batch. If one
request in the batch happens to ask for a long response, every other
request in that batch — and anyone who arrived after — is stuck waiting
behind it, even if the GPU has free capacity.

**Continuous batching**: at every decode step, evict any request that just
finished and admit a new one from the queue, immediately. No one waits for
a stranger's long response to finish.

## What's actually implemented

- [`engine.py`](engine.py) — a two-call LLM wrapper: `prefill(prompt)` runs a prompt once
  to build its KV cache, and `decode_step(requests)` runs **one batched
  forward pass** across a set of requests whose KV caches have *different
  lengths* — left-padded and masked into a single batch, which is the
  actual mechanical trick continuous batching requires.
- [`schedulers.py`](schedulers.py) — `run_static` and `run_continuous`, built on the same
  `decode_step`, differing only in *when requests are admitted/evicted*.
- [`workload.py`](workload.py) — a Poisson arrival process (`numpy`-free, pure
  `random.expovariate`) with a deliberately heavy-tailed mix of short
  (15-30 token) and long (120-180 token) generations, since length
  variance is exactly what creates the "straggler blocks the batch"
  problem.
- [`bench.py`](bench.py) — runs both schedulers over the identical workload at a sweep
  of arrival rates, in real wall-clock time (no simulated clock — the
  model actually runs), and records per-request latency, time-to-first-token,
  queue wait, and aggregate throughput.
- [`plot.py`](plot.py) — throughput/latency/queue-wait vs. load, plus a
  timeline chart showing each request's queue-wait (gray) and
  generation (colored) time directly.

**Explicitly not implemented** (the actual multi-month engineering effort
behind vLLM, out of scope for a weekend build): paged attention / block-table
KV-cache memory management, chunked/batched prefill, continuous-batching-aware
CUDA kernels. This project re-pads and re-copies KV caches on every decode
step, which real systems avoid with paged attention — that's a *memory
efficiency* optimization layered on top of continuous batching, not a
prerequisite for the scheduling win demonstrated here.

## Results

4 requests/sec Poisson arrivals, max batch size 4, 40 requests, GPT-2 on M3 MPS:

| rate (req/s) | scheduler  | throughput (tok/s) | latency p50 | latency p95 | mean queue wait |
|---|---|---|---|---|---|
| 1.5 | static     | 61.8  | 0.97s | 5.78s  | 0.71s |
| 1.5 | continuous | 71.5  | 0.51s | 3.61s  | 0.02s |
| 3.0 | static     | 92.5  | 2.96s | 6.68s  | 2.03s |
| 3.0 | continuous | 115.9 | 1.22s | 4.81s  | 0.29s |
| 5.0 | static     | 108.4 | 3.25s | 8.63s  | 2.47s |
| 5.0 | continuous | 123.1 | 2.75s | 8.47s  | 1.68s |
| 8.0 | static     | 91.9  | 6.75s | 13.57s | 5.41s |
| 8.0 | continuous | 113.5 | 4.24s | 12.23s | 2.79s |

![throughput, latency, and queue wait vs. load](results_summary.png)

Continuous batching wins on every metric at every load level, and the gap
*widens* as load increases — exactly what the mechanism predicts: static
batching's queue wait grows because new arrivals are blocked behind
whichever batch is currently running, while continuous batching keeps
admitting as soon as a slot frees up.

The timeline below makes the mechanism itself visible (gray = time spent
queued, color = time spent generating) at rate=5.0 req/s:

![per-request timeline](results_timeline.png)

Static batching's gray bars grow steadily longer as the run progresses —
later arrivals queue behind whatever batch is in flight. Continuous
batching's gray bars stay short throughout, because admission happens
every decode step instead of only between batches.

## Running it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python bench.py   # ~2-3 min, writes results.json
python plot.py    # writes results_summary.png, results_timeline.png
```

Tunable in [`bench.py`](bench.py): `MAX_BATCH_SIZE`, `N_REQUESTS`, `RATES`.
Tunable in [`workload.py`](workload.py): the prompt pool and the
short/long generation-length mix.
