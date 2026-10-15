"""Run both schedulers over the same Poisson workload at a few arrival
rates and save per-request + aggregate metrics to results.json."""

import json
import sys
import time

from engine import LLM
from schedulers import run_continuous, run_static
from workload import generate_requests

MAX_BATCH_SIZE = 4
N_REQUESTS = 40
RATES = [1.5, 3.0, 5.0, 8.0]  # requests/sec
SEED = 42


def summarize(name, rate, finished, wall_clock):
    total_tokens = sum(len(r.generated) for r in finished)
    latencies = [r.finish_time - r.arrival_time for r in finished]
    ttfts = [r.first_token_time - r.arrival_time for r in finished]
    queue_waits = [r.start_time - r.arrival_time for r in finished]
    return {
        "scheduler": name,
        "rate": rate,
        "n": len(finished),
        "wall_clock_s": wall_clock,
        "throughput_tok_s": total_tokens / wall_clock,
        "latency_mean": sum(latencies) / len(latencies),
        "latency_p50": sorted(latencies)[len(latencies) // 2],
        "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)],
        "ttft_mean": sum(ttfts) / len(ttfts),
        "queue_wait_mean": sum(queue_waits) / len(queue_waits),
        "per_request": [
            {
                "id": r.id,
                "arrival": r.arrival_time,
                "start": r.start_time,
                "first_token": r.first_token_time,
                "finish": r.finish_time,
                "max_new_tokens": r.max_new_tokens,
                "tokens": len(r.generated),
            }
            for r in finished
        ],
    }


def warmup(llm):
    """Trigger MPS kernel compilation for the shapes we'll actually use,
    so the timed runs below aren't polluted by one-time JIT overhead."""
    for prompt in ["hello", "The quick brown fox jumps over the lazy dog today"]:
        for _ in range(2):
            llm.prefill(prompt)
    reqs = []
    for i in range(MAX_BATCH_SIZE):
        r = type("R", (), {})()
        r.cur_len, r.pkv, r.last_token = llm.prefill("warmup request number " + str(i))
        reqs.append(r)
    for _ in range(3):
        llm.decode_step(reqs)


def main():
    llm = LLM()
    print(f"Loaded model on device: {llm.device}")
    warmup(llm)
    print("Warmup done.")

    results = []
    for rate in RATES:
        for name, fn, kw in [
            ("static", run_static, {"batch_size": MAX_BATCH_SIZE}),
            ("continuous", run_continuous, {"max_batch_size": MAX_BATCH_SIZE}),
        ]:
            reqs = generate_requests(N_REQUESTS, rate, seed=SEED)
            t0 = time.time()
            finished = fn(llm, reqs, **kw)
            wall = time.time() - t0
            summary = summarize(name, rate, finished, wall)
            results.append(summary)
            print(
                f"rate={rate:>4} {name:>10} | wall={wall:6.2f}s "
                f"throughput={summary['throughput_tok_s']:7.1f} tok/s "
                f"latency_p50={summary['latency_p50']:5.2f}s "
                f"latency_p95={summary['latency_p95']:5.2f}s "
                f"queue_wait={summary['queue_wait_mean']:5.2f}s"
            )

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results.json")


if __name__ == "__main__":
    main()
