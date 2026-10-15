import json

import matplotlib.pyplot as plt

with open("results.json") as f:
    results = json.load(f)

by_sched = {"static": [], "continuous": []}
for r in results:
    by_sched[r["scheduler"]].append(r)
for s in by_sched:
    by_sched[s].sort(key=lambda r: r["rate"])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
colors = {"static": "#d9534f", "continuous": "#5cb85c"}

ax = axes[0]
for s, rows in by_sched.items():
    ax.plot([r["rate"] for r in rows], [r["throughput_tok_s"] for r in rows], marker="o", label=s, color=colors[s])
ax.set_xlabel("arrival rate (req/s)")
ax.set_ylabel("throughput (tok/s)")
ax.set_title("Throughput vs. load")
ax.legend()

ax = axes[1]
for s, rows in by_sched.items():
    ax.plot([r["rate"] for r in rows], [r["latency_p50"] for r in rows], marker="o", label=f"{s} p50", color=colors[s])
    ax.plot([r["rate"] for r in rows], [r["latency_p95"] for r in rows], marker="x", linestyle="--", label=f"{s} p95", color=colors[s])
ax.set_xlabel("arrival rate (req/s)")
ax.set_ylabel("end-to-end latency (s)")
ax.set_title("Latency vs. load")
ax.legend(fontsize=8)

ax = axes[2]
for s, rows in by_sched.items():
    ax.plot([r["rate"] for r in rows], [r["queue_wait_mean"] for r in rows], marker="o", label=s, color=colors[s])
ax.set_xlabel("arrival rate (req/s)")
ax.set_ylabel("mean queue wait (s)")
ax.set_title("Queueing delay vs. load")
ax.legend()

plt.tight_layout()
plt.savefig("results_summary.png", dpi=150)
print("Saved results_summary.png")

# --- Timeline / gantt chart at one rate, showing the "bubble" effect ---
TARGET_RATE = 5.0
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
for ax, sched in zip(axes, ["static", "continuous"]):
    row = next(r for r in by_sched[sched] if r["rate"] == TARGET_RATE)
    reqs = sorted(row["per_request"], key=lambda r: r["arrival"])
    for i, r in enumerate(reqs):
        ax.plot([r["arrival"], r["arrival"]], [i - 0.4, i + 0.4], color="gray", linewidth=1)  # arrival tick
        ax.barh(i, r["finish"] - r["start"], left=r["start"], height=0.6, color=colors[sched], alpha=0.8)
        ax.barh(i, r["start"] - r["arrival"], left=r["arrival"], height=0.6, color="gray", alpha=0.4)
    ax.set_title(f"{sched} batching (rate={TARGET_RATE} req/s)\ngray = waiting in queue, color = generating")
    ax.set_xlabel("time (s)")
axes[0].set_ylabel("request # (by arrival order)")
plt.tight_layout()
plt.savefig("results_timeline.png", dpi=150)
print("Saved results_timeline.png")
