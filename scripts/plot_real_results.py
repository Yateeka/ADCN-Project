import matplotlib.pyplot as plt

# Replace this with printed results from run_experiments.py
results = {
    "Vector": {"p99": 40, "qps": 400, "imbalance": 2.5},
    "Dimension": {"p99": 20, "qps": 520, "imbalance": 1.2},
    "Harmony": {"p99": 15, "qps": 680, "imbalance": 1.05}
}

systems = list(results.keys())
p99 = [results[s]["p99"] for s in systems]
qps = [results[s]["qps"] for s in systems]
imbalance = [results[s]["imbalance"] for s in systems]

# --- P99 Latency ---
plt.figure()
plt.bar(systems, p99)
plt.title("P99 Latency Comparison (Skewed Workload)")
plt.ylabel("Latency (ms)")
plt.savefig("p99_latency.png")

# --- Throughput ---
plt.figure()
plt.bar(systems, qps)
plt.title("Throughput Comparison")
plt.ylabel("QPS")
plt.savefig("throughput.png")

# --- Load Imbalance ---
plt.figure()
plt.bar(systems, imbalance)
plt.title("Load Imbalance Ratio")
plt.ylabel("Max / Avg Load")
plt.savefig("imbalance.png")

plt.show()