import numpy as np
import matplotlib.pyplot as plt

from vector_baseline import run_cluster_partition, load_data
from dimension_baseline import run_dimension_baseline
from harmony_pipeline import query_pipeline


def compute_metrics(latencies, loads):
    lat = np.array(latencies)

    mean = lat.mean() * 1000
    p95 = np.percentile(lat, 95) * 1000
    p99 = np.percentile(lat, 99) * 1000

    total_time = lat.sum()
    qps = len(lat) / total_time

    loads = loads.astype(float)
    imbalance = loads.max() / loads.mean()

    return {
        "mean": mean,
        "p95": p95,
        "p99": p99,
        "qps": qps,
        "imbalance": imbalance
    }


def run_all():
    print("Loading data...")
    base, queries = load_data("skewed")

    print("Running Vector baseline...")
    vec_out = run_cluster_partition(
        base, queries, num_shards=4, top_k=10, nprobe=2
    )
    vec_metrics = compute_metrics(
        vec_out["latencies"], vec_out["shard_loads"]
    )

    print("Running Dimension baseline...")
    dim_out = run_dimension_baseline(
        base, queries, top_k=10, num_blocks=4
    )
    dim_metrics = compute_metrics(
        dim_out["latencies"], dim_out["block_loads"]
    )

    print("Running Harmony pipeline...")
    harm_out = query_pipeline(
        queries=queries,
        base=base,
        num_partitions=4,
        nprobe=2,
        num_blocks=4,
        top_k=10,
        warmup_queries=32,
        warmup_vectors=1000,
    )

    # IMPORTANT: use partition_compute_loads
    harm_metrics = compute_metrics(
        harm_out["latencies"],
        harm_out["partition_compute_loads"]
    )

    return {
        "Vector": vec_metrics,
        "Dimension": dim_metrics,
        "Harmony": harm_metrics
    }


def plot(results):
    systems = list(results.keys())

    p99 = [results[s]["p99"] for s in systems]
    qps = [results[s]["qps"] for s in systems]
    imbalance = [results[s]["imbalance"] for s in systems]

    plt.style.use("ggplot")

    # P99 Latency
    plt.figure()
    plt.bar(systems, p99)
    plt.title("P99 Latency under Skew")
    plt.ylabel("Latency (ms)")
    plt.savefig("p99_latency.png")

    # Throughput
    plt.figure()
    plt.bar(systems, qps)
    plt.title("Throughput (QPS)")
    plt.ylabel("QPS")
    plt.savefig("throughput.png")

    # Load Imbalance
    plt.figure()
    plt.bar(systems, imbalance)
    plt.title("Load Imbalance Ratio")
    plt.ylabel("Max / Avg Load")
    plt.savefig("imbalance.png")

    # BONUS: imbalance vs latency
    plt.figure()
    plt.scatter(imbalance, p99)

    for i, name in enumerate(systems):
        plt.text(imbalance[i], p99[i], name)

    plt.xlabel("Load Imbalance")
    plt.ylabel("P99 Latency")
    plt.title("Impact of Load Imbalance on Latency")
    plt.savefig("imbalance_vs_latency.png")

    plt.show()


if __name__ == "__main__":
    results = run_all()

    print("\n=== FINAL RESULTS ===")
    for k, v in results.items():
        print(k, v)

    plot(results)