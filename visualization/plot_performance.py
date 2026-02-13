import json
import os


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("rows", [])


def write_ascii_chart(output_path, rows):
    lines = []
    lines.append("Model Performance")
    for row in rows:
        label = row["model_id"]
        rouge = row["rougeL"]
        latency = row["avg_latency_ms"]
        efficiency = row.get("efficiency_score", 0.0)
        rouge_bar = "#" * int(rouge * 100)
        latency_bar = "#" * int(max(1.0, 10000.0 / (latency + 1.0)) // 10)
        efficiency_bar = "#" * int(efficiency // 2)
        lines.append(f"{label}")
        lines.append(f"  rougeL: {rouge:.3f} {rouge_bar}")
        lines.append(f"  latency_ms: {latency:.1f} {latency_bar}")
        lines.append(f"  efficiency: {efficiency:.1f} {efficiency_bar}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_with_matplotlib(output_path, rows):
    import matplotlib.pyplot as plt

    labels = [row["model_id"] for row in rows]
    rouge = [row["rougeL"] for row in rows]
    latency = [row["avg_latency_ms"] for row in rows]
    efficiency = [row.get("efficiency_score", 0.0) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    axes[0].bar(labels, rouge, color="#4C78A8")
    axes[0].set_title("ROUGE-L by Model")
    axes[0].set_ylabel("ROUGE-L")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(labels, latency, color="#F58518")
    axes[1].set_title("Latency by Model")
    axes[1].set_ylabel("Average Latency (ms)")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(labels, efficiency, color="#54A24B")
    axes[2].set_title("Efficiency Score by Model")
    axes[2].set_ylabel("Score")
    axes[2].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    output_dir = os.path.join("visualization", "outputs")
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise RuntimeError("metrics.json not found, run collect_metrics.py first")
    rows = load_metrics(metrics_path)
    if not rows:
        raise RuntimeError("metrics.json has no rows")
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "performance_comparison.png")
    txt_path = os.path.join(output_dir, "performance_comparison.txt")
    try:
        plot_with_matplotlib(png_path, rows)
        print(f"Wrote {png_path}")
    except Exception:
        write_ascii_chart(txt_path, rows)
        print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
