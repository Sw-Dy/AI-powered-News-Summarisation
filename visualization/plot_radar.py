import json
import math
import os


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("rows", [])


def normalize(values, higher_is_better=True):
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [1.0 for _ in values]
    normalized = []
    for value in values:
        score = (value - minimum) / (maximum - minimum)
        if not higher_is_better:
            score = 1.0 - score
        normalized.append(score)
    return normalized


def plot_with_matplotlib(output_path, rows):
    import matplotlib.pyplot as plt

    labels = [row["model_id"] for row in rows]
    rouge_values = [row["rougeL"] for row in rows]
    latency_values = [row["avg_latency_ms"] for row in rows]
    energy_values = [row["energy_kwh"] for row in rows]
    error_values = [row["errors"] for row in rows]

    rouge_norm = normalize(rouge_values, higher_is_better=True)
    latency_norm = normalize(latency_values, higher_is_better=False)
    energy_norm = normalize(energy_values, higher_is_better=False)
    error_norm = normalize(error_values, higher_is_better=False)

    categories = ["ROUGE-L", "Latency", "Energy", "Errors"]
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)

    for idx, label in enumerate(labels):
        values = [rouge_norm[idx], latency_norm[idx], energy_norm[idx], error_norm[idx]]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]], categories)
    ax.set_ylim(0, 1)
    ax.set_title("Efficiency Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_ascii_chart(output_path, rows):
    lines = []
    lines.append("Efficiency Radar Metrics")
    for row in rows:
        lines.append(
            f'{row["model_id"]} rougeL={row["rougeL"]:.3f} latency={row["avg_latency_ms"]:.1f} energy={row["energy_kwh"]:.6f} errors={row["errors"]}'
        )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    output_dir = os.path.join("visualization", "outputs")
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise RuntimeError("metrics.json not found, run collect_metrics.py first")
    rows = load_metrics(metrics_path)
    if not rows:
        raise RuntimeError("metrics.json has no rows")
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "efficiency_radar.png")
    txt_path = os.path.join(output_dir, "efficiency_radar.txt")
    try:
        plot_with_matplotlib(png_path, rows)
        print(f"Wrote {png_path}")
    except Exception:
        write_ascii_chart(txt_path, rows)
        print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
