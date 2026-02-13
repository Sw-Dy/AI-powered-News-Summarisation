import json
import os


def load_rows(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("rows", [])


def render_bar_chart(labels, values, title, ylabel, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def metric_value(row, metric, default=0.0):
    value = row.get(metric, default)
    if value is None:
        return default
    return float(value)


def main():
    output_dir = os.path.join("visualization", "outputs")
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise RuntimeError("metrics.json not found, run collect_metrics.py first")
    rows = load_rows(metrics_path)
    if not rows:
        raise RuntimeError("metrics.json has no rows")
    labels = [row["model_id"] for row in rows]
    charts = [
        ("rouge1", "ROUGE-1 by Model", "ROUGE-1", "compare_rouge1.png"),
        ("rouge2", "ROUGE-2 by Model", "ROUGE-2", "compare_rouge2.png"),
        ("rougeL", "ROUGE-L by Model", "ROUGE-L", "compare_rougeL.png"),
        ("avg_latency_ms", "Latency by Model", "Avg Latency (ms)", "compare_latency.png"),
        ("avg_summary_length", "Summary Length by Model", "Avg Summary Words", "compare_summary_length.png"),
        ("avg_input_length_words", "Input Length by Model", "Avg Input Words", "compare_input_words.png"),
        ("avg_input_length_chars", "Input Length (Chars) by Model", "Avg Input Chars", "compare_input_chars.png"),
        ("compression_ratio", "Compression Ratio by Model", "Compression Ratio", "compare_compression_ratio.png"),
        ("throughput_input_words_per_sec", "Throughput by Model", "Words/sec", "compare_throughput.png"),
        ("avg_latency_per_input_word_ms", "Latency per Word by Model", "ms/word", "compare_latency_per_word.png"),
        ("avg_unique_word_ratio", "Unique Word Ratio by Model", "Unique Ratio", "compare_unique_ratio.png"),
        ("avg_repetition_rate", "Repetition Rate by Model", "Repetition Rate", "compare_repetition_rate.png"),
    ]
    os.makedirs(output_dir, exist_ok=True)
    for metric, title, ylabel, filename in charts:
        values = [metric_value(row, metric) for row in rows]
        output_path = os.path.join(output_dir, filename)
        render_bar_chart(labels, values, title, ylabel, output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
