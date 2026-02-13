import json
import os


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("rows", [])


def write_ascii_chart(output_path, rows):
    lines = []
    lines.append("Model Carbon Emissions")
    for row in rows:
        label = row["model_id"]
        carbon = row["carbon_kg"]
        bar = "#" * int(carbon * 500)
        lines.append(f"{label}: {carbon:.6f} kg {bar}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_with_matplotlib(output_path, rows):
    import matplotlib.pyplot as plt

    labels = [row["model_id"] for row in rows]
    carbon = [row["carbon_kg"] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, carbon, color="#B279A2")
    ax.set_title("Estimated Carbon Emissions by Model")
    ax.set_ylabel("kg CO2e")
    ax.tick_params(axis="x", rotation=30)
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
    png_path = os.path.join(output_dir, "carbon_emissions.png")
    txt_path = os.path.join(output_dir, "carbon_emissions.txt")
    try:
        plot_with_matplotlib(png_path, rows)
        print(f"Wrote {png_path}")
    except Exception:
        write_ascii_chart(txt_path, rows)
        print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
