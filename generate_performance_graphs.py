import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from math import pi
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
METRICS_DIR = BASE_DIR / "metrics"
SUMMARY_PATH = BASE_DIR / "processed_output" / "processing_summary.json"
OUTPUT_DIR = BASE_DIR / "performance_graphs"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def save_fig(filename: str):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()


def normalize_query_rows(rows):
    normalized = []
    for r in rows:
        row = dict(r)
        total = row.get("total_query_duration_seconds")
        if total is None:
            total = row.get("total_latency_seconds", 0.0)
        row["total_query_duration_seconds"] = float(total or 0.0)
        row["total_latency_seconds"] = float(total or 0.0)

        gen = row.get("generation_duration_seconds")
        if gen is None:
            gen = row.get("generation_latency_seconds", 0.0)
        row["generation_duration_seconds"] = float(gen or 0.0)
        row["generation_latency_seconds"] = float(gen or 0.0)

        ret = row.get("retrieval_duration_seconds")
        if ret is None:
            ret = row.get("retrieval_latency_seconds", 0.0)
        row["retrieval_duration_seconds"] = float(ret or 0.0)
        row["retrieval_latency_seconds"] = float(ret or 0.0)

        row["retrieved_results"] = int(row.get("retrieved_results", 0) or 0)
        row["answer_length_chars"] = int(row.get("answer_length_chars", 0) or 0)
        row["mode"] = str(row.get("mode", "unknown"))
        normalized.append(row)
    return normalized


def plot_query_latency_timeline(query_rows):
    ordered = sorted(query_rows, key=lambda row: parse_time(row["timestamp"]))
    x = np.arange(1, len(ordered) + 1)
    total = [float(row["total_query_duration_seconds"]) for row in ordered]
    generation = [float(row["generation_duration_seconds"]) for row in ordered]
    retrieval = [float(row["retrieval_duration_seconds"]) for row in ordered]

    plt.figure(figsize=(10, 5.8))
    plt.plot(x, total, marker="o", linewidth=2.4, color="#c44e52", label="Total time")
    plt.plot(x, generation, marker="s", linewidth=2.1, color="#4c72b0", label="Generation time")
    plt.plot(x, retrieval, marker="^", linewidth=2.1, color="#55a868", label="Retrieval time")
    if len(x) > 30:
        step = max(1, len(x) // 15)
        indices = list(range(0, len(x), step))
        if indices[-1] != len(x) - 1:
            indices.append(len(x) - 1)
        plt.xticks([x[i] for i in indices], [f"Q{x[i]}" for i in indices], rotation=30)
    else:
        plt.xticks(x, [f"Q{i}" for i in x])
    plt.ylabel("Seconds")
    plt.xlabel("Query order")
    plt.title("Query Latency Timeline")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(frameon=False)
    save_fig("01_query_latency_timeline.png")


def plot_mode_distribution(query_rows):
    modes = [row["mode"] for row in query_rows]
    counts = Counter(modes)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    plt.figure(figsize=(7.5, 5.6))
    plt.bar(labels, values, color=colors[: len(labels)], width=0.55)
    for idx, value in enumerate(values):
        plt.text(idx, value + 0.05, str(value), ha="center", va="bottom", fontsize=11)
    plt.ylabel("Number of queries")
    plt.title("Query Mode Distribution")
    plt.grid(axis="y", alpha=0.2, linestyle="--")
    save_fig("02_query_mode_distribution.png")


def plot_response_length_vs_time(query_rows):
    durations = [float(row["total_query_duration_seconds"]) for row in query_rows]
    answer_lengths = [int(row["answer_length_chars"]) for row in query_rows]
    retrieved = [int(row["retrieved_results"]) for row in query_rows]
    labels = [f"Q{i}" for i in range(1, len(query_rows) + 1)]

    plt.figure(figsize=(8.5, 6.2))
    scatter = plt.scatter(
        durations,
        answer_lengths,
        s=[120 + (r * 120) for r in retrieved],
        c=retrieved,
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
    )
    step = 1 if len(query_rows) <= 25 else max(1, len(query_rows) // 20)
    for idx, (x_val, y_val, label) in enumerate(zip(durations, answer_lengths, labels)):
        if idx % step == 0 or idx == len(query_rows) - 1:
            plt.text(x_val + 0.25, y_val + 8, label, fontsize=9)
    plt.xlabel("Total query duration (s)")
    plt.ylabel("Answer length (chars)")
    plt.title("Response Length vs Query Time")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Retrieved results")
    plt.grid(alpha=0.25, linestyle="--")
    save_fig("03_response_length_vs_time.png")


def plot_chunk_distribution(summary):
    labels = list(summary.get("by_file_type", {}).keys())
    values = list(summary.get("by_file_type", {}).values())
    if not labels or sum(values) == 0:
        labels = ["text"]
        values = [summary.get("total_chunks", 1)]

    plt.figure(figsize=(7.2, 6.2))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#8172b3", "#937860"]
    wedges, _ = plt.pie(
        values,
        labels=None,
        colors=colors[: len(values)],
        startangle=90,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
    )
    plt.legend(
        wedges,
        [f"{label}: {value}" for label, value in zip(labels, values)],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    plt.title("Processed Chunk Distribution by File Type")
    save_fig("04_chunk_distribution_donut.png")


def plot_query_heatmap(query_rows):
    ordered = sorted(query_rows, key=lambda row: parse_time(row["timestamp"]))
    matrix = np.array(
        [
            [
                float(row["retrieval_duration_seconds"]),
                float(row["generation_duration_seconds"]),
                float(row["total_query_duration_seconds"]),
                int(row["retrieved_results"]),
                int(row["answer_length_chars"]),
            ]
            for row in ordered
        ],
        dtype=float,
    )

    normalized = matrix.copy()
    for col in range(normalized.shape[1]):
        column = normalized[:, col]
        max_value = column.max() if column.max() != 0 else 1.0
        normalized[:, col] = column / max_value

    plt.figure(figsize=(9.2, 5.4))
    plt.imshow(normalized, aspect="auto", cmap="magma")
    plt.colorbar(label="Normalized intensity")
    if len(ordered) > 30:
        step = max(1, len(ordered) // 15)
        indices = list(range(0, len(ordered), step))
        if indices[-1] != len(ordered) - 1:
            indices.append(len(ordered) - 1)
        plt.yticks(indices, [f"Q{i+1}" for i in indices])
    else:
        plt.yticks(np.arange(len(ordered)), [f"Q{i}" for i in range(1, len(ordered) + 1)])
    plt.xticks(
        np.arange(5),
        ["Retrieval", "Generation", "Total", "Results", "Answer size"],
        rotation=15,
    )
    plt.title("Query Performance Heatmap")
    save_fig("05_query_performance_heatmap.png")


def plot_mode_radar(query_rows):
    grouped = defaultdict(list)
    for row in query_rows:
        grouped[row["mode"]].append(row)

    metrics = ["retrieval", "generation", "total", "results", "answer_size"]
    labels = ["Retrieval", "Generation", "Total", "Results", "Answer Size"]

    aggregated = {}
    for mode, rows in grouped.items():
        aggregated[mode] = np.array(
            [
                np.mean([float(r["retrieval_duration_seconds"]) for r in rows]),
                np.mean([float(r["generation_duration_seconds"]) for r in rows]),
                np.mean([float(r["total_query_duration_seconds"]) for r in rows]),
                np.mean([int(r["retrieved_results"]) for r in rows]),
                np.mean([int(r["answer_length_chars"]) for r in rows]),
            ],
            dtype=float,
        )

    stacked = np.array(list(aggregated.values()), dtype=float)
    max_values = stacked.max(axis=0)
    max_values[max_values == 0] = 1.0

    angles = np.linspace(0, 2 * pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.4, 7.0))
    ax = plt.subplot(111, polar=True)
    palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    for idx, (mode, values) in enumerate(aggregated.items()):
        scaled = (values / max_values).tolist()
        scaled += scaled[:1]
        ax.plot(angles, scaled, linewidth=2.2, label=mode, color=palette[idx % len(palette)])
        ax.fill(angles, scaled, alpha=0.18, color=palette[idx % len(palette)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Mode Comparison Radar", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.24, 1.10), frameon=False)
    save_fig("06_mode_comparison_radar.png")


def write_summary(query_rows, upload_rows, summary):
    rag_queries = [row for row in query_rows if row["mode"] == "rag"]
    avg_total = np.mean([float(row["total_query_duration_seconds"]) for row in query_rows]) if query_rows else 0.0
    avg_generation = np.mean([float(row["generation_duration_seconds"]) for row in query_rows]) if query_rows else 0.0
    avg_answer_len = np.mean([int(row["answer_length_chars"]) for row in query_rows]) if query_rows else 0.0

    report_lines = [
        "Performance Graph Pack Summary",
        "=" * 32,
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Total queries analyzed: {len(query_rows)}",
        f"RAG queries analyzed: {len(rag_queries)}",
        f"Upload sessions analyzed: {len(upload_rows)}",
        f"Average total query time: {avg_total:.3f}s",
        f"Average generation time: {avg_generation:.3f}s",
        f"Average answer length: {avg_answer_len:.1f} chars",
        f"Latest processed chunks: {summary.get('total_chunks', 0)}",
        f"Latest processed tokens: {summary.get('total_tokens', 0)}",
        "",
        "Generated files:",
        "01_query_latency_timeline.png",
        "02_query_mode_distribution.png",
        "03_response_length_vs_time.png",
        "04_chunk_distribution_donut.png",
        "05_query_performance_heatmap.png",
        "06_mode_comparison_radar.png",
    ]
    (OUTPUT_DIR / "README.txt").write_text("\n".join(report_lines), encoding="utf-8")


def main():
    ensure_output_dir()

    query_rows = normalize_query_rows(load_jsonl(METRICS_DIR / "query_metrics.jsonl"))
    upload_rows = load_jsonl(METRICS_DIR / "upload_metrics.jsonl")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else {}

    if not query_rows:
        raise SystemExit("No query metrics found. Run a few queries first, then generate graphs.")

    plt.style.use("seaborn-v0_8-whitegrid")

    plot_query_latency_timeline(query_rows)
    plot_mode_distribution(query_rows)
    plot_response_length_vs_time(query_rows)
    plot_chunk_distribution(summary)
    plot_query_heatmap(query_rows)
    plot_mode_radar(query_rows)
    write_summary(query_rows, upload_rows, summary)

    print(f"Saved graphs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
