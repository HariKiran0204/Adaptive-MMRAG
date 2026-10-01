import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
METRICS_DIR = BASE_DIR / "metrics"
SUMMARY_PATH = BASE_DIR / "processed_output" / "processing_summary.json"
OUTPUT_DIR = BASE_DIR / "report_graphs"


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def ensure_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.22)


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_palette():
    return {
        "navy": "#17324d",
        "teal": "#1f7a8c",
        "gold": "#d9a441",
        "brick": "#b24a3f",
        "sage": "#6c8f6b",
        "slate": "#4f5d75",
        "cream": "#f6f2ea",
    }


def graph_kpi_dashboard(query_rows, upload_rows, summary):
    colors = build_palette()

    total_queries = len(query_rows)
    rag_queries = sum(1 for row in query_rows if row["mode"] == "rag")
    avg_total = np.mean([float(row["total_query_duration_seconds"]) for row in query_rows]) if query_rows else 0.0
    avg_generation = np.mean([float(row["generation_duration_seconds"]) for row in query_rows]) if query_rows else 0.0
    latest_chunks = summary.get("total_chunks", 0)
    latest_tokens = summary.get("total_tokens", 0)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    cards = [
        (4, 54, 28, 30, colors["navy"], "Total Queries", str(total_queries)),
        (36, 54, 28, 30, colors["teal"], "RAG Queries", str(rag_queries)),
        (68, 54, 28, 30, colors["gold"], "Avg Query Time", f"{avg_total:.2f}s"),
        (4, 14, 28, 30, colors["brick"], "Avg Generation", f"{avg_generation:.2f}s"),
        (36, 14, 28, 30, colors["sage"], "Latest Chunks", str(latest_chunks)),
        (68, 14, 28, 30, colors["slate"], "Latest Tokens", str(latest_tokens)),
    ]

    for x, y, w, h, color, title, value in cards:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="none", alpha=0.96)
        ax.add_patch(rect)
        ax.text(x + 3, y + h - 8, title, fontsize=14, color="white", fontweight="bold")
        ax.text(x + 3, y + 9, value, fontsize=25, color="white", fontweight="bold")

    ax.text(4, 94, "Offline Multimodal RAG: Run Summary", fontsize=22, fontweight="bold", color=colors["navy"])
    ax.text(
        4,
        88,
        f"Generated from local logs on {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        fontsize=11,
        color="#4a4a4a",
    )
    ax.text(
        4,
        6,
        f"Upload sessions logged: {len(upload_rows)} | Data source: metrics/*.jsonl + processed_output/processing_summary.json",
        fontsize=10.5,
        color="#555555",
    )
    save(fig, "01_kpi_dashboard.png")


def graph_query_breakdown(query_rows):
    colors = build_palette()
    rows = sorted(query_rows, key=lambda row: parse_ts(row["timestamp"]))
    x = np.arange(len(rows))
    retrieval = np.array([float(row["retrieval_duration_seconds"]) for row in rows])
    generation = np.array([float(row["generation_duration_seconds"]) for row in rows])
    labels = [f"Q{i+1}" for i in range(len(rows))]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.bar(x, retrieval, color=colors["teal"], label="Retrieval")
    ax.bar(x, generation, bottom=retrieval, color=colors["navy"], label="Generation")
    if len(rows) > 30:
        step = max(1, len(rows) // 15)
        indices = list(range(0, len(rows), step))
        if indices[-1] != len(rows) - 1:
            indices.append(len(rows) - 1)
        ax.set_xticks([x[i] for i in indices])
        ax.set_xticklabels([labels[i] for i in indices], rotation=30)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    ax.set_ylabel("Seconds")
    ax.set_xlabel("Query sequence")
    ax.set_title("Per-Query Runtime Breakdown", fontsize=18, fontweight="bold")
    style_axes(ax)
    ax.legend(frameon=False)
    save(fig, "02_query_runtime_breakdown.png")


def graph_response_quality_proxy(query_rows):
    colors = build_palette()
    rows = sorted(query_rows, key=lambda row: parse_ts(row["timestamp"]))
    x = np.arange(len(rows))
    answer_lengths = np.array([int(row["answer_length_chars"]) for row in rows])
    retrieved = np.array([int(row["retrieved_results"]) for row in rows])
    labels = [f"Q{i+1}" for i in range(len(rows))]

    fig, ax1 = plt.subplots(figsize=(10.8, 5.8))
    ax1.bar(x, answer_lengths, width=0.58, color=colors["gold"], label="Answer length")
    ax1.set_ylabel("Answer length (characters)", color=colors["gold"])
    ax1.tick_params(axis="y", colors=colors["gold"])
    if len(rows) > 30:
        step = max(1, len(rows) // 15)
        indices = list(range(0, len(rows), step))
        if indices[-1] != len(rows) - 1:
            indices.append(len(rows) - 1)
        ax1.set_xticks([x[i] for i in indices])
        ax1.set_xticklabels([labels[i] for i in indices], rotation=30)
    else:
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
    ax1.set_xlabel("Query sequence")
    ax1.set_title("Response Size and Retrieval Support", fontsize=18, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.20)

    ax2 = ax1.twinx()
    ax2.plot(x, retrieved, color=colors["brick"], marker="o", linewidth=2.4, label="Retrieved results")
    ax2.set_ylabel("Retrieved results", color=colors["brick"])
    ax2.tick_params(axis="y", colors=colors["brick"])
    ax2.spines["top"].set_visible(False)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="upper left")
    save(fig, "03_response_size_and_retrieval.png")


def graph_mode_share(query_rows):
    colors = build_palette()
    counts = Counter(row["mode"] for row in query_rows)
    labels = list(counts.keys())
    values = list(counts.values())
    pie_colors = [colors["navy"], colors["teal"], colors["gold"], colors["brick"]][: len(values)]

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.0f%%",
        startangle=110,
        colors=pie_colors,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    ax.set_title("Query Mode Share", fontsize=18, fontweight="bold")
    save(fig, "04_query_mode_share.png")


def graph_document_profile(summary):
    colors = build_palette()
    file_counts = summary.get("by_file_type", {})
    labels = list(file_counts.keys())
    values = list(file_counts.values())
    if not labels:
        labels = ["pdfs", "docx", "images", "audio"]
        values = [0, 0, 0, 0]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bars = ax.bar(labels, values, color=[colors["navy"], colors["teal"], colors["gold"], colors["brick"]][: len(labels)], width=0.58)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, str(value), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Processed files / chunk groups")
    ax.set_xlabel("Input modality")
    ax.set_title("Latest Processing Profile", fontsize=18, fontweight="bold")
    style_axes(ax)
    save(fig, "05_processing_profile.png")


def graph_efficiency_scatter(query_rows):
    colors = build_palette()
    total_time = np.array([float(row["total_query_duration_seconds"]) for row in query_rows])
    answer_lengths = np.array([int(row["answer_length_chars"]) for row in query_rows])
    retrieved = np.array([int(row["retrieved_results"]) for row in query_rows])
    modes = [row["mode"] for row in query_rows]

    mode_color = {
        "general_chat": colors["navy"],
        "rag": colors["teal"],
        "adaptive": colors["brick"],
        "baseline": colors["sage"],
    }

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    step = 1 if len(query_rows) <= 25 else max(1, len(query_rows) // 20)
    for idx, mode in enumerate(modes):
        ax.scatter(
            total_time[idx],
            answer_lengths[idx],
            s=160 + retrieved[idx] * 80,
            color=mode_color.get(mode, colors["gold"]),
            alpha=0.82,
            edgecolor="white",
            linewidth=1.2,
        )
        if idx % step == 0 or idx == len(query_rows) - 1:
            ax.text(total_time[idx] + 0.25, answer_lengths[idx] + 10, f"Q{idx+1}", fontsize=9)

    ax.set_xlabel("Total query time (s)")
    ax.set_ylabel("Answer length (characters)")
    ax.set_title("Efficiency Map of Query Responses", fontsize=18, fontweight="bold")
    style_axes(ax)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=m, markerfacecolor=mode_color.get(m, colors["gold"]), markersize=10)
        for m in sorted(set(modes))
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left")
    save(fig, "06_efficiency_map.png")


def write_notes(query_rows, summary):
    notes = [
        "Report Graph Pack",
        "=" * 18,
        "These figures were generated from actual local project logs.",
        "Recommended use in PPT/report: mention them as experimental run analytics.",
        "",
        f"Total logged queries: {len(query_rows)}",
        f"Latest processed chunks: {summary.get('total_chunks', 0)}",
        f"Latest processed source count: {len(summary.get('by_source', {}))}",
        "",
        "Generated images:",
        "01_kpi_dashboard.png",
        "02_query_runtime_breakdown.png",
        "03_response_size_and_retrieval.png",
        "04_query_mode_share.png",
        "05_processing_profile.png",
        "06_efficiency_map.png",
    ]
    (OUTPUT_DIR / "NOTES.txt").write_text("\n".join(notes), encoding="utf-8")


def main():
    ensure_dir()
    query_rows = normalize_query_rows(load_jsonl(METRICS_DIR / "query_metrics.jsonl"))
    upload_rows = load_jsonl(METRICS_DIR / "upload_metrics.jsonl")
    summary = {}
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    if not query_rows:
        raise SystemExit("No query metrics found. Run the system first and then regenerate the graphs.")

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    graph_kpi_dashboard(query_rows, upload_rows, summary)
    graph_query_breakdown(query_rows)
    graph_response_quality_proxy(query_rows)
    graph_mode_share(query_rows)
    graph_document_profile(summary)
    graph_efficiency_scatter(query_rows)
    write_notes(query_rows, summary)
    print(f"Saved polished report graphs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
