import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "evaluation_metrics_demo.json"
OUTPUT_DIR = BASE_DIR / "evaluation_graphs"


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def ensure_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def palette():
    return {
        "navy": "#17324d",
        "teal": "#2a7f92",
        "gold": "#d89b35",
        "brick": "#b34d41",
        "sage": "#6f8f72",
        "plum": "#705c8a",
        "sky": "#5d94c6"
    }


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.2)


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=260, bbox_inches="tight")
    plt.close(fig)


def graph_ragas(data):
    colors = palette()
    labels = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    bars = ax.bar(labels, values, color=[colors["navy"], colors["teal"], colors["gold"], colors["sage"]])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("RAGAS Assessment Breakdown", fontsize=18, fontweight="bold")
    style_axes(ax)
    plt.xticks(rotation=12)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=10)
    save(fig, "01_ragas_breakdown.png")


def graph_bertscore(data):
    colors = palette()
    labels = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    bars = ax.bar(labels, values, color=[colors["sky"], colors["teal"], colors["navy"]], width=0.55)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("BERTScore Evaluation", fontsize=18, fontweight="bold")
    style_axes(ax)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=11)
    save(fig, "02_bertscore_evaluation.png")


def graph_recall_at_k(data):
    colors = palette()
    k_values = data["k_values"]
    scores = data["scores"]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.plot(k_values, scores, marker="o", markersize=8, linewidth=2.6, color=colors["brick"])
    ax.fill_between(k_values, scores, alpha=0.15, color=colors["brick"])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall@K Retrieval Performance", fontsize=18, fontweight="bold")
    ax.set_xticks(k_values)
    style_axes(ax)
    for x, y in zip(k_values, scores):
        ax.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=10)
    save(fig, "03_recall_at_k.png")


def graph_semantic_similarity(data):
    colors = palette()
    labels = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    bars = ax.barh(labels, values, color=[colors["navy"], colors["teal"], colors["gold"], colors["plum"], colors["sage"]])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Semantic similarity")
    ax.set_title("Cross-Modal Semantic Similarity Analysis", fontsize=18, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=10)
    save(fig, "04_semantic_similarity.png")


def graph_memory(data):
    colors = palette()
    labels = list(data.keys())
    values = list(data.values())
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(x, values, marker="s", markersize=8, linewidth=2.8, color=colors["gold"])
    ax.fill_between(x, values, alpha=0.16, color=colors["gold"])
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Consumption Across Pipeline Stages", fontsize=18, fontweight="bold")
    style_axes(ax)
    for xi, yi in zip(x, values):
        ax.text(xi, yi + 18, f"{yi} MB", ha="center", fontsize=10)
    save(fig, "05_memory_consumption.png")


def graph_hallucination(data):
    colors = palette()
    labels = list(data.keys())
    grounded = [data[label]["Grounded"] for label in labels]
    partial = [data[label]["Partial"] for label in labels]
    hallucinated = [data[label]["Hallucinated"] for label in labels]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.bar(x, grounded, color=colors["sage"], label="Grounded")
    ax.bar(x, partial, bottom=grounded, color=colors["gold"], label="Partial")
    ax.bar(x, hallucinated, bottom=np.array(grounded) + np.array(partial), color=colors["brick"], label="Hallucinated")
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_title("Hallucination Analysis by Query Type", fontsize=18, fontweight="bold")
    style_axes(ax)
    ax.legend(frameon=False, ncols=3, loc="upper center")
    save(fig, "06_hallucination_analysis.png")


def graph_baseline_vs_adaptive(data):
    colors = palette()
    metrics = data["metrics"][:4]  # Quality metrics
    baseline = data["baseline"][:4]
    adaptive = data["adaptive"][:4]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.bar(x - width/2, baseline, width, label="Baseline (Fixed RAG)", color=colors["sky"])
    ax.bar(x + width/2, adaptive, width, label="Adaptive-MMRAG", color=colors["navy"])
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs Adaptive-MMRAG Quality Comparison", fontsize=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "07_baseline_vs_adaptive.png")


def graph_modality_distribution(data):
    colors = palette()
    labels = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=labels, 
        autopct='%1.1f%%', 
        colors=[colors["navy"], colors["teal"], colors["gold"], colors["plum"]],
        startangle=140,
        textprops=dict(color="black")
    )
    ax.set_title("Adaptive Modality Selection Distribution", fontsize=18, fontweight="bold")
    save(fig, "08_adaptive_modality_distribution.png")


def write_notes():
    notes = [
        "Adaptive-MMRAG Evaluation Graph Pack",
        "=====================================",
        "Generated files:",
        "01_ragas_breakdown.png",
        "02_bertscore_evaluation.png",
        "03_recall_at_k.png",
        "04_semantic_similarity.png",
        "05_memory_consumption.png",
        "06_hallucination_analysis.png",
        "07_baseline_vs_adaptive.png",
        "08_adaptive_modality_distribution.png"
    ]
    (OUTPUT_DIR / "README.txt").write_text("\n".join(notes), encoding="utf-8")


def main():
    ensure_dir()
    config = load_config()

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white"
    })

    graph_ragas(config["ragas"])
    graph_bertscore(config["bertscore"])
    graph_recall_at_k(config["recall_at_k"])
    graph_semantic_similarity(config["semantic_similarity"])
    graph_memory(config["memory_consumption_mb"])
    graph_hallucination(config["hallucination_analysis_percent"])
    
    if "baseline_vs_adaptive" in config:
        graph_baseline_vs_adaptive(config["baseline_vs_adaptive"])
    if "modality_selection_distribution" in config:
        graph_modality_distribution(config["modality_selection_distribution"])
    if "ablation_study_full" in config:
        graph_ablation_study(config["ablation_study_full"])
    if "multi_model_routing_stats" in config:
        graph_multi_model_routing(config["multi_model_routing_stats"])
    if "post_verification_metrics" in config:
        graph_post_verification(config["post_verification_metrics"])
        
    write_notes()
    print(f"Saved evaluation graphs to: {OUTPUT_DIR}")


def graph_ablation_study(data):
    colors = palette()
    arms = data["arms"]
    faithfulness = data["faithfulness"]
    hallucination = [h / 100.0 for h in data["hallucination_percent"]]
    latency = data["latency_seconds"]
    
    x = np.arange(len(arms))
    width = 0.26

    fig, ax1 = plt.subplots(figsize=(11.5, 6.2))
    
    b1 = ax1.bar(x - width, faithfulness, width, label="Faithfulness Score (0-1)", color=colors["navy"])
    b2 = ax1.bar(x, hallucination, width, label="Hallucination Rate (fraction)", color=colors["brick"])
    
    ax1.set_ylabel("Quality Score / Error Rate", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(arms, rotation=12, ha="right", fontsize=11)
    style_axes(ax1)
    
    # Secondary axis for latency
    ax2 = ax1.twinx()
    l1 = ax2.plot(x + width, latency, marker="o", linewidth=2.8, markersize=8, color=colors["gold"], label="Query Latency (s)")
    ax2.set_ylabel("Latency (seconds)", fontsize=13, fontweight="bold", color=colors["gold"])
    ax2.set_ylim(0, 6.0)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=False)
    
    ax1.set_title("Complete Adaptive-MMRAG Ablation Study (All 5 Experimental Arms)", fontsize=16, fontweight="bold")
    save(fig, "09_ablation_study_comparison.png")


def graph_multi_model_routing(data):
    colors = palette()
    labels = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    bars = ax.barh(labels, values, color=[colors["teal"], colors["navy"], colors["gold"]], height=0.55)
    ax.set_xlabel("Query Distribution Count", fontsize=12)
    ax.set_title("Resource-Aware Multi-Model Routing Decisions", fontsize=17, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.2)
    for bar, val in zip(bars, values):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2, f"{val} queries", va="center", fontsize=11)
    save(fig, "10_multi_model_resource_routing.png")


def graph_post_verification(data):
    colors = palette()
    labels = list(data.keys())[:3]
    values = [data[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    bars = ax.bar(labels, values, color=[colors["sage"], colors["gold"], colors["brick"]], width=0.55)
    ax.set_ylabel("Number of Responses", fontsize=12)
    ax.set_title("Post-Generation Evidence Verification & Grounding Audit", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 105)
    style_axes(ax)
    plt.xticks(rotation=8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val}", ha="center", fontsize=11, fontweight="bold")
    save(fig, "11_post_generation_verification_faithfulness.png")


if __name__ == "__main__":
    main()


