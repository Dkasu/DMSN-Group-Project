from pathlib import Path
import csv
from collections import Counter
import statistics
import matplotlib.pyplot as plt

# PURPOSE: point to the degree table made by the maildir Task 1 pipeline.
# WHAT IT IS DOING: tells the script where to read in/out/total degree + strength values from.
DEGREE_TABLE = Path("outputs/tables/degree_table_maildir.csv")

# PURPOSE: choose where plot images will be saved.
# WHAT IT IS DOING: keeps figures organised for slides and the report.
FIGURE_DIR = Path("outputs/figures")


def load_columns(csv_path: Path):
    # PURPOSE: read degree and strength columns into Python lists.
    # WHAT IT IS DOING: collects the values needed to plot distributions and print summary stats.
    in_deg, out_deg, total_deg = [], [], []
    in_str, out_str, total_str = [], [], []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            in_deg.append(int(row["in_degree"]))
            out_deg.append(int(row["out_degree"]))
            total_deg.append(int(row["total_degree"]))

            in_str.append(int(row["in_strength"]))
            out_str.append(int(row["out_strength"]))
            total_str.append(int(row["total_strength"]))

    return in_deg, out_deg, total_deg, in_str, out_str, total_str


def freq(values):
    # PURPOSE: convert raw values into a frequency distribution.
    # WHAT IT IS DOING: counts how many nodes have each value (degree/strength).
    c = Counter(values)
    x = sorted(c.keys())
    y = [c[v] for v in x]
    return x, y


def save_loglog_scatter(values, title, xlabel, output_path: Path):
    # PURPOSE: create a standard network-analysis distribution plot.
    # WHAT IT IS DOING: shows heavy-tailed behaviour clearly on log-log axes.
    x, y = freq(values)

    # remove zeros to avoid log(0) issues
    xy = [(a, b) for a, b in zip(x, y) if a > 0 and b > 0]
    if not xy:
        return

    x, y = zip(*xy)

    plt.figure(figsize=(8, 5))
    plt.loglog(x, y, marker="o", linestyle="None")
    plt.xlabel(xlabel)
    plt.ylabel("Number of nodes")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_hist(values, title, xlabel, output_path: Path, bins=30, log_y=True):
    # PURPOSE: make a simpler distribution chart that is often easier to read in slides.
    # WHAT IT IS DOING: plots a histogram; optionally log-scales the y-axis to show long tails.
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    if log_y:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Number of nodes")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def print_summary(name, values):
    # PURPOSE: give you quick numeric summaries for your slide text.
    # WHAT IT IS DOING: prints min/median/mean/max so you can describe the distribution.
    if not values:
        return
    print(f"\n{name} summary:")
    print("  min   :", min(values))
    print("  median:", statistics.median(values))
    print("  mean  :", round(statistics.mean(values), 3))
    print("  max   :", max(values))


if __name__ == "__main__":
    # PURPOSE: run the Task 1 plotting pipeline.
    # WHAT IT IS DOING: generates distribution plots for degrees (required) and strengths (useful extra).
    if not DEGREE_TABLE.exists():
        raise SystemExit(
            f"Could not find {DEGREE_TABLE}. "
            "Run the maildir Task 1 analysis script first to generate degree_table_maildir.csv."
        )

    in_deg, out_deg, total_deg, in_str, out_str, total_str = load_columns(DEGREE_TABLE)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # --- REQUIRED FOR TASK 1: in-degree and out-degree distributions ---
    save_loglog_scatter(
        in_deg,
        "In-Degree Distribution (Maildir Directed Graph)",
        "In-degree",
        FIGURE_DIR / "in_degree_loglog.png",
    )

    save_loglog_scatter(
        out_deg,
        "Out-Degree Distribution (Maildir Directed Graph)",
        "Out-degree",
        FIGURE_DIR / "out_degree_loglog.png",
    )

    # Optional but helpful: total degree
    save_loglog_scatter(
        total_deg,
        "Total Degree Distribution (Maildir Directed Graph)",
        "Total degree (in + out)",
        FIGURE_DIR / "total_degree_loglog.png",
    )

    # Slide-friendly histograms (often easier to read than loglog)
    save_hist(
        in_deg,
        "In-Degree Histogram (Maildir Directed Graph)",
        "In-degree",
        FIGURE_DIR / "in_degree_hist.png",
        bins=30,
        log_y=True,
    )

    save_hist(
        out_deg,
        "Out-Degree Histogram (Maildir Directed Graph)",
        "Out-degree",
        FIGURE_DIR / "out_degree_hist.png",
        bins=30,
        log_y=True,
    )

    # --- OPTIONAL EXTRA (but valuable because you have weights): strength distributions ---
    save_loglog_scatter(
        in_str,
        "In-Strength Distribution (Weighted by Email Count)",
        "In-strength (sum of incoming weights)",
        FIGURE_DIR / "in_strength_loglog.png",
    )

    save_loglog_scatter(
        out_str,
        "Out-Strength Distribution (Weighted by Email Count)",
        "Out-strength (sum of outgoing weights)",
        FIGURE_DIR / "out_strength_loglog.png",
    )

    save_loglog_scatter(
        total_str,
        "Total Strength Distribution (Weighted by Email Count)",
        "Total strength (in + out)",
        FIGURE_DIR / "total_strength_loglog.png",
    )

    # Print quick summaries to copy into your slide notes if needed
    print_summary("In-degree", in_deg)
    print_summary("Out-degree", out_deg)
    print_summary("Total degree", total_deg)
    print_summary("In-strength", in_str)
    print_summary("Out-strength", out_str)
    print_summary("Total strength", total_str)

    print("\nSaved plots into:", FIGURE_DIR.resolve())