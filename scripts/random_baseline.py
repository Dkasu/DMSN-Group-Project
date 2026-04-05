from pathlib import Path
from collections import Counter
import csv

import matplotlib
matplotlib.use("Agg")  # PURPOSE: save plots without opening a GUI window.
# WHAT IT IS DOING: lets the script create image files from the terminal.
import matplotlib.pyplot as plt
import networkx as nx


# =========================
# CONFIG
# =========================

EDGE_FILE = Path("data/email-Enron.txt")
TOTAL_DEGREE_THRESHOLD = 12
KEEP_LARGEST_WEAK_COMPONENT = True

OUTPUT_DIR = Path("outputs/random_baseline")


# =========================
# HELPERS
# =========================

def load_directed_graph(edge_file: Path) -> nx.DiGraph:
    # PURPOSE: load the Enron edge-list into a directed graph.
    # WHAT IT IS DOING: reads node pairs from the file and adds them as edges.
    G = nx.DiGraph()

    with edge_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            source = int(parts[0])
            target = int(parts[1])
            G.add_edge(source, target)

    return G


def filter_by_total_degree(G: nx.DiGraph, min_total_degree: int) -> nx.DiGraph:
    # PURPOSE: recreate the same Task 1 filtered subgraph.
    # WHAT IT IS DOING: keeps only nodes whose total degree meets the threshold.
    keep_nodes = [
        node for node in G.nodes()
        if (G.in_degree(node) + G.out_degree(node)) >= min_total_degree
    ]
    return G.subgraph(keep_nodes).copy()


def keep_largest_weak_component(G: nx.DiGraph) -> nx.DiGraph:
    # PURPOSE: keep the main connected structure only.
    # WHAT IT IS DOING: removes small disconnected pieces.
    largest_nodes = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(largest_nodes).copy()


def degree_frequency(values):
    # PURPOSE: count how many nodes have each degree value.
    # WHAT IT IS DOING: turns raw degree lists into degree distributions.
    counts = Counter(values)
    x = sorted(counts.keys())
    y = [counts[v] for v in x]
    return x, y


def save_loglog_distribution(values, title: str, xlabel: str, output_file: Path) -> None:
    # PURPOSE: visualise the degree distribution shape.
    # WHAT IT IS DOING: creates a log-log plot of degree vs number of nodes.
    x, y = degree_frequency(values)
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
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()


def save_comparison_table(rows, output_file: Path) -> None:
    # PURPOSE: save the key Task 3 comparison metrics in one table.
    # WHAT IT IS DOING: writes the real-vs-random results to CSV.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "real_graph", "random_graph"])
        writer.writerows(rows)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    tables_dir = OUTPUT_DIR / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not EDGE_FILE.exists():
        raise SystemExit(f"Could not find {EDGE_FILE}")

    # STEP 1
    # PURPOSE: rebuild the real Task 1 graph so the comparison is consistent.
    # WHAT IT IS DOING: loads, filters, and keeps the main weakly connected component.
    G_full = load_directed_graph(EDGE_FILE)
    G_filtered = filter_by_total_degree(G_full, TOTAL_DEGREE_THRESHOLD)

    if KEEP_LARGEST_WEAK_COMPONENT:
        G_real = keep_largest_weak_component(G_filtered)
    else:
        G_real = G_filtered

    print("Real graph nodes:", G_real.number_of_nodes())
    print("Real graph edges:", G_real.number_of_edges())

    # Use undirected projection for clustering / path / degree-shape comparison.
    # PURPOSE: make the baseline comparison structurally cleaner.
    # WHAT IT IS DOING: compares graph shape without directional duplication noise.
    G_real_u = G_real.to_undirected()

    n = G_real_u.number_of_nodes()
    m = G_real_u.number_of_edges()

    print("Real undirected nodes:", n)
    print("Real undirected edges:", m)

    # STEP 2
    # PURPOSE: generate an Erdős–Rényi graph with the same n and m.
    # WHAT IT IS DOING: creates the random baseline network.
    G_rand_u = nx.gnm_random_graph(n=n, m=m, seed=42)

    print("Random undirected nodes:", G_rand_u.number_of_nodes())
    print("Random undirected edges:", G_rand_u.number_of_edges())

    # STEP 3
    # PURPOSE: compute the comparison metrics required by Task 3.
    # WHAT IT IS DOING: measures clustering, path length, and degree distribution shape.
    real_clustering = nx.average_clustering(G_real_u)
    rand_clustering = nx.average_clustering(G_rand_u)

    # Average shortest path length requires connected graphs.
    # PURPOSE: avoid errors if the random graph is disconnected.
    # WHAT IT IS DOING: uses the largest connected component if needed.
    if nx.is_connected(G_real_u):
        G_real_path = G_real_u
    else:
        largest_real_cc = max(nx.connected_components(G_real_u), key=len)
        G_real_path = G_real_u.subgraph(largest_real_cc).copy()

    if nx.is_connected(G_rand_u):
        G_rand_path = G_rand_u
    else:
        largest_rand_cc = max(nx.connected_components(G_rand_u), key=len)
        G_rand_path = G_rand_u.subgraph(largest_rand_cc).copy()

    real_path_length = nx.average_shortest_path_length(G_real_path)
    rand_path_length = nx.average_shortest_path_length(G_rand_path)

    print("\nTask 3 comparison:")
    print("Real clustering:", real_clustering)
    print("Random clustering:", rand_clustering)
    print("Real avg shortest path length:", real_path_length)
    print("Random avg shortest path length:", rand_path_length)

    # STEP 4
    # PURPOSE: compare degree distribution shape visually.
    # WHAT IT IS DOING: plots real vs random degree distributions on log-log scale.
    real_degrees = [d for _, d in G_real_u.degree()]
    rand_degrees = [d for _, d in G_rand_u.degree()]

    save_loglog_distribution(
        real_degrees,
        "Real Graph Degree Distribution",
        "Degree",
        figures_dir / "real_degree_distribution_loglog.png"
    )

    save_loglog_distribution(
        rand_degrees,
        "Random Graph Degree Distribution",
        "Degree",
        figures_dir / "random_degree_distribution_loglog.png"
    )

    # Combined comparison plot
    # PURPOSE: put both degree shapes on one figure for easier reporting.
    # WHAT IT IS DOING: overlays real and random degree distributions.
    real_x, real_y = degree_frequency(real_degrees)
    rand_x, rand_y = degree_frequency(rand_degrees)

    real_xy = [(a, b) for a, b in zip(real_x, real_y) if a > 0 and b > 0]
    rand_xy = [(a, b) for a, b in zip(rand_x, rand_y) if a > 0 and b > 0]

    plt.figure(figsize=(8, 5))
    if real_xy:
        rx, ry = zip(*real_xy)
        plt.loglog(rx, ry, marker="o", linestyle="None", label="Real graph")
    if rand_xy:
        qx, qy = zip(*rand_xy)
        plt.loglog(qx, qy, marker="x", linestyle="None", label="Random graph")

    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title("Degree Distribution Shape: Real vs Random")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "degree_distribution_comparison_loglog.png", dpi=300)
    plt.close()

    # Save comparison table
    comparison_rows = [
        ["average_clustering_coefficient", real_clustering, rand_clustering],
        ["average_shortest_path_length", real_path_length, rand_path_length],
    ]
    save_comparison_table(comparison_rows, tables_dir / "task3_comparison_table.csv")

    # Save a short text summary
    summary_file = OUTPUT_DIR / "task3_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("TASK 3 RANDOM BASELINE COMPARISON\n")
        f.write("================================\n\n")
        f.write(f"Real graph nodes (undirected): {n}\n")
        f.write(f"Real graph edges (undirected): {m}\n\n")
        f.write(f"Real clustering coefficient: {real_clustering}\n")
        f.write(f"Random clustering coefficient: {rand_clustering}\n\n")
        f.write(f"Real average shortest path length: {real_path_length}\n")
        f.write(f"Random average shortest path length: {rand_path_length}\n\n")
        f.write("Degree distribution comparison saved as log-log plots.\n")

    print("\nSaved Task 3 outputs to:", OUTPUT_DIR)