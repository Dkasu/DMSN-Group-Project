from pathlib import Path
from collections import Counter
import csv

import matplotlib
matplotlib.use("Agg")  # save plots without opening a GUI window.
import matplotlib.pyplot as plt
import networkx as nx



EDGE_FILE = Path("data/email-Enron.txt")

# keep  only nodes with total directed degree >= 12

TOTAL_DEGREE_THRESHOLD = 12

# remove small disconnected pieces so path-based metrics make more sense
KEEP_LARGEST_WEAK_COMPONENT = True

# uses sampled betweenness instead of exact betweenness - makes betweenness centrality calculation much faster while still giving a good approximation of the top nodes
BETWEENNESS_SAMPLE_K = 200

# choose how many top nodes to save for centrality outputs.
TOP_N = 20


OUTPUT_DIR = Path("outputs/task1_edge_list_10000")



def load_directed_graph(edge_file: Path) -> nx.DiGraph:
    # load the Enron edge-list file into a directed graph.
    # reads edge pairs and skips comment/header lines.
    G = nx.DiGraph()

    with edge_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                source = int(parts[0])
                target = int(parts[1])
            except ValueError:
                continue

            G.add_edge(source, target)

    return G


def print_threshold_scan(G: nx.DiGraph, thresholds=range(10, 16)) -> None:
    # see which threshold gives a graph closest to 10,000 nodes
    # prints node/edge counts for a small range of degree thresholds
    print("\nThreshold scan:")
    for t in thresholds:
        keep_nodes = [
            node for node in G.nodes()
            if (G.in_degree(node) + G.out_degree(node)) >= t
        ]
        H = G.subgraph(keep_nodes)
        print(f"  threshold {t}: nodes={H.number_of_nodes()}, edges={H.number_of_edges()}")


def filter_by_total_degree(G: nx.DiGraph, min_total_degree: int) -> nx.DiGraph:
    # create the node analysis subgraph.
    # keeps only nodes that have in degree + out degree that meets threshold the threshold
    keep_nodes = [
        node for node in G.nodes()
        if (G.in_degree(node) + G.out_degree(node)) >= min_total_degree
    ]
    return G.subgraph(keep_nodes).copy()


def keep_largest_weak_component(G: nx.DiGraph) -> nx.DiGraph:
    # remove small disconnected fragments
    # keeps only the largest weakly connected component
    if G.number_of_nodes() == 0:
        return G.copy()

    largest_nodes = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(largest_nodes).copy()


def degree_frequency(values):
    # turn a list of degrees into a frequency distribution
    # counts how many nodes have each degree value
    counts = Counter(values)
    x = sorted(counts.keys())
    y = [counts[v] for v in x]
    return x, y


def save_loglog_distribution(values, title: str, xlabel: str, output_file: Path) -> None:
    # create a log-log degree distribution plot
    # visualises whether the distribution is heavy-tailed
    x, y = degree_frequency(values)

    # remove zeros to avoid log-scale issues
    # keeps only positive x and y values
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


def save_histogram(values, title: str, xlabel: str, output_file: Path, bins: int = 30) -> None:
    # create a slide-friendly histogram
    # shows the spread of degree values with a log y-axis
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Number of nodes")
    plt.title(title)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()


def save_degree_table(G_directed: nx.DiGraph, G_undirected: nx.Graph, output_file: Path) -> None:
    # save all node-level degree information in one CSV
    # gives you a reusable table for plots, checks, and reporting
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "node_id",
            "in_degree",
            "out_degree",
            "total_degree",
            "undirected_degree",
        ])

        for node in sorted(G_directed.nodes()):
            in_deg = G_directed.in_degree(node)
            out_deg = G_directed.out_degree(node)
            total_deg = in_deg + out_deg
            undirected_deg = G_undirected.degree(node)

            writer.writerow([node, in_deg, out_deg, total_deg, undirected_deg])


def save_top_metric_csv(metric_dict: dict, output_file: Path, metric_name: str, top_n: int = 20) -> None:
    # save the highest-ranked nodes for one metric
    # writes the top centrality results to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ranked = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", metric_name])

        for node_id, score in ranked:
            writer.writerow([node_id, score])


def save_summary(
    G_full: nx.DiGraph,
    G_filtered: nx.DiGraph,
    G_analysis: nx.DiGraph,
    G_analysis_undirected: nx.Graph,
    avg_clustering: float,
    output_file: Path
) -> None:
    # save the key Task 1 numbers in one place
    # creates a short summary file for slides/report notes
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        f.write("TASK 1 REDO SUMMARY (EDGE-LIST VERSION)\n")
        f.write("======================================\n\n")

        f.write("Dataset choice:\n")
        f.write("Using email-Enron.txt only\n")
        f.write("No weights available in the provided file\n\n")

        f.write("Full directed graph:\n")
        f.write(f"Nodes: {G_full.number_of_nodes()}\n")
        f.write(f"Edges: {G_full.number_of_edges()}\n\n")

        f.write("Filtering:\n")
        f.write(f"Total degree threshold: {TOTAL_DEGREE_THRESHOLD}\n")
        f.write(f"Filtered nodes: {G_filtered.number_of_nodes()}\n")
        f.write(f"Filtered edges: {G_filtered.number_of_edges()}\n\n")

        f.write("Analysis graph:\n")
        f.write(f"Directed nodes: {G_analysis.number_of_nodes()}\n")
        f.write(f"Directed edges: {G_analysis.number_of_edges()}\n")
        f.write(f"Undirected edges: {G_analysis_undirected.number_of_edges()}\n\n")

        f.write("Task 1 metric:\n")
        f.write(f"Average clustering coefficient (undirected projection): {avg_clustering}\n\n")

        f.write("Interpretation note:\n")
        f.write(
            "The supplied edge-list behaves like reciprocated communication links, "
            "so in-degree and out-degree distributions are effectively identical in this dataset.\n"
        )



if __name__ == "__main__":
    # make sure the output folders exist before saving files
    # creates a stable place for tables, figures, and graph exports
    tables_dir = OUTPUT_DIR / "tables"
    figures_dir = OUTPUT_DIR / "figures"
    graphs_dir = OUTPUT_DIR / "graphs"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    if not EDGE_FILE.exists():
        raise SystemExit(
            f"Could not find {EDGE_FILE}.\n"
            "Move email-Enron.txt into data/ or change EDGE_FILE."
        )

    # step 1: load the full directed graph from the edge-list file
    # reads the whole Enron edge-list into memory
    G_full = load_directed_graph(EDGE_FILE)

    print("Loaded full directed graph.")
    print("Full nodes:", G_full.number_of_nodes())
    print("Full edges:", G_full.number_of_edges())


    # helps edit the threshold to get closer to exactly 10,000 nodes
    # prints node counts for nearby thresholds
    print_threshold_scan(G_full, thresholds=range(10, 16))

    # step 2: filter to the target ~10k-node subgraph
    # removes low-degree nodes using the chosen threshold
    G_filtered = filter_by_total_degree(G_full, TOTAL_DEGREE_THRESHOLD)

    print("\nAfter total-degree filtering:")
    print("Filtered nodes:", G_filtered.number_of_nodes())
    print("Filtered edges:", G_filtered.number_of_edges())

    # step 3: keep only the largest weakly connected component
    # makes path-based metrics and Gephi visuals cleaner
    # removes tiny disconnected fragments
    if KEEP_LARGEST_WEAK_COMPONENT:
        G_analysis = keep_largest_weak_component(G_filtered)
    else:
        G_analysis = G_filtered

    print("\nAnalysis graph:")
    print("Analysis nodes:", G_analysis.number_of_nodes())
    print("Analysis edges:", G_analysis.number_of_edges())

    # step 4: build the undirected projection
    # computes clustering and make a clearer Gephi visual
    # collapses mirrored directed pairs into undirected links
    G_analysis_undirected = G_analysis.to_undirected()

    print("Undirected edges:", G_analysis_undirected.number_of_edges())

    # step 5: save degree table for plots/reporting
    # writes in-degree, out-degree, total degree, and undirected degree
    degree_table = tables_dir / "degree_table_10000.csv"
    save_degree_table(G_analysis, G_analysis_undirected, degree_table)
    print("Saved degree table to:", degree_table)

    # step 6: create Task 1 degree plots
    # generates the required in-degree and out-degree distributions
    # saves both histogram and log-log views
    in_degrees = [G_analysis.in_degree(n) for n in G_analysis.nodes()]
    out_degrees = [G_analysis.out_degree(n) for n in G_analysis.nodes()]

    save_histogram(
        in_degrees,
        "In-Degree Histogram (10k Edge-List Subgraph)",
        "In-degree",
        figures_dir / "in_degree_hist_10000.png",
    )

    save_histogram(
        out_degrees,
        "Out-Degree Histogram (10k Edge-List Subgraph)",
        "Out-degree",
        figures_dir / "out_degree_hist_10000.png",
    )

    save_loglog_distribution(
        in_degrees,
        "In-Degree Distribution (10k Edge-List Subgraph)",
        "In-degree",
        figures_dir / "in_degree_loglog_10000.png",
    )

    save_loglog_distribution(
        out_degrees,
        "Out-Degree Distribution (10k Edge-List Subgraph)",
        "Out-degree",
        figures_dir / "out_degree_loglog_10000.png",
    )

    print("Saved Task 1 plots to:", figures_dir)

    # step 7: compute average clustering coefficient
    # measures local triangle/team structure in the undirected projection
    avg_clustering = nx.average_clustering(G_analysis_undirected)
    print("\nAverage clustering coefficient:", avg_clustering)

    # step 8: compute betweenness centrality (sampled)
    # approximates betweenness so runtime stays manageable
    print("\nComputing betweenness centrality (sampled)...")
    k = min(BETWEENNESS_SAMPLE_K, G_analysis_undirected.number_of_nodes())
    betweenness = nx.betweenness_centrality(
        G_analysis_undirected,
        k=k,
        normalized=True,
        seed=42,
        weight=None,
    )

    betweenness_csv = tables_dir / "top_betweenness_10000.csv"
    save_top_metric_csv(betweenness, betweenness_csv, "betweenness", top_n=TOP_N)
    print("Saved top betweenness to:", betweenness_csv)

    # step 9: compute closeness centrality
    # computes closeness on the undirected analysis graph
    print("Computing closeness centrality...")
    closeness = nx.closeness_centrality(G_analysis_undirected)

    closeness_csv = tables_dir / "top_closeness_10000.csv"
    save_top_metric_csv(closeness, closeness_csv, "closeness", top_n=TOP_N)
    print("Saved top closeness to:", closeness_csv)

    # step 10: export graphs for Gephi
    # saves both directed and undirected GEXF files
    directed_gexf = graphs_dir / "enron_10000_directed.gexf"
    undirected_gexf = graphs_dir / "enron_10000_undirected.gexf"

    nx.write_gexf(G_analysis, directed_gexf)
    nx.write_gexf(G_analysis_undirected, undirected_gexf)

    print("\nSaved directed GEXF to:", directed_gexf)
    print("Saved undirected GEXF to:", undirected_gexf)

    # step 11: save summary file
    # writes the Task 1 setup and results to a plain text file
    summary_file = OUTPUT_DIR / "task1_redo_summary.txt"
    save_summary(
        G_full=G_full,
        G_filtered=G_filtered,
        G_analysis=G_analysis,
        G_analysis_undirected=G_analysis_undirected,
        avg_clustering=avg_clustering,
        output_file=summary_file,
    )

    print("Saved summary to:", summary_file)

    ## task 2 ##

    # use the same filtered graph from Task 1 for broker analysis.
    # analyses the final ~10k-node graph rather than the full raw graph.
    G_task2 = G_analysis_undirected

    # betweenness centrality
    print("\nComputing Task 2 betweenness centrality...")
    k = min(10000, G_task2.number_of_nodes())  # approximation for speed
    betweenness_task2 = nx.betweenness_centrality(G_task2, k=k, seed=42)

    # degree
    degree_task2 = dict(G_task2.degree())

    # closeness
    print("Computing Task 2 closeness centrality...")
    closeness_task2 = nx.closeness_centrality(G_task2)

    # top 10 betweenness nodes
    top_brokers = sorted(betweenness_task2.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\nTop broker candidates (by betweenness):")
    for node, score in top_brokers:
        print(f"Node {node} | Betweenness: {score:.4f} | Degree: {degree_task2[node]}")

    # simple broker classification
    print("\nBroker classification (betweenness vs degree):")
    for node, b in top_brokers:
        d = degree_task2[node]

        if d > 300:
            label = "Likely hub / routine bridge"
        else:
            label = "Potential handler-like intermediary"

        print(f"Node {node}: Degree={d}, Betweenness={b:.4f} -> {label}")

    # louvain communities
    print("\nDetecting communities with Louvain...")
    partition = community_louvain.best_partition(G_task2)

    print("\nBroker positions relative to communities:")
    for node, _ in top_brokers:
        neighbours = list(G_task2.neighbors(node))
        neighbour_communities = set(partition[n] for n in neighbours if n in partition)

        print(f"Node {node} connects {len(neighbour_communities)} communities")

    # look for more strategic broker candidates rather than obvious global hubs.
    # filters the top betweenness nodes to find nodes with strong betweenness
    # but less extreme degree, which may better match handler-like intermediary positions.

    print("\nSearching for more handler-like broker candidates...")

    # step 1: take a wider broker pool
    top_100_betweenness = sorted(
        betweenness_task2.items(),
        key=lambda x: x[1],
        reverse=True
    )[:100]

    # step 2: inspect the degree distribution of those nodes
    top_100_degrees = [degree_task2[node] for node, _ in top_100_betweenness]

    print("\nTop 100 betweenness degree summary:")
    print("Min degree:", min(top_100_degrees))
    print("Max degree:", max(top_100_degrees))
    print("Median degree:", sorted(top_100_degrees)[len(top_100_degrees) // 2])

    # step 3: choose a degree cutoff to remove the biggest hubs
    # you can adjust this after seeing the summary.
    HUB_DEGREE_CUTOFF = 500

    # step 4: keep high-betweenness nodes with lower degree
    filtered_candidates = []
    for node, b in top_100_betweenness:
        d = degree_task2[node]
        if d < HUB_DEGREE_CUTOFF:
            neighbours = list(G_task2.neighbors(node))
            neighbour_communities = set(partition[n] for n in neighbours if n in partition)
            communities_connected = len(neighbour_communities)

            # create a simple score favouring high betweenness and broad community reach,
            # while penalising very high degree.
            # pushes more "strategic" connectors upwards.
            broker_score = b * communities_connected / d

            filtered_candidates.append(
                (node, b, d, communities_connected, broker_score)
            )

    # step 5: rank the filtered candidates
    filtered_candidates = sorted(
        filtered_candidates,
        key=lambda x: x[4],
        reverse=True
    )

    print("\nPotential handler-like intermediary candidates:")
    for node, b, d, c, score in filtered_candidates[:10]:
        print(
            f"Node {node} | Betweenness: {b:.4f} | Degree: {d} | "
            f"Communities connected: {c} | Broker score: {score:.6f}"
        )

    print("\nDone.")