from pathlib import Path
import csv
import networkx as nx

# PURPOSE: point to the Enron network file.
# WHAT IT IS DOING: tells the script where to read the edge list from.
EDGE_FILE = Path("data/email-Enron.txt")

# PURPOSE: set the minimum number of unique neighbours a node must have.
# WHAT IT IS DOING: filters the undirected graph by actual connectivity, not doubled directed degree.
UNDIRECTED_DEGREE_THRESHOLD = 10

# PURPOSE: decide whether to keep only the largest connected component after filtering.
# WHAT IT IS DOING: makes path-based metrics and Gephi visuals cleaner and easier to explain.
KEEP_LARGEST_CONNECTED_COMPONENT = True

# PURPOSE: control whether betweenness is exact or approximate.
# WHAT IT IS DOING: allows you to speed up betweenness if the graph is still large.
BETWEENNESS_SAMPLE_K = 200  # set to None for exact betweenness


def load_directed_graph(edge_file: Path) -> nx.DiGraph:
    # PURPOSE: load the dataset in its original stored format.
    # WHAT IT IS DOING: reads the file as a directed edge list and skips comment/header lines.
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


def build_undirected_graph(G_directed: nx.DiGraph) -> nx.Graph:
    # PURPOSE: create the graph used for the main structural analysis.
    # WHAT IT IS DOING: collapses reciprocal directed pairs into one undirected edge.
    G = nx.Graph()
    G.add_edges_from(G_directed.edges())
    return G


def check_reciprocity(G_directed: nx.DiGraph) -> tuple[int, int]:
    # PURPOSE: measure how many directed edges have their reverse edge present.
    # WHAT IT IS DOING: helps verify whether the dataset behaves like an undirected network.
    total_directed_edges = G_directed.number_of_edges()

    reciprocated = 0
    for u, v in G_directed.edges():
        if G_directed.has_edge(v, u):
            reciprocated += 1

    return reciprocated, total_directed_edges


def filter_by_undirected_degree(G_undirected: nx.Graph, min_degree: int) -> nx.Graph:
    # PURPOSE: remove low-activity nodes based on unique neighbours.
    # WHAT IT IS DOING: keeps only nodes whose undirected degree meets the threshold.
    keep_nodes = [node for node, degree in G_undirected.degree() if degree >= min_degree]
    return G_undirected.subgraph(keep_nodes).copy()


def keep_largest_component(G_undirected: nx.Graph) -> nx.Graph:
    # PURPOSE: isolate the main connected part of the filtered graph.
    # WHAT IT IS DOING: keeps only the largest connected component.
    if G_undirected.number_of_nodes() == 0:
        return G_undirected.copy()

    largest_nodes = max(nx.connected_components(G_undirected), key=len)
    return G_undirected.subgraph(largest_nodes).copy()


def save_degree_table(
    G_directed_view: nx.DiGraph,
    G_undirected_view: nx.Graph,
    output_file: Path
) -> None:
    # PURPOSE: save node-level degree information for Task 1.
    # WHAT IT IS DOING: writes in-degree, out-degree, and undirected degree into one CSV.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "node_id",
            "in_degree",
            "out_degree",
            "directed_total_degree",
            "undirected_degree"
        ])

        for node in sorted(G_undirected_view.nodes()):
            in_deg = G_directed_view.in_degree(node)
            out_deg = G_directed_view.out_degree(node)
            directed_total = in_deg + out_deg
            undirected_deg = G_undirected_view.degree(node)

            writer.writerow([
                node,
                in_deg,
                out_deg,
                directed_total,
                undirected_deg
            ])


def save_top_metric_csv(metric_dict: dict, output_file: Path, metric_name: str, top_n: int = 20) -> None:
    # PURPOSE: save the top-ranked nodes for a centrality metric.
    # WHAT IT IS DOING: writes the highest scoring nodes and their values to CSV.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ranked = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", metric_name])

        for node_id, score in ranked:
            writer.writerow([node_id, score])


def save_summary_txt(
    G_directed_full: nx.DiGraph,
    G_undirected_full: nx.Graph,
    G_undirected_filtered: nx.Graph,
    G_undirected_analysis: nx.Graph,
    reciprocity_count: int,
    reciprocity_total: int,
    avg_clustering: float,
    output_file: Path
) -> None:
    # PURPOSE: save the key numbers you will need for meetings, slides, and the report.
    # WHAT IT IS DOING: creates a readable text summary of the dataset and Task 1 setup.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    reciprocity_ratio = (reciprocity_count / reciprocity_total) if reciprocity_total else 0.0

    with output_file.open("w", encoding="utf-8") as f:
        f.write("ENRON TASK 1 REVISED SUMMARY\n")
        f.write("===========================\n\n")

        f.write("Full directed graph:\n")
        f.write(f"Nodes: {G_directed_full.number_of_nodes()}\n")
        f.write(f"Edges: {G_directed_full.number_of_edges()}\n\n")

        f.write("Full undirected graph:\n")
        f.write(f"Nodes: {G_undirected_full.number_of_nodes()}\n")
        f.write(f"Edges: {G_undirected_full.number_of_edges()}\n\n")

        f.write("Reciprocity check on directed graph:\n")
        f.write(f"Reciprocated directed edges: {reciprocity_count}\n")
        f.write(f"Total directed edges: {reciprocity_total}\n")
        f.write(f"Reciprocity ratio: {reciprocity_ratio:.6f}\n\n")

        f.write("Filtering setup:\n")
        f.write(f"Undirected degree threshold: {UNDIRECTED_DEGREE_THRESHOLD}\n")
        f.write(f"Filtered undirected nodes: {G_undirected_filtered.number_of_nodes()}\n")
        f.write(f"Filtered undirected edges: {G_undirected_filtered.number_of_edges()}\n\n")

        f.write("Analysis graph (after optional largest connected component step):\n")
        f.write(f"Nodes: {G_undirected_analysis.number_of_nodes()}\n")
        f.write(f"Edges: {G_undirected_analysis.number_of_edges()}\n\n")

        f.write("Task 1 metric:\n")
        f.write(f"Average clustering coefficient: {avg_clustering}\n")


if __name__ == "__main__":
    # PURPOSE: run the full revised Task 1 pipeline.
    # WHAT IT IS DOING: loads the graph, checks reciprocity, builds the undirected view,
    # filters it, computes metrics, and exports outputs.

    # Step 1: Load the original directed representation.
    G_directed_full = load_directed_graph(EDGE_FILE)
    print("Loaded directed graph.")
    print("Directed nodes:", G_directed_full.number_of_nodes())
    print("Directed edges:", G_directed_full.number_of_edges())

    # Step 2: Check reciprocity in the stored directed graph.
    reciprocity_count, reciprocity_total = check_reciprocity(G_directed_full)
    reciprocity_ratio = (reciprocity_count / reciprocity_total) if reciprocity_total else 0.0
    print("\nReciprocity check:")
    print("Reciprocated directed edges:", reciprocity_count)
    print("Total directed edges:", reciprocity_total)
    print("Reciprocity ratio:", round(reciprocity_ratio, 6))

    # Step 3: Build the undirected graph used for structural analysis.
    G_undirected_full = build_undirected_graph(G_directed_full)
    print("\nBuilt undirected graph.")
    print("Undirected nodes:", G_undirected_full.number_of_nodes())
    print("Undirected edges:", G_undirected_full.number_of_edges())

    # Step 4: Filter by undirected degree.
    G_undirected_filtered = filter_by_undirected_degree(
        G_undirected_full,
        UNDIRECTED_DEGREE_THRESHOLD
    )
    print("\nAfter undirected-degree filtering:")
    print("Filtered nodes:", G_undirected_filtered.number_of_nodes())
    print("Filtered edges:", G_undirected_filtered.number_of_edges())

    # Step 5: Optionally keep only the largest connected component.
    if KEEP_LARGEST_CONNECTED_COMPONENT:
        G_undirected_analysis = keep_largest_component(G_undirected_filtered)
        print("\nAfter keeping largest connected component:")
        print("Analysis nodes:", G_undirected_analysis.number_of_nodes())
        print("Analysis edges:", G_undirected_analysis.number_of_edges())
    else:
        G_undirected_analysis = G_undirected_filtered

    # Step 6: Build the matching directed subgraph on the same analysis nodes.
    analysis_nodes = list(G_undirected_analysis.nodes())
    G_directed_analysis = G_directed_full.subgraph(analysis_nodes).copy()

    # Step 7: Save the undirected analysis graph for Gephi.
    graph_output = Path("outputs/graphs/enron_analysis_undirected.gexf")
    graph_output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G_undirected_analysis, graph_output)
    print(f"\nSaved Gephi graph to: {graph_output}")

    # Step 8: Save the degree table.
    degree_table_output = Path("outputs/tables/degree_table_revised.csv")
    save_degree_table(G_directed_analysis, G_undirected_analysis, degree_table_output)
    print(f"Saved degree table to: {degree_table_output}")

    # Step 9: Compute average clustering on the undirected analysis graph.
    avg_clustering = nx.average_clustering(G_undirected_analysis)
    print("\nAverage clustering coefficient:", avg_clustering)

    # Step 10: Compute betweenness centrality on the undirected analysis graph.
    print("\nComputing betweenness centrality...")
    if BETWEENNESS_SAMPLE_K is None:
        betweenness = nx.betweenness_centrality(G_undirected_analysis, normalized=True)
    else:
        betweenness = nx.betweenness_centrality(
            G_undirected_analysis,
            k=BETWEENNESS_SAMPLE_K,
            normalized=True,
            seed=42
        )

    betweenness_output = Path("outputs/tables/top_betweenness_revised.csv")
    save_top_metric_csv(betweenness, betweenness_output, "betweenness", top_n=20)
    print(f"Saved top betweenness nodes to: {betweenness_output}")

    # Step 11: Compute closeness centrality on the undirected analysis graph.
    print("Computing closeness centrality...")
    closeness = nx.closeness_centrality(G_undirected_analysis)

    closeness_output = Path("outputs/tables/top_closeness_revised.csv")
    save_top_metric_csv(closeness, closeness_output, "closeness", top_n=20)
    print(f"Saved top closeness nodes to: {closeness_output}")

    # Step 12: Save a summary text file.
    summary_output = Path("outputs/tables/task1_summary_revised.txt")
    save_summary_txt(
        G_directed_full=G_directed_full,
        G_undirected_full=G_undirected_full,
        G_undirected_filtered=G_undirected_filtered,
        G_undirected_analysis=G_undirected_analysis,
        reciprocity_count=reciprocity_count,
        reciprocity_total=reciprocity_total,
        avg_clustering=avg_clustering,
        output_file=summary_output
    )
    print(f"Saved summary to: {summary_output}")