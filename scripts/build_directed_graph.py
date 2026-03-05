from pathlib import Path
from collections import Counter
from email import policy
from email.parser import BytesHeaderParser
from email.utils import getaddresses, parseaddr
from typing import Optional, List, Tuple
import csv
import networkx as nx

# PURPOSE: point to the raw Enron maildir root.
# WHAT IT IS DOING: tells the script where to start scanning mailbox folders.
MAILDIR_ROOT = Path("data/maildir")

# PURPOSE: define which mailbox folders count as "sent" folders across different users.
# WHAT IT IS DOING: captures common sent-folder naming variants found in the Enron maildir dataset.
SENT_FOLDER_NAMES = {
    "_sent_mail",
    "_sent",
    "old_sent",
    "sent",
    "sent_items",
    # Optional variants (include only if you actually see them in your dataset):
    "sent_mail",
    "sent-mail",
    "sent mail",
}

# PURPOSE: decide whether copied recipients should count as communication ties.
# WHAT IT IS DOING: includes CC/BCC recipients as additional directed edges if present.
INCLUDE_CC = True
INCLUDE_BCC = True

# PURPOSE: choose the first activity threshold to test on the directed graph.
# WHAT IT IS DOING: removes very low-activity nodes using in_degree + out_degree.
DEGREE_THRESHOLD = 10

# PURPOSE: decide whether to keep only the largest weakly connected component.
# WHAT IT IS DOING: makes path-based metrics and Gephi visuals easier to interpret.
KEEP_LARGEST_WEAK_COMPONENT = True

# PURPOSE: speed up betweenness on larger graphs.
# WHAT IT IS DOING: uses sampling for approximate betweenness if set to an integer.
BETWEENNESS_SAMPLE_K = 200  # set to None for exact betweenness

# PURPOSE: optionally limit files during testing.
# WHAT IT IS DOING: lets you do a quick dry run before parsing the full dataset.
MAX_FILES = None  # e.g. set to 500 for a quick test

# PURPOSE: parse headers only, not the full body.
# WHAT IT IS DOING: makes the script much faster and avoids unnecessary work.
HEADER_PARSER = BytesHeaderParser(policy=policy.default)


def normalise_email(value: Optional[str]) -> Optional[str]:
    # PURPOSE: turn raw header text into a clean email address.
    # WHAT IT IS DOING: extracts the actual email and standardises casing/spacing.
    if not value:
        return None

    _, addr = parseaddr(value)
    addr = addr.strip().lower()

    if "@" not in addr:
        return None

    return addr


def extract_recipients(msg) -> List[str]:
    # PURPOSE: collect all recipient email addresses from To / CC / BCC.
    # WHAT IT IS DOING: returns a cleaned, de-duplicated list of recipients for one email.
    header_values = []

    header_values.extend(msg.get_all("to", []))

    if INCLUDE_CC:
        header_values.extend(msg.get_all("cc", []))

    if INCLUDE_BCC:
        header_values.extend(msg.get_all("bcc", []))

    recipients: List[str] = []
    seen = set()

    for _, addr in getaddresses(header_values):
        addr = addr.strip().lower()

        if "@" not in addr:
            continue

        if addr in seen:
            continue

        seen.add(addr)
        recipients.append(addr)

    return recipients


def is_in_sent_folder(file_path: Path) -> bool:
    # PURPOSE: decide whether a message file lives inside a sent-like folder.
    # WHAT IT IS DOING: checks all folder names in the file's path for an exact sent-folder match.
    folder_parts = [part.lower() for part in file_path.parts[:-1]]
    return any(part in SENT_FOLDER_NAMES for part in folder_parts)


def build_edge_counter(maildir_root: Path) -> Tuple[Counter, int, int]:
    # PURPOSE: create weighted sender->recipient edges from the raw maildir.
    # WHAT IT IS DOING: scans message files, extracts headers, and counts repeated interactions.
    edge_counter: Counter = Counter()
    sent_folder_files_seen = 0
    valid_messages_used = 0

    for file_path in maildir_root.rglob("*"):
        if not file_path.is_file():
            continue

        if not is_in_sent_folder(file_path):
            continue

        sent_folder_files_seen += 1

        if MAX_FILES is not None and sent_folder_files_seen > MAX_FILES:
            break

        try:
            with file_path.open("rb") as f:
                msg = HEADER_PARSER.parse(f)
        except Exception:
            continue

        sender = normalise_email(msg.get("from"))
        if not sender:
            continue

        recipients = extract_recipients(msg)
        if not recipients:
            continue

        added_any_edge = False

        for recipient in recipients:
            # PURPOSE: avoid self-loop edges like a sender emailing themselves.
            # WHAT IT IS DOING: skips sender==recipient pairs.
            if recipient == sender:
                continue

            edge_counter[(sender, recipient)] += 1
            added_any_edge = True

        if added_any_edge:
            valid_messages_used += 1

    return edge_counter, sent_folder_files_seen, valid_messages_used


def build_weighted_directed_graph(edge_counter: Counter) -> nx.DiGraph:
    # PURPOSE: convert counted interactions into a NetworkX directed graph.
    # WHAT IT IS DOING: creates one edge per sender->recipient pair with a weight count.
    G = nx.DiGraph()

    for (source, target), weight in edge_counter.items():
        G.add_edge(source, target, weight=weight)

    return G


def filter_by_total_degree(G: nx.DiGraph, min_total_degree: int) -> nx.DiGraph:
    # PURPOSE: remove low-activity nodes.
    # WHAT IT IS DOING: keeps nodes whose in-degree + out-degree meets the threshold.
    keep_nodes = [
        node for node in G.nodes()
        if (G.in_degree(node) + G.out_degree(node)) >= min_total_degree
    ]

    return G.subgraph(keep_nodes).copy()


def keep_largest_weak_component(G: nx.DiGraph) -> nx.DiGraph:
    # PURPOSE: isolate the main connected part of the directed graph.
    # WHAT IT IS DOING: keeps only the largest weakly connected component.
    if G.number_of_nodes() == 0:
        return G.copy()

    largest_nodes = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(largest_nodes).copy()


def save_edge_list_csv(edge_counter: Counter, output_file: Path) -> None:
    # PURPOSE: save the parsed network as a reusable weighted edge list.
    # WHAT IT IS DOING: writes source, target, and interaction count to CSV.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight"])

        for (source, target), weight in edge_counter.items():
            writer.writerow([source, target, weight])


def save_degree_table(G: nx.DiGraph, output_file: Path) -> None:
    # PURPOSE: save node-level degree and strength values.
    # WHAT IT IS DOING: creates a CSV for Task 1 tables and degree plots.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "node_id",
            "in_degree",
            "out_degree",
            "total_degree",
            "in_strength",
            "out_strength",
            "total_strength",
        ])

        for node in G.nodes():
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            total_deg = in_deg + out_deg

            in_strength = G.in_degree(node, weight="weight")
            out_strength = G.out_degree(node, weight="weight")
            total_strength = in_strength + out_strength

            writer.writerow([
                node,
                in_deg,
                out_deg,
                total_deg,
                in_strength,
                out_strength,
                total_strength,
            ])


def save_top_metric_csv(metric_dict: dict, output_file: Path, metric_name: str, top_n: int = 20) -> None:
    # PURPOSE: save the highest-ranked nodes for a metric.
    # WHAT IT IS DOING: writes the top node scores to CSV for reporting.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ranked = sorted(metric_dict.items(), key=lambda item: item[1], reverse=True)[:top_n]

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", metric_name])

        for node_id, score in ranked:
            writer.writerow([node_id, score])


def save_summary_txt(
    sent_folder_files_seen: int,
    valid_messages_used: int,
    edge_counter: Counter,
    G_full: nx.DiGraph,
    G_filtered: nx.DiGraph,
    G_analysis: nx.DiGraph,
    avg_clustering: float,
    output_file: Path
) -> None:
    # PURPOSE: save the key Task 1 setup and output numbers in one place.
    # WHAT IT IS DOING: creates a summary file for meetings, slides, and the report.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        f.write("RAW MAILDIR TASK 1 SUMMARY\n")
        f.write("==========================\n\n")

        f.write("Data extraction:\n")
        f.write(f"Sent-folder files scanned: {sent_folder_files_seen}\n")
        f.write(f"Valid messages used: {valid_messages_used}\n")
        f.write(f"Unique directed sender->recipient edges: {len(edge_counter)}\n")
        f.write(f"Total counted interactions (sum of weights): {sum(edge_counter.values())}\n\n")

        f.write("Full directed graph:\n")
        f.write(f"Nodes: {G_full.number_of_nodes()}\n")
        f.write(f"Edges: {G_full.number_of_edges()}\n\n")

        f.write("Filtering:\n")
        f.write(f"Total-degree threshold: {DEGREE_THRESHOLD}\n")
        f.write(f"Filtered nodes: {G_filtered.number_of_nodes()}\n")
        f.write(f"Filtered edges: {G_filtered.number_of_edges()}\n\n")

        f.write("Analysis graph:\n")
        f.write(f"Nodes: {G_analysis.number_of_nodes()}\n")
        f.write(f"Edges: {G_analysis.number_of_edges()}\n\n")

        f.write("Task 1 metric:\n")
        f.write("Average clustering coefficient (undirected projection):\n")
        f.write(f"{avg_clustering}\n")


if __name__ == "__main__":
    # PURPOSE: run the full raw-maildir Task 1 pipeline.
    # WHAT IT IS DOING: parses emails, builds the directed graph, filters it,
    # computes Task 1 metrics, and saves files for Gephi and analysis.

    if not MAILDIR_ROOT.exists():
        raise SystemExit(
            f"MAILDIR_ROOT does not exist: {MAILDIR_ROOT}\n"
            "Move the raw 'maildir' folder into data/ or change MAILDIR_ROOT."
        )

    # Step 1: Parse raw emails into weighted sender->recipient edges.
    edge_counter, sent_folder_files_seen, valid_messages_used = build_edge_counter(MAILDIR_ROOT)

    print("Finished parsing raw maildir.")
    print("Sent-folder files scanned:", sent_folder_files_seen)
    print("Valid messages used:", valid_messages_used)
    print("Unique directed edges:", len(edge_counter))
    print("Total counted interactions:", sum(edge_counter.values()))

    # Step 2: Save the raw weighted edge list.
    edge_list_output = Path("outputs/tables/enron_maildir_edge_list.csv")
    save_edge_list_csv(edge_counter, edge_list_output)
    print(f"\nSaved edge list to: {edge_list_output}")

    # Step 3: Build the full weighted directed graph.
    G_full = build_weighted_directed_graph(edge_counter)
    print("\nBuilt full directed graph.")
    print("Full nodes:", G_full.number_of_nodes())
    print("Full edges:", G_full.number_of_edges())

    # Step 4: Filter low-activity nodes using total degree.
    G_filtered = filter_by_total_degree(G_full, DEGREE_THRESHOLD)
    print("\nAfter total-degree filtering:")
    print("Filtered nodes:", G_filtered.number_of_nodes())
    print("Filtered edges:", G_filtered.number_of_edges())

    # Step 5: Optionally keep only the largest weakly connected component.
    if KEEP_LARGEST_WEAK_COMPONENT:
        G_analysis = keep_largest_weak_component(G_filtered)
        print("\nAfter keeping largest weakly connected component:")
        print("Analysis nodes:", G_analysis.number_of_nodes())
        print("Analysis edges:", G_analysis.number_of_edges())
    else:
        G_analysis = G_filtered

    # Step 6: Build the undirected projection for clustering and cleaner Gephi visuals.
    G_analysis_undirected = G_analysis.to_undirected()

    # Step 7: Export graphs for Gephi.
    graph_dir = Path("outputs/graphs")
    graph_dir.mkdir(parents=True, exist_ok=True)

    directed_gexf = graph_dir / "enron_maildir_analysis_directed.gexf"
    undirected_gexf = graph_dir / "enron_maildir_analysis_undirected.gexf"

    nx.write_gexf(G_analysis, directed_gexf)
    nx.write_gexf(G_analysis_undirected, undirected_gexf)

    print(f"\nSaved directed Gephi graph to: {directed_gexf}")
    print(f"Saved undirected Gephi graph to: {undirected_gexf}")

    # Step 8: Save the degree/strength table.
    degree_table_output = Path("outputs/tables/degree_table_maildir.csv")
    save_degree_table(G_analysis, degree_table_output)
    print(f"Saved degree table to: {degree_table_output}")

    # Step 9: Compute average clustering on the undirected projection.
    avg_clustering = nx.average_clustering(G_analysis_undirected)
    print("\nAverage clustering coefficient (undirected projection):", avg_clustering)

    # Step 10: Compute betweenness centrality on the directed graph topology.
    # NOTE:
    # PURPOSE: rank structurally important bridge-like nodes.
    # WHAT IT IS DOING: uses the directed graph but ignores 'weight' as distance,
    # because larger email counts should not be treated as longer paths.
    print("\nComputing betweenness centrality...")
    if BETWEENNESS_SAMPLE_K is None:
        betweenness = nx.betweenness_centrality(G_analysis, normalized=True, weight=None)
    else:
        betweenness = nx.betweenness_centrality(
            G_analysis,
            k=BETWEENNESS_SAMPLE_K,
            normalized=True,
            weight=None,
            seed=42
        )

    betweenness_output = Path("outputs/tables/top_betweenness_maildir.csv")
    save_top_metric_csv(betweenness, betweenness_output, "betweenness", top_n=20)
    print(f"Saved top betweenness nodes to: {betweenness_output}")

    # Step 11: Compute closeness centrality on the directed graph.
    print("Computing closeness centrality...")
    closeness = nx.closeness_centrality(G_analysis)
    closeness_output = Path("outputs/tables/top_closeness_maildir.csv")
    save_top_metric_csv(closeness, closeness_output, "closeness", top_n=20)
    print(f"Saved top closeness nodes to: {closeness_output}")

    # Step 12: Save a summary text file.
    summary_output = Path("outputs/tables/task1_summary_maildir.txt")
    save_summary_txt(
        sent_folder_files_seen=sent_folder_files_seen,
        valid_messages_used=valid_messages_used,
        edge_counter=edge_counter,
        G_full=G_full,
        G_filtered=G_filtered,
        G_analysis=G_analysis,
        avg_clustering=avg_clustering,
        output_file=summary_output,
    )
    print(f"Saved summary to: {summary_output}")