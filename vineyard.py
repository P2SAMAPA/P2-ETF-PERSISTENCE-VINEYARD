import numpy as np
import gudhi as gd
import networkx as nx

def compute_persistence_diagram(returns_df, window, dim=1):
    if len(returns_df) < window:
        return None, None
    ret_win = returns_df.iloc[-window:]
    corr = ret_win.corr().abs().values
    dist = 1 - corr
    rips = gd.RipsComplex(distance_matrix=dist, max_edge_length=1.0)
    simplex_tree = rips.create_simplex_tree(max_dimension=dim+1)
    persistence = simplex_tree.persistence()
    diagram = [(b, d) for dim_idx, (b, d) in persistence if dim_idx == dim]
    return diagram, simplex_tree

def get_representative_cycle(simplex_tree, birth, death):
    """
    Retrieve the edges that represent the persistent 1‑cycle at the given interval.
    Uses a heuristic: take all 1‑simplices that appear before the death threshold.
    Returns a list of edges (tuple of node indices).
    """
    # Use the death threshold (midpoint between birth and death) to avoid including many edges
    threshold = (birth + death) / 2.0
    edges = []
    for simplex, filtration in simplex_tree.get_filtration():
        if len(simplex) == 2 and filtration < threshold:
            edges.append(tuple(simplex))
    return edges

def compute_etf_scores_from_diagram(diagram, simplex_tree, etf_names, top_n=3):
    """
    Compute per‑ETF scores based on all persistent 1‑dim loops.
    For each loop (birth, death), compute its persistence = death - birth.
    Find the edges representing the loop (via get_representative_cycle).
    Then for each ETF that is part of the loop, add persistence / (loop size) to its score.
    This gives higher score to ETFs that belong to longer‑lasting loops.
    """
    if not diagram:
        return {etf: 0.0 for etf in etf_names}
    scores = {etf: 0.0 for etf in etf_names}
    # Sort loops by persistence descending to give more weight to persistent features
    loops = sorted(diagram, key=lambda x: x[1]-x[0], reverse=True)[:top_n]
    for (b, d) in loops:
        persistence = d - b
        edges = get_representative_cycle(simplex_tree, b, d)
        if not edges:
            continue
        # Build graph to get nodes in the loop
        G = nx.Graph()
        G.add_edges_from(edges)
        # Get all nodes in the connected component that contains the loop (should be the loop itself)
        # Actually, we want the nodes that are part of the cycle. For simplicity, take all nodes in the graph (which might be more than the loop if edges are many)
        nodes = set()
        for u, v in edges:
            nodes.add(u)
            nodes.add(v)
        # Number of nodes in the loop
        loop_size = len(nodes)
        if loop_size == 0:
            continue
        # Add persistence / loop_size to each node
        for node in nodes:
            if node < len(etf_names):
                scores[etf_names[node]] += persistence / loop_size
    # Normalise
    total = sum(scores.values())
    if total > 0:
        for etf in scores:
            scores[etf] /= total
    return scores
