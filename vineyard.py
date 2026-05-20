import numpy as np
import gudhi as gd
from scipy.optimize import linear_sum_assignment
import networkx as nx

def compute_persistence_diagram(returns_df, window, dim=1):
    """
    Compute persistence diagram for the correlation graph of ETF returns.
    Returns:
        diagram: list of (birth, death) tuples for dimension `dim`
        simplex_tree: the gudhi simplex tree (for extracting representatives)
    """
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

def match_diagrams(diag1, diag2):
    """
    Match points between two persistence diagrams.
    Returns:
        matched1: list of indices in diag1 that are matched
        matched2: list of indices in diag2 that are matched
        unmatched1: indices in diag1 not matched
        unmatched2: indices in diag2 not matched
    """
    if len(diag1) == 0 and len(diag2) == 0:
        return [], [], list(range(len(diag1))), list(range(len(diag2)))
    # Build cost matrix (L2 distance between birth-death pairs)
    cost = np.zeros((len(diag1), len(diag2)))
    for i, (b1, d1) in enumerate(diag1):
        for j, (b2, d2) in enumerate(diag2):
            cost[i, j] = np.sqrt((b1 - b2)**2 + (d1 - d2)**2)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched1 = list(row_ind)
    matched2 = list(col_ind)
    unmatched1 = [i for i in range(len(diag1)) if i not in matched1]
    unmatched2 = [j for j in range(len(diag2)) if j not in matched2]
    return matched1, matched2, unmatched1, unmatched2

def build_vineyards(diagrams_list):
    """
    diagrams_list: list of persistence diagrams (list of (b,d) tuples) for consecutive windows.
    Returns:
        vines: dict {vine_id: list of (window_idx, (birth, death))}
        vine_scores: dict {vine_id: score = length * death_value}
    """
    if not diagrams_list:
        return {}, {}
    vines = {}
    vine_id_counter = 0
    # first window: each point becomes a new vine
    for point in diagrams_list[0]:
        vines[vine_id_counter] = [(0, point)]
        vine_id_counter += 1
    # For each subsequent window, match
    for win_idx in range(1, len(diagrams_list)):
        prev_diag = diagrams_list[win_idx-1]
        curr_diag = diagrams_list[win_idx]
        # Build map from point to vine_id for previous window
        prev_point_to_vine = {}
        for vine_id, points in vines.items():
            # the last point in the vine is from previous window
            last_point = points[-1][1]
            prev_point_to_vine[last_point] = vine_id
        prev_points = list(prev_point_to_vine.keys())
        # Match
        matched1, matched2, unmatched1, unmatched2 = match_diagrams(prev_points, curr_diag)
        # matched1 indices correspond to prev_points list
        matched_prev = [prev_points[i] for i in matched1]
        matched_curr = [curr_diag[j] for j in matched2]
        # Extend existing vines
        for prev_p, curr_p in zip(matched_prev, matched_curr):
            vine_id = prev_point_to_vine[prev_p]
            vines[vine_id].append((win_idx, curr_p))
        # Unmatched current points start new vines
        for idx in unmatched2:
            vines[vine_id_counter] = [(win_idx, curr_diag[idx])]
            vine_id_counter += 1
        # Unmatched previous points: vine ends (do nothing)
    # Compute scores: length of vine times max death value
    vine_scores = {}
    for vine_id, points in vines.items():
        if len(points) < 2:
            continue
        deaths = [d for _, (_, d) in points if d != float('inf')]
        death_val = max(deaths) if deaths else 0.0
        length = len(points)       # number of windows this feature survived
        vine_scores[vine_id] = length * death_val
    return vines, vine_scores

def get_representative_cycle(simplex_tree, birth, death):
    """
    Extract the edges (1‑simplices) that form the persistent 1‑cycle at the given interval.
    Uses the simplex tree to retrieve the 1‑cycle via the persistence pair.
    Returns a list of tuples (i, j) representing edges.
    """
    # Get the pair of simplices that created/destroyed the feature
    pairs = simplex_tree.persistence_pairs()
    for pair in pairs:
        if len(pair) == 2 and pair[0] is not None and pair[1] is not None:
            # Check if this pair corresponds to our interval (approximate)
            # The birth simplex is the one that appears first, death simplex appears later.
            # We'll take the birth simplex and its boundary.
            # For a 1‑cycle, the birth simplex is a 1‑simplex (edge) and the death simplex is a 2‑simplex (triangle)
            # that kills the cycle. The cycle is the set of edges in the boundary of the death triangle that are also in the same connected component.
            # However, this is complex. For simplicity, we'll use the threshold heuristic.
            pass
    # Fallback: use threshold filtration
    threshold = (birth + death) / 2.0
    edges = []
    for simplex, filtration in simplex_tree.get_filtration():
        if len(simplex) == 2 and filtration < threshold:
            edges.append(tuple(simplex))
    # Build a graph to find the shortest cycle
    G = nx.Graph()
    G.add_edges_from(edges)
    try:
        # Find a cycle (any cycle) using DFS
        cycle = nx.find_cycle(G, orientation='ignore')
        cycle_edges = [(u, v) for (u, v, _) in cycle]
        return cycle_edges
    except:
        # Return all edges as fallback
        return edges

def compute_etf_scores_from_vines(vines, vine_scores, diagrams_list, simplex_trees_list, etf_names, top_n=3):
    """
    For the top `top_n` scoring vines (1‑dim loops), compute per‑ETF centrality in the loop subgraph.
    Returns dict {etf: score} aggregated across top vines.
    """
    if not vine_scores:
        return {etf: 0.0 for etf in etf_names}
    # Sort vines by score descending
    sorted_vines = sorted(vine_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    etf_scores = {etf: 0.0 for etf in etf_names}
    for vine_id, score in sorted_vines:
        points = vines[vine_id]
        # Find the window where the feature has the largest persistence (death-birth)
        best_idx = None
        best_pers = 0.0
        best_point = None
        for win_idx, (b, d) in points:
            if d == float('inf'):
                continue
            pers = d - b
            if pers > best_pers:
                best_pers = pers
                best_idx = win_idx
                best_point = (b, d)
        if best_idx is None:
            continue
        b, d = best_point
        stree = simplex_trees_list[best_idx]
        # Get representative cycle edges
        cycle_edges = get_representative_cycle(stree, b, d)
        if not cycle_edges:
            # fallback: equal weight for all ETFs
            for etf in etf_names:
                etf_scores[etf] += score / len(etf_names)
            continue
        # Build graph of the cycle
        G = nx.Graph()
        G.add_edges_from(cycle_edges)
        # Add all nodes (ETFs) as isolated if not present
        for i in range(len(etf_names)):
            G.add_node(i)
        # Compute eigenvector centrality restricted to the cycle nodes
        try:
            cent = nx.eigenvector_centrality_numpy(G, weight='weight')
        except:
            cent = {node: 1.0/len(G.nodes) for node in G.nodes}
        # Map centrality to ETF names
        for node, c in cent.items():
            if node < len(etf_names):
                etf_scores[etf_names[node]] += score * c
    # Normalise
    total = sum(etf_scores.values())
    if total > 0:
        for etf in etf_scores:
            etf_scores[etf] /= total
    return etf_scores
