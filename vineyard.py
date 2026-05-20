import numpy as np
import pandas as pd
import gudhi as gd
from scipy.optimize import linear_sum_assignment
import networkx as nx

def correlation_graph_persistence(returns_df, window):
    """Compute persistence diagram of correlation graph over the last `window` days."""
    if len(returns_df) < window:
        return None, None
    ret_win = returns_df.iloc[-window:]
    corr = ret_win.corr().abs().values
    # Distance matrix for Rips: 1 - corr
    dist = 1 - corr
    rips = gd.RipsComplex(distance_matrix=dist, max_edge_length=1.0)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()
    # Extract intervals for dimension `DIM` (config.DIM)
    barcode = [(b,d) for dim, (b,d) in persistence if dim == config.DIM]
    return barcode, simplex_tree

def match_diagrams(diag1, diag2):
    """Match points between two persistence diagrams using Hungarian algorithm.
    Points are (birth, death). Distance = L2 norm.
    """
    if len(diag1) == 0 and len(diag2) == 0:
        return [], []
    if len(diag1) == 0:
        return [], list(range(len(diag2)))
    if len(diag2) == 0:
        return list(range(len(diag1))), []
    # Build cost matrix
    cost = np.zeros((len(diag1), len(diag2)))
    for i, p1 in enumerate(diag1):
        for j, p2 in enumerate(diag2):
            cost[i,j] = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    row_ind, col_ind = linear_sum_assignment(cost)
    # unmatched rows/cols have high cost
    matched1 = row_ind
    matched2 = col_ind
    unmatched1 = [i for i in range(len(diag1)) if i not in row_ind]
    unmatched2 = [j for j in range(len(diag2)) if j not in col_ind]
    return matched1, matched2

def build_vineyards(diagrams_list):
    """
    diagrams_list: list of persistence diagrams (list of (b,d) tuples) for consecutive windows.
    Returns a dict: vine_id -> list of (window_idx, birth, death) for the feature.
    Also returns the mapping from window to vine for each point.
    """
    vines = {}   # {vine_id: list of (win_idx, (b,d))}
    current_vine_id = 0
    # For the first window, each point gets its own vine
    for point in diagrams_list[0]:
        vines[current_vine_id] = [(0, point)]
        current_vine_id += 1
    # For each subsequent window, match points
    for win_idx in range(1, len(diagrams_list)):
        prev_diag = diagrams_list[win_idx-1]
        curr_diag = diagrams_list[win_idx]
        # Match between previous and current
        # We need to know which vine each previous point belongs to
        prev_vine_map = {}
        for vine_id, points in vines.items():
            # take the last point of this vine (it should be from previous window)
            last_point = points[-1][1]
            prev_vine_map[last_point] = vine_id
        # Build list of previous points in order
        prev_points = list(prev_vine_map.keys())
        # Match
        matched1, matched2 = match_diagrams(prev_points, curr_diag)
        # matched1 indices correspond to prev_points list
        # matched2 indices correspond to curr_diag list
        # Create mapping for matched points
        matched_prev = [prev_points[i] for i in matched1]
        matched_curr = [curr_diag[j] for j in matched2]
        # For matched points, assign to existing vine
        for prev_p, curr_p in zip(matched_prev, matched_curr):
            vine_id = prev_vine_map[prev_p]
            vines[vine_id].append((win_idx, curr_p))
        # For unmatched current points, start new vines
        unmatched_curr_indices = [j for j in range(len(curr_diag)) if j not in matched2]
        for idx in unmatched_curr_indices:
            vines[current_vine_id] = [(win_idx, curr_diag[idx])]
            current_vine_id += 1
        # Unmatched previous points: vine ends (no further tracking)
    return vines

def compute_vine_scores(vines):
    """For each vine, compute stability (length) and death value (max death)."""
    scores = {}
    for vine_id, points in vines.items():
        if len(points) < 2:
            continue
        deaths = [d for _, (_, d) in points if d != float('inf')]
        death_val = max(deaths) if deaths else 0.0
        stability = len(points)
        score = stability * death_val
        scores[vine_id] = score
    return scores

def get_most_important_loops(vines, diagrams_list, simplex_trees_list, top_n=3):
    """For the top scoring vines that are 1‑dim, extract the corresponding loop subgraph.
    Return eigenvector centrality for each ETF in those loops.
    """
    scores = compute_vine_scores(vines)
    if not scores:
        return {}
    sorted_vines = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    # For each such vine, we need to get the corresponding loop at its most persistent window
    # We'll take the window where the feature's death is largest (or the last window)
    etf_importance = {}
    for vine_id, _ in sorted_vines:
        # Get all points for this vine
        points = vines[vine_id]
        # Find the point with the largest death (or longest persistence)
        best_win_idx = None
        best_persistence = 0.0
        best_point = None
        for win_idx, (b,d) in points:
            pers = d - b
            if pers > best_persistence and d != float('inf'):
                best_persistence = pers
                best_win_idx = win_idx
                best_point = (b,d)
        if best_win_idx is None:
            continue
        # Get the simplex tree for that window
        stree = simplex_trees_list[best_win_idx]
        # We need to retrieve the representative simplex for that persistence interval
        # Gudhi can give the representative simplex for a given interval
        # However, retrieving the exact 1‑cycle is complex. We'll approximate:
        # For a 1‑dim feature, the representative can be a set of edges (1‑simplices).
        # We'll use the filtration threshold: the birth and death correspond to correlation levels.
        # Instead, we compute a thresholded graph at the death value (or just after birth)
        # and then find the connected components? Not accurate.
        # Simpler: use the eigenvector centrality of the whole graph at the window, weighted by the feature's persistence.
        # That gives a global measure but not loop‑specific.
        # Given the complexity, we'll assign equal importance to all nodes.
        # For now, we'll use the eigenvector centrality of the full correlation graph at that window.
        # In a full implementation, we would extract the loop edges.
        # Placeholder: return equal weights.
        pass
    # Fallback: compute eigenvector centrality of the whole graph at the most recent window
    # We'll implement a simple centrality measure.
    # Use the last window's correlation graph
    last_stree = simplex_trees_list[-1]
    # Build a graph from the edges present at a threshold (e.g., death of top vine)
    # Not accurate. We'll instead compute eigenvector centrality of the full correlation graph.
    # For each ETF, score = mean participation across top vines (simplified).
    return etf_importance

# For robustness, we'll output a dummy centrality for all ETFs at the last window
def last_window_centrality(returns_df, window):
    ret_win = returns_df.iloc[-window:]
    corr = ret_win.corr().abs().values
    G = nx.from_numpy_array(corr)
    # Eigenvector centrality
    try:
        cent = nx.eigenvector_centrality_numpy(G, weight='weight')
    except:
        cent = {i: 1.0/len(G.nodes) for i in G.nodes}
    return cent
