import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from vineyard import compute_persistence_diagram, get_representative_cycle

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

def build_sparsified_graph(corr, threshold=0.5):
    """Create graph with edges only where correlation > threshold."""
    n = corr.shape[0]
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            if corr[i, j] > threshold:
                G.add_edge(i, j, weight=corr[i, j])
    return G

def get_centrality_scores(G, etf_names):
    """Compute eigenvector centrality, fallback to degree if fails."""
    try:
        cent = nx.eigenvector_centrality_numpy(G, weight='weight')
    except:
        # fallback to degree centrality
        cent = nx.degree_centrality(G)
    # Convert to list in node order
    scores = [cent.get(i, 0.0) for i in range(len(etf_names))]
    return {etf_names[i]: scores[i] for i in range(len(etf_names))}

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Persistence Vineyard) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 10:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            # Compute persistence diagram and simplex tree
            diag, stree = compute_persistence_diagram(returns, win, dim=1)
            # Compute correlation matrix for fallback
            ret_win = returns.iloc[-win:]
            corr = ret_win.corr().abs().values
            # If no persistent 1‑dim features, use sparsified graph centrality
            if diag is None or len(diag) == 0:
                G = build_sparsified_graph(corr, threshold=0.5)
                scores = get_centrality_scores(G, tickers)
            else:
                # Find most persistent 1‑dim loop
                best_point = max(diag, key=lambda x: x[1]-x[0])
                b, d = best_point
                cycle_edges = get_representative_cycle(stree, b, d)
                if not cycle_edges or len(cycle_edges) < 3:
                    # Not a proper loop, fallback to sparsified graph
                    G = build_sparsified_graph(corr, threshold=0.5)
                    scores = get_centrality_scores(G, tickers)
                else:
                    # Build graph from cycle edges
                    G = nx.Graph()
                    G.add_edges_from(cycle_edges)
                    # Ensure all ETF nodes are present
                    for i in range(len(tickers)):
                        G.add_node(i)
                    scores = get_centrality_scores(G, tickers)
            window_results[win] = scores
            for etf, score in scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "centrality": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by topological centrality: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/persistence_vineyard_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Persistence Vineyard Engine complete ===")

if __name__ == "__main__":
    main()
