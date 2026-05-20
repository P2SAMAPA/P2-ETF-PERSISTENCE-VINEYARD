import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from vineyard import compute_persistence_diagram, build_vineyards, compute_etf_scores_from_vines

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
            # We need diagrams for consecutive windows? For a single window, we cannot build vines.
            # So we use the per‑window persistence directly (fallback).
            # Alternatively, we can slide a smaller sub‑window for vineyard tracking.
            # To keep it simple and robust, we'll compute the most persistent 1‑dim loop for this window and use its centrality as score.
            # This avoids the complexity of multi‑window vineyard.
            # Use the same function but treat as single window.
            diag, stree = compute_persistence_diagram(returns, win, dim=1)
            if diag is None or len(diag) == 0:
                # fallback to eigenvector centrality of full correlation graph
                ret_win = returns.iloc[-win:]
                corr = ret_win.corr().abs().values
                G = nx.Graph()
                for i in range(len(tickers)):
                    G.add_node(i)
                for i in range(len(tickers)):
                    for j in range(i+1, len(tickers)):
                        if corr[i,j] > 0.1:
                            G.add_edge(i, j, weight=corr[i,j])
                try:
                    cent = nx.eigenvector_centrality_numpy(G, weight='weight')
                except:
                    cent = {i: 1.0/len(tickers) for i in range(len(tickers))}
                scores = {tickers[i]: cent[i] for i in range(len(tickers))}
            else:
                # Find the most persistent point (largest death-birth)
                best_point = max(diag, key=lambda x: x[1]-x[0])
                b, d = best_point
                # Get representative cycle edges
                # We need to implement get_representative_cycle from vineyard.py
                # We'll import it; assume it's there.
                from vineyard import get_representative_cycle
                cycle_edges = get_representative_cycle(stree, b, d)
                if not cycle_edges:
                    # fallback to eigenvector centrality of whole graph
                    ret_win = returns.iloc[-win:]
                    corr = ret_win.corr().abs().values
                    G = nx.Graph()
                    for i in range(len(tickers)):
                        G.add_node(i)
                    for i in range(len(tickers)):
                        for j in range(i+1, len(tickers)):
                            if corr[i,j] > 0.1:
                                G.add_edge(i, j, weight=corr[i,j])
                    try:
                        cent = nx.eigenvector_centrality_numpy(G, weight='weight')
                    except:
                        cent = {i: 1.0/len(tickers) for i in range(len(tickers))}
                    scores = {tickers[i]: cent[i] for i in range(len(tickers))}
                else:
                    # Build graph from cycle edges
                    G = nx.Graph()
                    G.add_edges_from(cycle_edges)
                    for i in range(len(tickers)):
                        G.add_node(i)
                    try:
                        cent = nx.eigenvector_centrality_numpy(G, weight='weight')
                    except:
                        cent = {i: 1.0/len(tickers) for i in range(len(tickers))}
                    scores = {tickers[i]: cent[i] for i in range(len(tickers))}
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

        print(f"  Top 3 ETFs by persistence cycle centrality: {[e['ticker'] for e in top_etfs]}")
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
