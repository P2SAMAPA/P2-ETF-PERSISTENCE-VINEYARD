import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from vineyard import compute_persistence_diagram, build_vineyards, compute_etf_scores_from_vines, get_representative_cycle

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
        if returns.empty or len(returns) < max(config.WINDOWS) + config.INNER_WIN + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for outer_win in config.WINDOWS:
            if len(returns) < outer_win + config.INNER_WIN + 10:
                print(f"  Skipping window {outer_win}d (insufficient data)")
                continue
            print(f"  Processing outer window {outer_win}d...")
            # Use the last `outer_win` days of returns
            ret_outer = returns.iloc[-outer_win:]
            # Create a sequence of consecutive sub‑windows of length `INNER_WIN`
            # We need at least `N_SEQ` sub‑windows. Total available: `outer_win - INNER_WIN + 1`.
            total_sub = len(ret_outer) - config.INNER_WIN + 1
            if total_sub < config.N_SEQ:
                print(f"    Not enough sub‑windows (need {config.N_SEQ}, have {total_sub}) – skipping")
                continue
            # Take the last `N_SEQ` sub‑windows
            diagrams = []
            simplex_trees = []
            for start_idx in range(total_sub - config.N_SEQ, total_sub):
                sub_ret = ret_outer.iloc[start_idx:start_idx+config.INNER_WIN]
                diag, stree = compute_persistence_diagram(sub_ret, config.INNER_WIN, dim=1)
                if diag is None:
                    break
                diagrams.append(diag)
                simplex_trees.append(stree)
            if len(diagrams) < config.N_SEQ:
                print(f"    Could not compute enough diagrams – skipping")
                continue
            # Build vineyards from the sequence of diagrams
            vines, vine_scores = build_vineyards(diagrams)
            if not vine_scores:
                print("    No persistent vines found – falling back to eigenvector centrality")
                # Fallback: use eigenvector centrality of the last sub‑window's correlation graph
                last_ret = ret_outer.iloc[-config.INNER_WIN:]
                corr = last_ret.corr().abs().values
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
                # Compute ETF scores from top vines (using the most persistent features)
                # We need the list of simplex trees for each window index (use the ones we have)
                etf_scores_dict = compute_etf_scores_from_vines(vines, vine_scores, diagrams, simplex_trees, tickers, top_n=config.TOP_VINES)
                # Normalise
                total = sum(etf_scores_dict.values())
                if total > 0:
                    scores = {etf: etf_scores_dict[etf] / total for etf in tickers}
                else:
                    scores = {etf: 0.0 for etf in tickers}
            window_results[outer_win] = scores
            for etf, score in scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, outer_win)

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

        print(f"  Top 3 ETFs by persistence vineyard score: {[e['ticker'] for e in top_etfs]}")
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
