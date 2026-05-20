# Topological Persistence Vineyard Engine

Tracks how persistence diagrams of the ETF correlation graph evolve over time (vineyard). Features that persist across many windows are considered stable. The engine computes eigenvector centrality at the window where the most persistent topological feature (1‑dim loop) appears. Higher centrality ETFs are more central in the persistent topological structure → overweight signal.

- **Persistence:** 1‑dim loops from Rips complex on correlation distance
- **Vineyard:** Hungarian matching of features across consecutive windows
- **Score:** eigenvector centrality at the optimal window
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
