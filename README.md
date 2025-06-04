# Deep Multilayer Graph Kernels for Systemic Risk Prediction

**What it is:**  
A concise implementation of the framework from Rosu et al. (April 2025) that predicts systemic risk in financial markets by leveraging deep graph kernels on multilayer networks (intra- and intersector).

**Key Components:**  
- **Data Prep:** Load historical price data, compute log-returns, build sector-based node lists.  
- **Network Construction:** Build a sequence of multilayer graphs representing sector correlations over time.  
- **Feature Extraction:** Use a lightweight GNN to learn node embeddings from each graph snapshot.  
- **Graph Kernel:** Compute Weisfeiler-Lehman kernels on multilayer snapshots to capture structural changes.  
- **Risk Prediction:** Train a kernel-based regressor to estimate a DebtRank-style systemic risk score.  
