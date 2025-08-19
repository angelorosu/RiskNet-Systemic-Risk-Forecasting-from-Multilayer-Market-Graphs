# Deep Multilayer Graph Kernels for Systemic Risk Prediction  

This project implements a framework for **systemic risk prediction in financial markets** using **deep graph kernels on multilayer networks**. The approach combines **graph neural networks (GNNs)** with **kernel methods** to capture both temporal dynamics and structural changes in correlated financial systems.  

---

## 🔑 Key Components  

- **Data Preparation**  
  - Load historical equity price data  
  - Compute log-returns  
  - Build sector-based node lists (12+ sectors, 1000+ equities)  

- **Network Construction**  
  - Construct **multilayer graphs** representing intra- and inter-sector correlations over rolling windows  
  - Encode evolving market dependencies as dynamic graph snapshots  

- **Feature Extraction**  
  - Apply a lightweight **GraphSAGE-based GNN** to learn node embeddings from each snapshot  
  - Capture cross-sector interactions and temporal shifts  

- **Graph Kernel**  
  - Compute **Weisfeiler-Lehman (WL) graph kernels** on multilayer snapshots  
  - Quantify structural similarity and detect regime changes  

- **Systemic Risk Prediction**  
  - Train a kernel-based SVR to estimate a **DebtRank-style systemic risk score**  
  - Evaluate predictive accuracy against real stress events  

---

## ⚙️ Pipeline Overview  

1. Raw market data → log-returns  
2. Sector correlation matrices → multilayer graphs  
3. Graph snapshots → GNN embeddings + WL kernel features  
4. Kernel regression → systemic risk score prediction  

---

## 📈 Results (Example Highlights)  

- WL kernel improved RMSE by **65%** over baseline correlation models  
- Achieved **Spearman ρ ≈ 0.99**, demonstrating strong rank-order consistency  
- Demonstrated potential for **real-time systemic stress detection** and **risk propagation forecasting**  

---


## 📂 Applications  

- Financial stability monitoring  
- Stress-testing frameworks  
- Early warning systems for cross-sector contagion  
