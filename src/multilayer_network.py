import pandas as pd
import networkx as nx
from itertools import combinations
import numpy as np
from joblib import Parallel, delayed
import pickle
import os


class MultilayerNetwork:
    def __init__(self, log_returns, sector_mapping, window_size=200, threshold=0.5, inter_threshold=0.3):
        self.log_returns = log_returns
        self.sector_mapping = sector_mapping
        self.window_size = window_size
        self.threshold = threshold
        self.inter_threshold = inter_threshold
        self.networks = []           # list of multilayer network snapshots
        self.centrality_measures = []  # list of centrality measures per snapshot

    def build_network(self):
        """
        Build multilayer network snapshots using a sliding window.
        The window slides by 7 days.
        This version parallelizes the snapshot construction and metric computation.
        """
        dates = self.log_returns.index
        num_dates = len(dates)
        step = 14
        indices = range(0, num_dates - self.window_size + 1, step)
        
        def process_window(i):
            window_data = self.log_returns.iloc[i:i+self.window_size]
            snapshot = self._build_snapshot(window_data)
            metrics = self._compute_centrality(snapshot)
            return snapshot, metrics

        # Parallelize over all window indices. n_jobs=-1 uses all available cores.
        results = Parallel(n_jobs=-1)(delayed(process_window)(i) for i in indices)
        
        # Unpack the results.
        self.networks = [res[0] for res in results]
        self.centrality_measures = [res[1] for res in results]

    def _build_snapshot(self, window_data):
        """
        Constructs a single multilayer network snapshot from window_data.
        - For each sector, create an intra-layer graph using correlations above the threshold.
        - Compute inter-layer edges between sectors when absolute correlation is above inter_threshold.
        """
        layers = {}

        for sector in self.sector_mapping['Sector'].unique():
            # List of tickers for the sector.
            tickers = self.sector_mapping[self.sector_mapping['Sector'] == sector]['Ticker'].values
            # Tickers available in the current window.
            available_tickers = [ticker for ticker in tickers if ticker in window_data.columns]
            if not available_tickers:
                continue

            corr_matrix = window_data[available_tickers].corr()
            G = nx.Graph()
            G.add_nodes_from(available_tickers)

            for ticker1, ticker2 in combinations(available_tickers, 2):
                corr_val = corr_matrix.loc[ticker1, ticker2]
                # Ensure corr_val is a scalar.
                if not np.isscalar(corr_val):
                    try:
                        corr_val = float(corr_val)
                    except Exception as e:
                        print(f"Error converting correlation value for {ticker1} and {ticker2}: {e}")
                        continue
                if abs(corr_val) >= self.threshold:
                    G.add_edge(ticker1, ticker2, weight=corr_val)
            layers[sector] = G

        # Compute inter-layer edges.
        inter_edges = []
        sectors = list(layers.keys())
        for i in range(len(sectors)):
            for j in range(i + 1, len(sectors)):
                sector1 = sectors[i]
                sector2 = sectors[j]
                tickers1 = list(layers[sector1].nodes())
                tickers2 = list(layers[sector2].nodes())
                if tickers1 and tickers2:
                    combined_tickers = tickers1 + tickers2
                    corr_matrix_combined = window_data[combined_tickers].corr()
                    # Submatrix for tickers1 vs tickers2.
                    corr_sub = corr_matrix_combined.loc[tickers1, tickers2]
                    for t1 in tickers1:
                        for t2 in tickers2:
                            corr_val = corr_sub.loc[t1, t2]
                            if not np.isscalar(corr_val):
                                try:
                                    corr_val = float(corr_val)
                                except Exception as e:
                                    print(f"Error converting inter-layer correlation value for {t1} and {t2}: {e}")
                                    continue
                            if abs(corr_val) >= self.inter_threshold:
                                inter_edges.append((sector1, t1, sector2, t2, corr_val))

        snapshot = {
            'layers': layers,
            'inter_edges': inter_edges,
            'window_start': window_data.index[0],
            'window_end': window_data.index[-1]
        }
        return snapshot

    def _compute_centrality(self, snapshot):
        """
        Computes centrality metrics for each sector graph in the snapshot:
        - Degree centrality.
        - Eigenvector centrality (if convergent).
        - Clustering coefficient.
        """
        metrics = {}
        for sector, G in snapshot['layers'].items():
            sector_metrics = {}
            sector_metrics['degree'] = nx.degree_centrality(G)
            try:
                sector_metrics['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
            except nx.NetworkXException:
                sector_metrics['eigenvector'] = {}
            sector_metrics['clustering'] = nx.clustering(G)
            metrics[sector] = sector_metrics
        return metrics
    

    def save_snapshots(self, path):
        """
        Saves the entire list of network snapshots and the centrality measures
        in two single pickle files.
        """
        snapshots_path = os.path.join(path, "snapshots.pkl")
        centrality_path = os.path.join(path, "centrality_measures.pkl")

        with open(snapshots_path, "wb") as f:
            pickle.dump(self.networks, f)

        with open(centrality_path, "wb") as f:
            pickle.dump(self.centrality_measures, f)

        print(f"Saved {len(self.networks)} snapshots to {snapshots_path}")
        print(f"Saved centrality measures to {centrality_path}")

    def load_snapshots(self, path):
        """
        Loads the entire list of network snapshots and centrality measures
        from their respective pickle files.
        """
        snapshots_path = os.path.join(path, "snapshots.pkl")
        centrality_path = os.path.join(path, "centrality_measures.pkl")

        with open(snapshots_path, "rb") as f:
            self.networks = pickle.load(f)

        with open(centrality_path, "rb") as f:
            self.centrality_measures = pickle.load(f)

        print(f"Loaded {len(self.networks)} snapshots from {snapshots_path}")
        print(f"Loaded centrality measures from {centrality_path}")