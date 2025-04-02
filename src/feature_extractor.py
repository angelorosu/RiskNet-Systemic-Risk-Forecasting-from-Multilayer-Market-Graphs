import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class FeatureExtractor:
    def __init__(self, network):
        """
        network is an instance of the MultilayerNetwork class.
        """
        self.network = network
        self.centrality_features = {}  # dict mapping snapshot index to {node: {feature: value}}
        self.financial_features = {}   # dict mapping snapshot index to {node: {feature: value}}
        self.features = {}             # dict mapping snapshot index to DataFrame of combined features

    def compute_centrality(self):
        """
        Extracts centrality features: degree, eigenvector, clustering coefficient for each snapshot
        from MultilayerNetwork.
        """
        for idx, snapshot in enumerate(self.network.networks):
            snapshot_features = {}
            if idx < len(self.network.centrality_measures):
                metrics = self.network.centrality_measures[idx]
                # metrics is a dictionary: {sector: {'degree': {...}, 'eigenvector': {...}, 'clustering': {...}}}
                for sector, cent_dict in metrics.items():
                    for node, value in cent_dict.get('degree', {}).items():
                        if node not in snapshot_features:
                            snapshot_features[node] = {}
                        snapshot_features[node]['degree'] = value
                    for node, value in cent_dict.get('eigenvector', {}).items():
                        if node not in snapshot_features:
                            snapshot_features[node] = {}
                        snapshot_features[node]['eigenvector'] = value
                    for node, value in cent_dict.get('clustering', {}).items():
                        if node not in snapshot_features:
                            snapshot_features[node] = {}
                        snapshot_features[node]['clustering'] = value
            self.centrality_features[idx] = snapshot_features

    def compute_financial_indicators(self):
        """
        For each snapshot, compute mean return and volatility using log returns from the network.
        """
        for idx, snapshot in enumerate(self.network.networks):
            window_start = snapshot['window_start']
            window_end = snapshot['window_end']
            window_data = self.network.log_returns.loc[window_start:window_end]
            snapshot_features = {}
            # All nodes present in any layer of snapshot.
            all_nodes = set()
            for layer in snapshot['layers'].values():
                all_nodes.update(layer.nodes())
            for node in all_nodes:
                if node in window_data.columns:
                    mean_return = window_data[node].mean()
                    volatility = window_data[node].std()
                    snapshot_features[node] = {
                        'mean_return': mean_return,
                        'volatility': volatility
                    }
            self.financial_features[idx] = snapshot_features

    def combine_features(self):
        """
        Combines centrality and financial features into a single DataFrame for each snapshot.
        """
        for idx in self.centrality_features.keys():
            if idx in self.financial_features:
                # Combine both feature sets
                nodes = set(self.centrality_features[idx].keys()) | set(self.financial_features[idx].keys())
                combined_features = {}
                
                for node in nodes:
                    node_features = []
                    
                    # Add centrality features (or zeros if missing)
                    if node in self.centrality_features[idx]:
                        cent_feats = self.centrality_features[idx][node]
                        node_features.extend([
                            cent_feats.get('degree', 0),
                            cent_feats.get('eigenvector', 0),
                            cent_feats.get('clustering', 0)
                        ])
                    else:
                        node_features.extend([0, 0, 0])  # Default values
                    
                    # Add financial features (or zeros if missing)
                    if node in self.financial_features[idx]:
                        fin_feats = self.financial_features[idx][node]
                        node_features.extend([
                            fin_feats.get('mean_return', 0),
                            fin_feats.get('volatility', 0)
                        ])
                    else:
                        node_features.extend([0, 0])  # Default values
                    
                    combined_features[node] = node_features
                
                self.features[idx] = combined_features


        