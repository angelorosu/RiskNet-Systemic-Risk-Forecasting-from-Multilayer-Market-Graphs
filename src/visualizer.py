# visualizer.py
import matplotlib.pyplot as plt
import networkx as nx

class Visualizer:
    def __init__(self, network, predictions, actuals):
        """
        Args:
            network: MultilayerNetwork instance.
            predictions: Array-like, predicted systemic risk values.
            actuals: Array-like, actual systemic risk (DebtRank) values.
        """
        self.network = network
        self.predictions = predictions
        self.actuals = actuals

    def plot_network(self, snapshot_index=0):
        """
        Plots the multilayer network structure for a given snapshot.
        
        Args:
            snapshot_index: Integer index of the snapshot to plot.
        """
        snapshot = self.network.networks[snapshot_index]
        # Merge all intra-layer graphs and inter-layer edges into one combined graph.
        G = nx.Graph()
        # Add intra-layer edges from each sector.
        for sector, layer in snapshot['layers'].items():
            G.add_nodes_from(layer.nodes(data=True))
            G.add_edges_from(layer.edges(data=True))
        # Add inter-layer edges.
        for edge in snapshot['inter_edges']:
            # Each edge is a tuple: (sector1, ticker1, sector2, ticker2, weight)
            _, ticker1, _, ticker2, weight = edge
            G.add_edge(ticker1, ticker2, weight=weight)
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
        plt.title(f"Multilayer Network Snapshot {snapshot_index}")
        plt.show()

    def plot_connectivity(self):
        """
        Plots cross-sector connectivity over time.
        Here, connectivity is measured as the number of inter-layer edges per snapshot.
        """
        connectivity = []
        times = []
        for snapshot in self.network.networks:
            times.append(snapshot['window_start'])
            connectivity.append(len(snapshot['inter_edges']))
        plt.figure(figsize=(10, 6))
        plt.plot(times, connectivity, marker='o', linestyle='-')
        plt.xlabel("Time")
        plt.ylabel("Number of Inter-layer Edges")
        plt.title("Cross-Sector Connectivity Over Time")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_predictions(self):
        """
        Plots predicted vs. actual systemic risk (DebtRank) values.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(self.actuals, self.predictions, color='blue', label='Predictions')
        # Plot a reference line (ideal predictions).
        min_val = min(min(self.actuals), min(self.predictions))
        max_val = max(max(self.actuals), max(self.predictions))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal')
        plt.xlabel("Actual DebtRank")
        plt.ylabel("Predicted DebtRank")
        plt.title("Predicted vs. Actual Systemic Risk")
        plt.legend()
        plt.show()
