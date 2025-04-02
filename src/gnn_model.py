import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, AttentionalAggregation
import networkx as nx
import numpy as np
from grakel import GraphKernel
from joblib import Parallel, delayed
# graphkernel
from grakel.graph import Graph as grakelGraph

# Enable MPS fallback (optional, uncomment if needed)
# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

def convert_snapshot_to_pyg(snapshot, features_dict, feature_dim=5):
    """
    Convert a network snapshot to PyTorch Geometric data object with proper features.
    
    Args:
        snapshot: Network snapshot dictionary
        features_dict: Dictionary of node features
        feature_dim: Dimension of node features (default=5)
    
    Returns:
        PyTorch Geometric Data object
    """
    # Create combined graph from all layers
    G_combined = nx.Graph()
    for sector, G_layer in snapshot['layers'].items():
        G_combined.add_nodes_from(G_layer.nodes())
        G_combined.add_edges_from(G_layer.edges(data=True))
    
    # Add inter-edges
    for edge in snapshot['inter_edges']:
        _, t1, _, t2, weight = edge
        G_combined.add_edge(t1, t2, weight=weight)
    
    # Create node index mapping
    node_list = list(G_combined.nodes())
    node_idx = {node: i for i, node in enumerate(node_list)}
    
    # Create edge index tensor
    edge_index = []
    for u, v in G_combined.edges():
        edge_index.append([node_idx[u], node_idx[v]])
        edge_index.append([node_idx[v], node_idx[u]])  # Add reverse edge for undirected graph
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Create node feature tensors
    feature_list = []
    for node in node_list:
        if node in features_dict:
            f = torch.tensor(features_dict[node], dtype=torch.float)
        else:
            f = torch.zeros(feature_dim, dtype=torch.float)
        feature_list.append(f)
    
    if feature_list:
        x = torch.stack(feature_list, dim=0)
    else:
        x = torch.empty((0, feature_dim), dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index)

class GNNModel:
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=16, epochs=1, lr=0.01):
        self.gnn = GraphSAGE(in_channels, hidden_channels, out_channels)
        self.epochs = epochs
        self.lr = lr
        self.pyg_data = []
        self.wl_kernel_matrix = None
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Fixed device selection
        # Alternative: Force CPU
        # self.device = torch.device('cpu')
        print("Using: ", self.device)
        self.gnn.to(self.device)
        self.multilayer_network = None

        self.pred_head = nn.Linear(out_channels, 1).to(self.device)
        self.att_pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )).to(self.device)

    def convert_to_pyg(self, multilayer_network, features):
        """
        Convert all network snapshots to PyG data objects
        
        Args:
            multilayer_network: MultilayerNetwork instance
            features: Dictionary of features from FeatureExtractor
        """
        self.multilayer_network = multilayer_network
        snapshots = multilayer_network.networks
        
        # Determine feature dimension based on first snapshot with features
        feature_dim = 5  # Default
        for i, snapshot in enumerate(snapshots):
            if i in features and features[i]:
                # Get first node's feature vector length
                first_node = next(iter(features[i]))
                feature_dim = len(features[i][first_node])
                break
        
        print(f"Using feature dimension: {feature_dim}")
        
        # Convert snapshots to PyG data
        self.pyg_data = Parallel(n_jobs=-1)(
            delayed(convert_snapshot_to_pyg)(snapshot, features[i], feature_dim)
            for i, snapshot in enumerate(snapshots) if i in features
        )
        
        print(f"Converted {len(self.pyg_data)} snapshots to PyG format.")

    def sample_negative_edges(self, edge_index, num_nodes, num_samples):
        pos_set = set()
        edge_index_np = edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            u, v = edge_index_np[0, i], edge_index_np[1, i]
            pos_set.add((u, v))
            pos_set.add((v, u))
        neg_edges = []
        while len(neg_edges) < num_samples:
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u == v or (u, v) in pos_set:
                continue
            neg_edges.append((u, v))
        return torch.tensor(neg_edges, dtype=torch.long).to(self.device).t().contiguous()

    def train_gnn(self, y_labels):
        self.gnn.train()
        self.pred_head.train()
        self.att_pool.train()

        optimizer = torch.optim.Adam(
            list(self.gnn.parameters()) + 
            list(self.pred_head.parameters()) + 
            list(self.att_pool.parameters()), 
            lr=self.lr
        )

        for epoch in range(self.epochs):
            total_loss = 0
            for i, data_obj in enumerate(self.pyg_data):
                data_obj = data_obj.to(self.device)
                optimizer.zero_grad()

                out = self.gnn(data_obj.x, data_obj.edge_index)
                batch = torch.zeros(out.size(0), dtype=torch.long, device=out.device)
                graph_embedding = self.att_pool(out, batch)
                y_pred = self.pred_head(graph_embedding)
                y_true = torch.tensor([y_labels[i]], dtype=torch.float32, device=self.device)

                loss = F.mse_loss(y_pred.squeeze(), y_true)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                print("training", i)

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(self.pyg_data):.4f}")

    @torch.no_grad()
    def predict(self):
        self.gnn.eval()
        self.pred_head.eval()
        self.att_pool.eval()
        preds = []
        for data_obj in self.pyg_data:
            data_obj = data_obj.to(self.device)
            out = self.gnn(data_obj.x, data_obj.edge_index)
            batch = torch.zeros(out.size(0), dtype=torch.long, device=out.device)
            graph_embedding = self.att_pool(out, batch)
            y_pred = self.pred_head(graph_embedding)
            preds.append(y_pred.item())
        return preds

    @torch.no_grad()
    def get_embeddings(self):
        self.gnn.eval()
        self.att_pool.eval()
        embeddings = []
        for data_obj in self.pyg_data:
            data_obj = data_obj.to(self.device)
            out = self.gnn(data_obj.x, data_obj.edge_index)
            batch = torch.zeros(out.size(0), dtype=torch.long, device=out.device)
            graph_embedding = self.att_pool(out, batch)
            embeddings.append(graph_embedding.squeeze(0).cpu().numpy())
        return np.array(embeddings)

    def compute_wl_kernel(self, n_iter=3, normalize=True):
        if self.multilayer_network is None:
            print("No multilayer network available for WL kernel computation.")
            return

        graphs = []
        for snapshot in self.multilayer_network.networks:
            G_combined = nx.Graph()
            for sector, G_layer in snapshot['layers'].items():
                G_combined.add_nodes_from(G_layer.nodes())
                G_combined.add_edges_from(G_layer.edges())
            for _, t1, _, t2, _ in snapshot['inter_edges']:
                G_combined.add_edge(t1, t2)

            nodes = list(G_combined.nodes())
            node_id_map = {node: i for i, node in enumerate(nodes)}
            
            # Meaningful node labels
            deg_centrality = nx.degree_centrality(G_combined)
            node_labels = {}
            for node in nodes:
                sector_label = node.split("_")[0] if "_" in node else "unknown"
                d = deg_centrality.get(node, 0)
                bin_label = str(int(d * 5))
                node_labels[node_id_map[node]] = f"{sector_label}_{bin_label}"

            edges = [(node_id_map[u], node_id_map[v]) for u, v in G_combined.edges()]

            try:
                g = grakelGraph(edges, node_labels=node_labels)
                graphs.append(g)
            except Exception as e:
                print(f"Graph creation failed for snapshot {snapshot['window_start']}: {e}")

        if len(graphs) == 0:
            raise ValueError("All graphs failed. WL kernel cannot be computed.")

        wl = GraphKernel(kernel={"name": "weisfeiler_lehman", "n_iter": n_iter}, normalize=normalize)
        self.wl_kernel_matrix = wl.fit_transform(graphs)
        print("WL kernel computed:", self.wl_kernel_matrix.shape)
