import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

class SystemicRiskPredictor:
    def __init__(self,kernel_matrix,network):
        self.kernel_matrix = kernel_matrix
        self.network = network
        self.svm = SVR(kernel='precomputed')
        self.debtrank = None

    def compute_debtrank(self):
        """
        Compute DebtRank for each snapshot as a systemic risk measure.
        
        DebtRank is computed using a weighted combination of:
        1. Network centrality (degree, eigenvector)
        2. Interconnectedness (clustering)
        3. Market volatility
        """
        debtrank_values = []

        for idx, measures in enumerate(self.network.centrality_measures):
            # Extract all centrality metrics across sectors
            degree_vals = []
            eigenvector_vals = []
            clustering_vals = []
            
            # Collect centrality metrics from all sectors
            for sector, cent_dict in measures.items():
                degree_vals.extend(list(cent_dict.get('degree', {}).values()))
                eigenvector_vals.extend(list(cent_dict.get('eigenvector', {}).values()))
                clustering_vals.extend(list(cent_dict.get('clustering', {}).values()))
            
            # Calculate statistics for each centrality measure
            avg_degree = np.mean(degree_vals) if degree_vals else 0
            avg_eigenvector = np.mean(eigenvector_vals) if eigenvector_vals else 0
            avg_clustering = np.mean(clustering_vals) if clustering_vals else 0
            
            # Calculate interconnectedness as number of inter-edges divided by 
            # the total possible inter-edges (as an approximation)
            snapshot = self.network.networks[idx]
            num_inter_edges = len(snapshot['inter_edges'])
            total_nodes = sum(len(G.nodes()) for G in snapshot['layers'].values())
            inter_connectivity = num_inter_edges / total_nodes if total_nodes > 0 else 0
            
            # Weight the different components to compute DebtRank
            # Increased weight on eigenvector centrality as it better captures systemic importance
            debtrank = (0.3 * avg_degree + 
                        0.4 * avg_eigenvector + 
                        0.1 * avg_clustering + 
                        0.2 * inter_connectivity)
            
            debtrank_values.append(debtrank)
        
        self.debtrank = np.array(debtrank_values)
        print(f"Computed DebtRank for {len(debtrank_values)} snapshots.")
        return self.debtrank

    def train_svm(self, y_train, test_size=0.2):
        """
        Train SVR kernel using precomputed kernel matrix and debtrank values
        with proper train/test split.
        
        Args:
            y_train: Target debtrank values
            test_size: Proportion of data to use for testing (default=0.2)
        """
        if self.debtrank is None:
            self.compute_debtrank()
        
        if self.kernel_matrix.shape[0] != len(self.debtrank):
            raise ValueError(f"Kernel matrix shape {self.kernel_matrix.shape[0]} and debtrank length {len(self.debtrank)} mismatch.")
        
        # Split data into training and testing sets
        n_samples = len(self.debtrank)
        n_test = int(test_size * n_samples)
        n_train = n_samples - n_test
        
        # Indices for train/test split
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Extract train/test kernel matrices and targets
        self.train_indices = train_indices
        self.test_indices = test_indices
        
        X_train = self.kernel_matrix[np.ix_(train_indices, train_indices)]
        y_train = self.debtrank[train_indices]
        
        # Train the model
        self.svm.fit(X_train, y_train)
        print(f"Trained SVM model with {n_train} samples.")

    def predict(self, kernel_matrix_new=None):
        """
        PREDICTs systemic risk 
        if a new kernel marix is provided, use it
        else use the original kernel matrix
        """
        if kernel_matrix_new is not None:
            predictions = self.svm.predict(kernel_matrix_new)
        else:
            predictions = self.svm.predict(self.kernel_matrix)
        return predictions
    
    def evaluate(self):
        """
        Evaluate the SVM model on the test set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not hasattr(self, 'train_indices') or not hasattr(self, 'test_indices'):
            print("No train/test split found. Evaluating on training data.")
            predictions = self.predict()
            rmse = np.sqrt(mean_squared_error(self.debtrank, predictions))
            spearman_corr, _ = spearmanr(self.debtrank, predictions)
        else:
            # Extract test kernel matrix
            n_train = len(self.train_indices)
            n_test = len(self.test_indices)
            
            # For kernel methods, we need the kernel between test and training points
            K_test_train = self.kernel_matrix[np.ix_(self.test_indices, self.train_indices)]
            
            # Make predictions
            predictions = self.svm.predict(K_test_train)
            
            # Evaluate
            y_test = self.debtrank[self.test_indices]
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            spearman_corr, _ = spearmanr(y_test, predictions)
        
        eval_metrics = {
            'RMSE': rmse,
            'Spearman Correlation': spearman_corr
        }
        print("Evaluation metrics:", eval_metrics)
        return eval_metrics