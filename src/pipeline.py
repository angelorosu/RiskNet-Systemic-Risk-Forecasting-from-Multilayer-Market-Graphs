# pipeline.py
import os
from src.data_manager import DataManager
from src.multilayer_network import MultilayerNetwork
from src.feature_extractor import FeatureExtractor
from src.gnn_model import GNNModel
from src.systemic_risk_predictor import SystemicRiskPredictor
from src.visualizer import Visualizer

class Pipeline:
    def __init__(self, data_directory):
        # Initialize DataManager with the directory containing your CSVs.
        self.data_manager = DataManager(data_directory)
        # Other components will be initialized after data is loaded.
        self.network = None
        self.feature_extractor = None
        self.gnn_model = None
        self.predictor = None
        self.visualizer = None

    def run(self):
        # Step 1: Run the DataManager pipeline to load and preprocess data.
        print("Running DataManager...")
        self.data_manager.run_pipeline()
        data = self.data_manager.get_data()

        print("Data loaded and processed:")
        print("Raw Data:", data['raw_data'].head())
        print("Log Returns:", data['log_returns'].head())
        print("Sector Mapping:", data['sector_mapping'].head())
        print("Training Data:", data['training_data'].head())
        print("Test Data:", data['test_data'].head())
        print("Tickers:", data['tickers'])

        # Step 2: Construct the multilayer network.
        print("Building multilayer network...")
        self.network = MultilayerNetwork(
            log_returns=data['log_returns'],
            sector_mapping=data['sector_mapping'],
            window_size=200,
            threshold=0.5,
            inter_threshold=0.3
        )
        if os.path.exists("results/snapshots.pkl"):
            print("Loading existing network snapshots...")
            self.network.load_snapshots("results")
        else:
            print("Creating new network snapshots...")
            self.network.build_network()
            self.network.save_snapshots("results")
        print(f"Built {len(self.network.networks)} network snapshots.")

        # Step 3: Feature Extraction.
        print("Extracting features from network snapshots...")
        self.feature_extractor = FeatureExtractor(self.network)
        self.feature_extractor.compute_centrality()
        self.feature_extractor.compute_financial_indicators()
        self.feature_extractor.combine_features()
        print(f"Extracted features for {len(self.feature_extractor.features)} snapshots.")
        if 0 in self.feature_extractor.features:
            print("Features for snapshot 0:")
            print(self.feature_extractor.features[0].head())
        
        # Step 4: Initialize predictor and compute DebtRank.
        print("Computing DebtRank for each snapshot...")
        # Initialize predictor with a dummy kernel (None) for now.
        self.predictor = SystemicRiskPredictor(None, self.network)
        self.predictor.compute_debtrank()
        # Now self.predictor.debtrank contains the target values (a list or tensor of length = number of snapshots).
        print("Computed DebtRank for", len(self.predictor.debtrank), "snapshots.")

        # Step 5: Train the GNN.
        print("Initializing and training GNN model...")
        self.gnn_model = GNNModel(in_channels=5, hidden_channels=16, out_channels=16, epochs=1, lr=0.01)
        self.gnn_model.convert_to_pyg(self.network, self.feature_extractor.features)
        self.gnn_model.train_gnn(self.predictor.debtrank)

        print("Performing systemic risk prediction using kernel SVM...")
        gnn_embeddings = self.gnn_model.get_embeddings()
        rbf_kernel_matrix = rbf_kernel(gnn_embeddings, gamma=1.0)

        # Step 6: Compute WL kernel.
        print("Computing WL kernel...")
        self.gnn_model.compute_wl_kernel()
        print("WL kernel computation complete.")

        # Step 7: Train kernel SVM for systemic risk prediction.
        print("Performing systemic risk prediction using kernel SVM...")
        from sklearn.metrics.pairwise import rbf_kernel
        import numpy as np

        # Ensure gnn_model is trained and gnn_model.get_embeddings() is available
        gnn_embeddings = self.gnn_model.get_embeddings()
        rbf_kernel_matrix = rbf_kernel(gnn_embeddings, gamma=1.0)

        # Re-initialize predictor with the computed RBF kernel
        predictor_rbf = SystemicRiskPredictor(rbf_kernel_matrix, self.network)
        predictor_rbf.compute_debtrank()
        predictor_rbf.train_svm(predictor_rbf.debtrank)
        evaluation_metrics_rbf = predictor_rbf.evaluate()
        print("Evaluation metrics for RBF kernel SVM:",evaluation_metrics_rbf)

        # Step 7: Train kernel SVM for systemic risk prediction.
        print("Performing systemic risk prediction using WL kernel...")
        # Initialize predictor with the computed WL kernel
        predictor_wl = SystemicRiskPredictor(self.gnn_model.wl_kernel_matrix, self.network)
        predictor_wl.compute_debtrank()
        predictor_wl.train_svm(predictor_wl.debtrank)
        evaluation_metrics_wl = predictor_wl.evaluate()
        print("Evaluation metrics for WL kernel SVM:", evaluation_metrics_wl)

        # Use the best predictor for visualization
        self.predictor = predictor_wl if evaluation_metrics_wl['RMSE'] < evaluation_metrics_rbf['RMSE'] else predictor_rbf

        # Step 8: Visualization.
        print("Visualizing results...")
        predictions = self.predictor.predict()
        actuals = self.predictor.debtrank
        self.visualizer = Visualizer(self.network, predictions, actuals)
        self.visualizer.plot_network(snapshot_index=0)
        self.visualizer.plot_connectivity()
        self.visualizer.plot_predictions()

        print("Pipeline executed successfully.")

if __name__ == '__main__':
    data_directory = "raw_data"  # Update to your actual data directory.
    pipeline = Pipeline(data_directory)
    pipeline.run()
