import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
import os
import json
from datetime import datetime
import seaborn as sns
from pathlib import Path

class TrainingVisualizer:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.plots_dir = Path(log_dir) / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def log_metrics(self, metrics, step):
        """Log metrics to tensorboard"""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
    
    def plot_loss_curves(self, train_losses, val_losses, epochs):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(self.plots_dir / 'loss_curves.png')
        plt.close()
    
    def visualize_embeddings(self, embeddings, labels, n_samples=1000):
        """Create t-SNE visualization of embeddings with triplet-specific coloring"""
        # Sample if too many points
        if len(embeddings) > n_samples:
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Check if labels represent triplet categories (0=anchor, 1=positive, 2=negative)
        unique_labels = np.unique(labels)
        if len(unique_labels) == 3 and set(unique_labels) == {0, 1, 2}:
            # Triplet-specific visualization with meaningful colors
            plt.figure(figsize=(12, 10))
            colors = ['red', 'green', 'blue']
            category_names = ['Anchor', 'Positive', 'Negative']
            
            for i, (color, name) in enumerate(zip(colors, category_names)):
                mask = labels == i
                if np.any(mask):
                    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                              c=color, label=name, alpha=0.7, s=30)
            
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.title('t-SNE Visualization of Embeddings\n(Anchor-Positive-Negative)')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
        else:
            # Generic visualization for other label types
            plt.figure(figsize=(10, 10))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10')
            plt.colorbar(scatter)
            plt.title('t-SNE visualization of embeddings')
        
        plt.savefig(self.plots_dir / 'embedding_space.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_example_triplets(self, anchors, positives, negatives, n_examples=5):
        """Plot example triplets with their distances"""
        fig, axes = plt.subplots(n_examples, 3, figsize=(15, 3*n_examples))
        for i in range(n_examples):
            axes[i, 0].imshow(anchors[i].permute(1, 2, 0))
            axes[i, 0].set_title('Anchor')
            axes[i, 1].imshow(positives[i].permute(1, 2, 0))
            axes[i, 1].set_title('Positive')
            axes[i, 2].imshow(negatives[i].permute(1, 2, 0))
            axes[i, 2].set_title('Negative')
            
            for ax in axes[i]:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'example_triplets.png')
        plt.close()
    
    def save_final_report(self, training_stats):
        """Save final training report as JSON"""
        report_path = Path(self.log_dir) / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(training_stats, f, indent=4)
    
    def create_html_report(self):
        """Create an HTML report combining all visualizations"""
        html_content = f"""
        <html>
        <head>
            <title>Training Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .plot {{ margin: 20px 0; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Training Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}</h1>
            
            <div class="plot">
                <h2>Loss Curves</h2>
                <img src="plots/loss_curves.png">
            </div>
            
            <div class="plot">
                <h2>Embedding Space Visualization</h2>
                <img src="plots/embedding_space.png">
            </div>
            
            <div class="plot">
                <h2>Example Triplets</h2>
                <img src="plots/example_triplets.png">
            </div>
        </body>
        </html>
        """
        
        with open(Path(self.log_dir) / 'results.html', 'w') as f:
            f.write(html_content)
    
    def close(self):
        """Close the tensorboard writer"""
        self.writer.close() 