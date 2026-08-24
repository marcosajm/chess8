#!/usr/bin/env python3
"""
NNUE Data Visualization Tool
Creates colorful plots to visualize the evaluation data from NNUE training data generator
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import os
import argparse
from collections import defaultdict

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class NNUEVisualizer:
    def __init__(self, data_file):
        """
        Initialize visualizer with a NNUE data file.
        
        Args:
            data_file: Path to the NNUE training data file (.bin)
        """
        self.data_file = data_file
        self.positions = []
        self.features = []
        self.scores = []
        self.results = []
        self.tactical_scores = []
        self.load_data()
    
    def load_data(self):
        """Load and parse the NNUE training data file."""
        print(f"📊 Loading data from: {self.data_file}")
        
        try:
            with open(self.data_file, 'rb') as f:
                # Read header
                header = f.read(8)
                if len(header) < 8:
                    print("❌ Invalid file format: Header too short")
                    return
                
                magic = header[:4].decode('utf-8', errors='ignore')
                num_positions = struct.unpack('I', header[4:8])[0]
                
                print(f"  Magic: {magic}, Positions: {num_positions}")
                
                # Read all positions
                for i in range(num_positions):
                    # Read features (780 floats = 3120 bytes)
                    feature_bytes = f.read(780 * 4)
                    if len(feature_bytes) < 780 * 4:
                        break
                    
                    features = np.frombuffer(feature_bytes, dtype=np.float32).copy()
                    
                    # Read score, result, tactical_score (3 floats = 12 bytes)
                    score_data = f.read(12)
                    if len(score_data) < 12:
                        break
                    
                    score, result, tactical_score = struct.unpack('fff', score_data)
                    
                    self.features.append(features)
                    self.scores.append(score)
                    self.results.append(result)
                    self.tactical_scores.append(tactical_score)
                    
                    if (i + 1) % 50000 == 0:
                        print(f"  Loaded {i+1} positions...")
            
            print(f"✅ Loaded {len(self.scores)} positions successfully")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def create_dashboard(self, output_dir='nnue_visualizations'):
        """Create a comprehensive dashboard with multiple plots."""
        if not self.scores:
            print("❌ No data loaded!")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n🎨 Creating visualization dashboard...")
        
        # 1. Score Distribution
        self.plot_score_distribution(output_dir)
        
        # 2. Score by Game Result
        self.plot_score_by_result(output_dir)
        
        # 3. Score Evolution (if sequential)
        self.plot_score_evolution(output_dir)
        
        # 4. Feature Heatmap (PCA visualization)
        self.plot_feature_heatmap(output_dir)
        
        # 5. Tactical Score vs Evaluation
        self.plot_tactical_vs_evaluation(output_dir)
        
        # 6. Result Distribution
        self.plot_result_distribution(output_dir)
        
        # 7. Score Histogram with Statistics
        self.plot_detailed_score_stats(output_dir)
        
        # 8. 3D Scatter Plot (Score, Result, Tactical)
        self.plot_3d_analysis(output_dir)
        
        # 9. Feature Importance (top features)
        self.plot_feature_importance(output_dir)
        
        # 10. Correlation Matrix
        self.plot_correlation_matrix(output_dir)
        
        print(f"\n✅ All visualizations saved to: {output_dir}/")
        print("📁 Files created:")
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                print(f"  - {f}")
    
    def plot_score_distribution(self, output_dir):
        """Plot distribution of evaluation scores."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        axes[0].hist(self.scores, bins=50, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0].axvline(np.mean(self.scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.scores):.3f}')
        axes[0].axvline(np.median(self.scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(self.scores):.3f}')
        axes[0].set_xlabel('Evaluation Score (centipawns/100)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Score Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot with statistics
        box = axes[1].boxplot(self.scores, patch_artist=True, 
                              boxprops=dict(facecolor='lightblue', color='navy'),
                              whiskerprops=dict(color='navy'),
                              capprops=dict(color='navy'),
                              medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('Evaluation Score', fontsize=12)
        axes[1].set_title('Score Distribution Summary', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Min: {np.min(self.scores):.3f}\n'
        stats_text += f'Max: {np.max(self.scores):.3f}\n'
        stats_text += f'Mean: {np.mean(self.scores):.3f}\n'
        stats_text += f'Std: {np.std(self.scores):.3f}\n'
        stats_text += f'Q1: {np.percentile(self.scores, 25):.3f}\n'
        stats_text += f'Q3: {np.percentile(self.scores, 75):.3f}'
        axes[1].text(1.1, 0.5, stats_text, transform=axes[1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10, verticalalignment='center')
        
        plt.suptitle('NNUE Evaluation Score Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/1_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_score_by_result(self, output_dir):
        """Plot score distribution by game result."""
        results = ['Loss (0)', 'Draw (0.5)', 'Win (1.0)']
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plots by result
        data_by_result = []
        for result in [0.0, 0.5, 1.0]:
            scores = [s for s, r in zip(self.scores, self.results) if abs(r - result) < 0.01]
            data_by_result.append(scores)
        
        box_positions = [0, 1, 2]
        bp = axes[0].boxplot(data_by_result, positions=box_positions, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', color='navy'),
                            whiskerprops=dict(color='navy'),
                            capprops=dict(color='navy'),
                            medianprops=dict(color='red', linewidth=2))
        
        # Color the boxes
        for box, color in zip(bp['boxes'], colors):
            box.set_facecolor(color)
            box.set_alpha(0.7)
        
        axes[0].set_xticks(box_positions)
        axes[0].set_xticklabels(results)
        axes[0].set_ylabel('Evaluation Score', fontsize=12)
        axes[0].set_title('Scores by Game Result', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Violin plots (more detailed)
        for i, (data, color, label) in enumerate(zip(data_by_result, colors, results)):
            parts = axes[1].violinplot(data, positions=[i], showmeans=False, showmedians=True)
            parts['bodies'][0].set_facecolor(color)
            parts['bodies'][0].set_alpha(0.7)
        
        axes[1].set_xticks(box_positions)
        axes[1].set_xticklabels(results)
        axes[1].set_ylabel('Evaluation Score', fontsize=12)
        axes[1].set_title('Score Distribution by Result (Violin Plot)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.suptitle('Game Result vs Evaluation Score', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/2_score_by_result.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_score_evolution(self, output_dir):
        """Plot score evolution over positions (if ordered)."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Overall evolution
        axes[0, 0].plot(self.scores[:5000], alpha=0.6, linewidth=0.5, color='blue')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        # Moving average
        window = min(100, len(self.scores) // 100)
        if window > 1:
            moving_avg = np.convolve(self.scores, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.scores[:5000])), 
                           moving_avg[:len(self.scores[:5000])-window+1], 
                           color='red', linewidth=2, label='Moving Average')
        axes[0, 0].set_xlabel('Position Index', fontsize=12)
        axes[0, 0].set_ylabel('Score', fontsize=12)
        axes[0, 0].set_title('Score Evolution', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Score by position (first 1000)
        axes[0, 1].scatter(range(min(1000, len(self.scores))), 
                          self.scores[:min(1000, len(self.scores))], 
                          c=self.scores[:min(1000, len(self.scores))], 
                          cmap='RdYlGn', s=5, alpha=0.6)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Position Index', fontsize=12)
        axes[0, 1].set_ylabel('Score', fontsize=12)
        axes[0, 1].set_title('Score Scatter (First 1000 positions)', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram with density
        axes[1, 0].hist(self.scores, bins=60, density=True, alpha=0.7, color='skyblue', edgecolor='navy')
        axes[1, 0].axvline(np.mean(self.scores), color='red', linestyle='--', linewidth=2, label='Mean')
        axes[1, 0].axvline(np.median(self.scores), color='green', linestyle='--', linewidth=2, label='Median')
        # KDE
        from scipy import stats
        kde = stats.gaussian_kde(self.scores)
        x_range = np.linspace(min(self.scores), max(self.scores), 200)
        axes[1, 0].plot(x_range, kde(x_range), color='darkblue', linewidth=2, label='KDE')
        axes[1, 0].set_xlabel('Score', fontsize=12)
        axes[1, 0].set_ylabel('Density', fontsize=12)
        axes[1, 0].set_title('Score Distribution with KDE', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_scores = np.sort(self.scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[1, 1].plot(sorted_scores, cumulative, color='blue', linewidth=2)
        axes[1, 1].axvline(np.mean(self.scores), color='red', linestyle='--', alpha=0.5, label='Mean')
        axes[1, 1].axvline(np.median(self.scores), color='green', linestyle='--', alpha=0.5, label='Median')
        axes[1, 1].set_xlabel('Score', fontsize=12)
        axes[1, 1].set_ylabel('Cumulative Probability', fontsize=12)
        axes[1, 1].set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Score Evolution and Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/3_score_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_heatmap(self, output_dir):
        """Plot feature heatmap using PCA or random subset."""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Sample data if too large
            n_samples = min(1000, len(self.features))
            indices = np.random.choice(len(self.features), n_samples, replace=False)
            sampled_features = np.array([self.features[i] for i in indices])
            
            # Standardize
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(sampled_features)
            
            # PCA
            pca = PCA(n_components=50)
            features_pca = pca.fit_transform(features_scaled)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # PCA explained variance
            axes[0, 0].bar(range(1, 21), pca.explained_variance_ratio_[:20] * 100, 
                          color='skyblue', edgecolor='navy')
            axes[0, 0].set_xlabel('Principal Component', fontsize=12)
            axes[0, 0].set_ylabel('Explained Variance (%)', fontsize=12)
            axes[0, 0].set_title('PCA Explained Variance (Top 20)', fontsize=14, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Cumulative variance
            cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100
            axes[0, 1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                           color='blue', linewidth=2)
            axes[0, 1].axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90%')
            axes[0, 1].axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95%')
            axes[0, 1].set_xlabel('Number of Components', fontsize=12)
            axes[0, 1].set_ylabel('Cumulative Variance (%)', fontsize=12)
            axes[0, 1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 2D PCA scatter colored by score
            scatter = axes[1, 0].scatter(features_pca[:, 0], features_pca[:, 1], 
                                       c=[self.scores[i] for i in indices],
                                       cmap='RdYlGn', s=20, alpha=0.6)
            axes[1, 0].set_xlabel('PC1', fontsize=12)
            axes[1, 0].set_ylabel('PC2', fontsize=12)
            axes[1, 0].set_title('PCA 2D Projection (colored by score)', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=axes[1, 0], label='Score')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 2D PCA colored by result
            result_colors = ['#ff6b6b' if r < 0.01 else '#ffd93d' if r < 0.99 else '#6bcf7f' 
                           for r in [self.results[i] for i in indices]]
            axes[1, 1].scatter(features_pca[:, 0], features_pca[:, 1], 
                              c=result_colors, s=20, alpha=0.6)
            axes[1, 1].set_xlabel('PC1', fontsize=12)
            axes[1, 1].set_ylabel('PC2', fontsize=12)
            axes[1, 1].set_title('PCA 2D Projection (colored by result)', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#ff6b6b', alpha=0.6, label='Loss'),
                Patch(facecolor='#ffd93d', alpha=0.6, label='Draw'),
                Patch(facecolor='#6bcf7f', alpha=0.6, label='Win')
            ]
            axes[1, 1].legend(handles=legend_elements)
            
            plt.suptitle('Feature Space Analysis (PCA)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/4_feature_pca.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Feature heatmap could not be created: {e}")
            # Create a simple heatmap of feature correlations
            self._create_simple_feature_heatmap(output_dir)
    
    def _create_simple_feature_heatmap(self, output_dir):
        """Create a simple heatmap of feature correlations."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Sample features for correlation matrix
        n_samples = min(500, len(self.features))
        indices = np.random.choice(len(self.features), n_samples, replace=False)
        sampled_features = np.array([self.features[i] for i in indices])
        
        # Correlation of first 50 features (to keep matrix manageable)
        corr_matrix = np.corrcoef(sampled_features[:, :50].T)
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title('Feature Correlation Matrix (First 50 features)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature Index', fontsize=12)
        ax.set_ylabel('Feature Index', fontsize=12)
        plt.colorbar(im, ax=ax, label='Correlation')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/4_feature_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_tactical_vs_evaluation(self, output_dir):
        """Plot tactical scores vs evaluation scores."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        scatter = axes[0].scatter(self.tactical_scores, self.scores, 
                                 c=self.results, cmap='RdYlGn', s=5, alpha=0.5)
        axes[0].set_xlabel('Tactical Score', fontsize=12)
        axes[0].set_ylabel('Evaluation Score', fontsize=12)
        axes[0].set_title('Tactical Score vs Evaluation', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label='Result')
        
        # Hexbin plot for density
        hb = axes[1].hexbin(self.tactical_scores, self.scores, gridsize=40, cmap='viridis', mincnt=1)
        axes[1].set_xlabel('Tactical Score', fontsize=12)
        axes[1].set_ylabel('Evaluation Score', fontsize=12)
        axes[1].set_title('Tactical Score vs Evaluation (Density)', fontsize=14, fontweight='bold')
        plt.colorbar(hb, ax=axes[1], label='Count')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Tactical Awareness Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/5_tactical_vs_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_result_distribution(self, output_dir):
        """Plot distribution of game results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count results
        losses = sum(1 for r in self.results if r < 0.01)
        draws = sum(1 for r in self.results if 0.49 <= r <= 0.51)
        wins = sum(1 for r in self.results if r > 0.99)
        
        labels = ['Loss (0)', 'Draw (0.5)', 'Win (1.0)']
        counts = [losses, draws, wins]
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
        
        # Pie chart
        axes[0].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
                   explode=(0.05, 0.05, 0.05), shadow=True)
        axes[0].set_title('Game Result Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = axes[1].bar(labels, counts, color=colors, edgecolor='navy', linewidth=2)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Game Result Counts', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Game Results Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/6_result_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_detailed_score_stats(self, output_dir):
        """Plot detailed statistics of scores."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a more detailed histogram with bins
        n, bins, patches = ax.hist(self.scores, bins=80, color='skyblue', 
                                   edgecolor='navy', alpha=0.7, density=True)
        
        # Color the bins based on value
        for patch, value in zip(patches, bins[:-1]):
            if value < 0:
                patch.set_facecolor('#ff6b6b')
                patch.set_alpha(0.7)
            else:
                patch.set_facecolor('#6bcf7f')
                patch.set_alpha(0.7)
        
        # Statistics
        mean_val = np.mean(self.scores)
        median_val = np.median(self.scores)
        std_val = np.std(self.scores)
        
        # Add vertical lines for statistics
        ax.axvline(mean_val, color='red', linestyle='-', linewidth=3, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='purple', linestyle='--', linewidth=3, label=f'Median: {median_val:.3f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'±1σ: {mean_val-std_val:.3f}, {mean_val+std_val:.3f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        # Add text box with stats
        stats_text = f'Total Positions: {len(self.scores):,}\n'
        stats_text += f'Mean: {mean_val:.4f}\n'
        stats_text += f'Median: {median_val:.4f}\n'
        stats_text += f'Std Dev: {std_val:.4f}\n'
        stats_text += f'Skewness: {stats.skew(self.scores):.4f}\n'
        stats_text += f'Kurtosis: {stats.kurtosis(self.scores):.4f}\n'
        stats_text += f'Min: {np.min(self.scores):.4f}\n'
        stats_text += f'Max: {np.max(self.scores):.4f}'
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, verticalalignment='top', horizontalalignment='right')
        
        ax.set_xlabel('Evaluation Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Detailed Score Statistics', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/7_detailed_stats.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_3d_analysis(self, output_dir):
        """Create 3D scatter plot analysis."""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Sample data
            n_samples = min(2000, len(self.scores))
            indices = np.random.choice(len(self.scores), n_samples, replace=False)
            
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 3D scatter
            scatter = ax.scatter([self.scores[i] for i in indices],
                                [self.tactical_scores[i] for i in indices],
                                [self.results[i] for i in indices],
                                c=[self.scores[i] for i in indices],
                                cmap='RdYlGn', s=15, alpha=0.6)
            
            ax.set_xlabel('Evaluation Score', fontsize=12, labelpad=10)
            ax.set_ylabel('Tactical Score', fontsize=12, labelpad=10)
            ax.set_zlabel('Result', fontsize=12, labelpad=10)
            ax.set_title('3D Analysis: Score, Tactical, Result', fontsize=16, fontweight='bold')
            
            plt.colorbar(scatter, ax=ax, label='Evaluation Score')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/8_3d_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  3D plot could not be created: {e}")
    
    def plot_feature_importance(self, output_dir):
        """Plot feature importance using correlation with scores."""
        try:
            # Sample data
            n_samples = min(2000, len(self.features))
            indices = np.random.choice(len(self.features), n_samples, replace=False)
            
            features_matrix = np.array([self.features[i] for i in indices])
            scores_vector = np.array([self.scores[i] for i in indices])
            
            # Calculate correlation between each feature and score
            correlations = []
            for i in range(features_matrix.shape[1]):
                corr = np.corrcoef(features_matrix[:, i], scores_vector)[0, 1]
                if not np.isnan(corr):
                    correlations.append((i, abs(corr)))
                else:
                    correlations.append((i, 0))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_n = min(30, len(correlations))
            top_features = correlations[:top_n]
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Top features by correlation
            feature_indices = [f[0] for f in top_features]
            feature_correlations = [f[1] for f in top_features]
            
            colors = ['#ff6b6b' if c < 0 else '#6bcf7f' for c in feature_correlations]
            bars = axes[0].barh(range(len(feature_indices)), feature_correlations, 
                               color=plt.cm.viridis(np.array(feature_correlations)))
            axes[0].set_yticks(range(len(feature_indices)))
            axes[0].set_yticklabels([f'F{i}' for i in feature_indices])
            axes[0].set_xlabel('Absolute Correlation with Score', fontsize=12)
            axes[0].set_title('Top 30 Most Important Features', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Cumulative importance
            cumulative = np.cumsum(sorted([c[1] for c in correlations], reverse=True))
            axes[1].plot(range(1, len(cumulative) + 1), cumulative / cumulative[-1] * 100,
                        color='blue', linewidth=2)
            axes[1].axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90%')
            axes[1].axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95%')
            axes[1].set_xlabel('Number of Features', fontsize=12)
            axes[1].set_ylabel('Cumulative Importance (%)', fontsize=12)
            axes[1].set_title('Feature Cumulative Importance', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/9_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Feature importance plot could not be created: {e}")
    
    def plot_correlation_matrix(self, output_dir):
        """Plot correlation matrix between score, tactical, and result."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create correlation matrix
        data = np.array([self.scores, self.tactical_scores, self.results])
        corr_matrix = np.corrcoef(data)
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Add labels
        labels = ['Score', 'Tactical', 'Result']
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                              ha='center', va='center', color='black' if abs(corr_matrix[i, j]) < 0.5 else 'white',
                              fontweight='bold')
        
        ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Correlation')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/10_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize NNUE training data')
    parser.add_argument('--file', '-f', type=str, 
                       help='Path to NNUE data file (.bin)',
                       default='training_data_prod.bin')
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory for visualizations',
                       default='nnue_visualizations')
    
    args = parser.parse_args()
    
    print("🎨 NNUE Data Visualization Tool")
    print("=" * 50)
    
    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        print("📁 Available .bin files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.bin'):
                print(f"  - {f}")
        return
    
    try:
        visualizer = NNUEVisualizer(args.file)
        visualizer.create_dashboard(args.output)
        print(f"\n✨ Visualizations complete!")
        print(f"📁 Check the '{args.output}' directory for all plots.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()