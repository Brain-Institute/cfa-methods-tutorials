"""
Validation Study Visualizer

Step-by-step matplotlib visualization of validation study generation.
Shows data generation at each stage with explanatory text for verification.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Dict, Optional, List
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import pdist, squareform


class ValidationVisualizer:
    """
    Step-by-step visualization of validation study generation.

    Creates matplotlib figures at each generation stage to help users
    verify that synthetic data is being created correctly.

    Example:
        >>> from imbiotype.validation import ValidationStudyConfig, ValidationVisualizer
        >>> config = ValidationStudyConfig(n_samples=200, visualize=True)
        >>> visualizer = ValidationVisualizer(config)
        >>> visualizer.show_stage1_mixture_model(data)
    """

    # Color palette for clusters (colorblind-friendly)
    CLUSTER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    def __init__(self, config, block: bool = True):
        """
        Initialize visualizer.

        Args:
            config: ValidationStudyConfig with generation parameters
            block: Whether plt.show() blocks execution
        """
        self.config = config
        self.block = block
        self.stage_count = 0

    def _get_cluster_colors(self, labels: np.ndarray) -> List[str]:
        """Get color for each sample based on cluster label."""
        return [self.CLUSTER_COLORS[l % len(self.CLUSTER_COLORS)] for l in labels]

    def _create_figure(self, title: str, nrows: int, ncols: int,
                       figsize: Optional[tuple] = None) -> tuple:
        """Create figure with suptitle."""
        self.stage_count += 1
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows + 0.5)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.suptitle(f"Stage {self.stage_count}: {title}",
                     fontsize=14, fontweight='bold', y=0.98)
        return fig, axes

    def _add_explanation(self, fig, text: str):
        """Add explanation text box at bottom of figure."""
        fig.text(0.5, 0.01, text, ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                           edgecolor='gray', alpha=0.9),
                 wrap=True)

    def _add_cluster_legend(self, ax, n_clusters: int):
        """Add cluster color legend to axis."""
        handles = [Patch(color=self.CLUSTER_COLORS[i], label=f'Cluster {i}')
                   for i in range(n_clusters)]
        ax.legend(handles=handles, loc='best', fontsize=8)

    def show_stage1_mixture_model(self, data: Dict):
        """
        Visualize latent space and cluster structure.

        Shows:
        - PCA of latent factors colored by cluster
        - Cluster size distribution
        - Pairwise cluster distances
        - Latent factor distributions by cluster
        """
        fig, axes = self._create_figure("Biotype Mixture Model", 2, 2)

        latent = np.array(data['latent_factors'])
        labels = np.array(data['cluster_labels'])
        n_clusters = self.config.n_clusters
        colors = self._get_cluster_colors(labels)

        # Top-left: PCA of latent factors
        ax = axes[0, 0]
        if latent.shape[1] > 2:
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        else:
            latent_2d = latent[:, :2]
            ax.set_xlabel('Latent Dim 1')
            ax.set_ylabel('Latent Dim 2')
        ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=colors, alpha=0.6, s=20)
        ax.set_title('Latent Space (PCA)')
        self._add_cluster_legend(ax, n_clusters)

        # Top-right: Cluster size distribution
        ax = axes[0, 1]
        cluster_counts = np.bincount(labels, minlength=n_clusters)
        bars = ax.bar(range(n_clusters), cluster_counts,
                      color=self.CLUSTER_COLORS[:n_clusters])
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Sample Count')
        ax.set_title('Cluster Sizes')
        ax.set_xticks(range(n_clusters))
        for i, (bar, count) in enumerate(zip(bars, cluster_counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{count}\n({count/len(labels):.0%})',
                    ha='center', va='bottom', fontsize=8)

        # Bottom-left: Pairwise cluster distances
        ax = axes[1, 0]
        gt = data.get('ground_truth', {})
        mm = gt.get('mixture_model', {})
        centroids = mm.get('cluster_centroids')
        if centroids is not None:
            centroids = np.array(centroids)
            distances = squareform(pdist(centroids))
            im = ax.imshow(distances, cmap='YlOrRd')
            ax.set_xticks(range(n_clusters))
            ax.set_yticks(range(n_clusters))
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Cluster')
            ax.set_title('Centroid Distances')
            for i in range(n_clusters):
                for j in range(n_clusters):
                    ax.text(j, i, f'{distances[i, j]:.2f}',
                            ha='center', va='center', fontsize=8,
                            color='white' if distances[i, j] > distances.max()/2 else 'black')
            fig.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'Centroids not available', ha='center', va='center')
            ax.set_title('Centroid Distances')

        # Bottom-right: Latent factor distributions (first 3 dims)
        ax = axes[1, 1]
        n_dims_to_show = min(3, latent.shape[1])
        positions = []
        data_to_plot = []
        colors_violin = []
        for dim in range(n_dims_to_show):
            for c in range(n_clusters):
                mask = labels == c
                data_to_plot.append(latent[mask, dim])
                positions.append(dim * (n_clusters + 1) + c)
                colors_violin.append(self.CLUSTER_COLORS[c])

        parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_violin[i])
            pc.set_alpha(0.7)
        ax.set_title('Latent Factor Distributions')
        ax.set_xlabel('Latent Dimension')
        xticks = [(n_clusters - 1) / 2 + i * (n_clusters + 1) for i in range(n_dims_to_show)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'Dim {i}' for i in range(n_dims_to_show)])

        # Explanation text
        balance = "balanced" if self.config.cluster_balance == "balanced" else \
                  f"imbalanced ({self.config.imbalance_ratio:.0%})"
        explanation = (
            f"Generated {self.config.n_samples} samples in {n_clusters} clusters | "
            f"Separation: {self.config.cluster_separation} (Cohen's d) | "
            f"Balance: {balance} | "
            f"Latent dims: {self.config.n_latent_dims}"
        )

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        self._add_explanation(fig, explanation)
        plt.show(block=self.block)

    def show_stage2_multimodal(self, data: Dict):
        """
        Visualize view generation and loadings.

        Shows:
        - PCA of View X and Y colored by cluster
        - Cross-view correlation matrix
        - Loading matrix heatmaps
        - Target vs achieved canonical correlations
        """
        fig, axes = self._create_figure("Multimodal Data Generation", 2, 3)

        view_x = np.array(data['view_x'])
        view_y = np.array(data['view_y'])
        labels = np.array(data['cluster_labels'])
        colors = self._get_cluster_colors(labels)
        gt = data.get('ground_truth', {})

        # Handle NaN values for PCA
        view_x_clean = np.nan_to_num(view_x, nan=0)
        view_y_clean = np.nan_to_num(view_y, nan=0)

        # Top-left: PCA of View X
        ax = axes[0, 0]
        pca_x = PCA(n_components=2)
        x_2d = pca_x.fit_transform(view_x_clean)
        ax.scatter(x_2d[:, 0], x_2d[:, 1], c=colors, alpha=0.6, s=20)
        ax.set_xlabel(f'PC1 ({pca_x.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca_x.explained_variance_ratio_[1]:.1%})')
        ax.set_title(f'View X: {self.config.modality_x.upper()}')
        self._add_cluster_legend(ax, self.config.n_clusters)

        # Top-middle: PCA of View Y
        ax = axes[0, 1]
        pca_y = PCA(n_components=2)
        y_2d = pca_y.fit_transform(view_y_clean)
        ax.scatter(y_2d[:, 0], y_2d[:, 1], c=colors, alpha=0.6, s=20)
        ax.set_xlabel(f'PC1 ({pca_y.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca_y.explained_variance_ratio_[1]:.1%})')
        ax.set_title(f'View Y: {self.config.modality_y.upper()}')

        # Top-right: Cross-view correlation (subsample features)
        ax = axes[0, 2]
        n_feat_show = min(10, view_x.shape[1], view_y.shape[1])
        cross_corr = np.corrcoef(view_x_clean[:, :n_feat_show].T,
                                  view_y_clean[:, :n_feat_show].T)
        cross_corr = cross_corr[:n_feat_show, n_feat_show:]
        im = ax.imshow(cross_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xlabel('View Y features')
        ax.set_ylabel('View X features')
        ax.set_title('Cross-View Correlation')
        fig.colorbar(im, ax=ax, shrink=0.8)

        # Bottom-left: Loading matrix W_x
        ax = axes[1, 0]
        loadings_x = gt.get('true_loadings_x')
        if loadings_x is not None:
            loadings_x = np.array(loadings_x)
            n_show = min(20, loadings_x.shape[0])
            im = ax.imshow(loadings_x[:n_show, :], cmap='RdBu_r', aspect='auto',
                           vmin=-1, vmax=1)
            ax.set_xlabel('Latent Dimension')
            ax.set_ylabel('Feature')
            ax.set_title(f'Loadings X (top {n_show})')
            fig.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'Loadings not available', ha='center', va='center')
            ax.set_title('Loadings X')

        # Bottom-middle: Loading matrix W_y
        ax = axes[1, 1]
        loadings_y = gt.get('true_loadings_y')
        if loadings_y is not None:
            loadings_y = np.array(loadings_y)
            n_show = min(20, loadings_y.shape[0])
            im = ax.imshow(loadings_y[:n_show, :], cmap='RdBu_r', aspect='auto',
                           vmin=-1, vmax=1)
            ax.set_xlabel('Latent Dimension')
            ax.set_ylabel('Feature')
            ax.set_title(f'Loadings Y (top {n_show})')
            fig.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'Loadings not available', ha='center', va='center')
            ax.set_title('Loadings Y')

        # Bottom-right: Target vs achieved canonical correlations
        ax = axes[1, 2]
        target_corrs = gt.get('true_canonical_correlations')
        if target_corrs is not None:
            target_corrs = np.array(target_corrs)
            n_comps = len(target_corrs)
            x_pos = np.arange(n_comps)
            ax.bar(x_pos, target_corrs, color='steelblue', alpha=0.8)
            ax.set_xlabel('Component')
            ax.set_ylabel('Canonical Correlation')
            ax.set_title('Target Canonical Correlations')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'CC{i+1}' for i in range(n_comps)])
            ax.set_ylim(0, 1)
            for i, v in enumerate(target_corrs):
                ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Correlations not available', ha='center', va='center')
            ax.set_title('Canonical Correlations')

        # Explanation text
        explanation = (
            f"View X: {self.config.n_features_x} features ({self.config.modality_x}), SNR={self.config.snr_x} | "
            f"View Y: {self.config.n_features_y} features ({self.config.modality_y}), SNR={self.config.snr_y} | "
            f"Shared components: {gt.get('n_shared_components', 'N/A')}"
        )

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        self._add_explanation(fig, explanation)
        plt.show(block=self.block)

    def show_stage3_noise(self, data: Dict):
        """
        Visualize noise and confound effects.

        Shows:
        - Site effects in PCA space
        - Missing data pattern
        - Age confound relationship
        - Sex confound relationship
        """
        fig, axes = self._create_figure("Noise & Confounds", 2, 2)

        view_x = np.array(data['view_x'])
        labels = np.array(data['cluster_labels'])
        noise_gt = data.get('noise_ground_truth', {})
        confounds = data.get('confounds', {})

        # Handle NaN for PCA
        view_x_clean = np.nan_to_num(view_x, nan=0)

        # Top-left: Site effects
        ax = axes[0, 0]
        site_labels = data.get('site_labels')
        if site_labels is not None and self.config.n_sites > 1:
            site_labels = np.array(site_labels)
            pca = PCA(n_components=2)
            x_2d = pca.fit_transform(view_x_clean)
            site_colors = [self.CLUSTER_COLORS[s % len(self.CLUSTER_COLORS)]
                           for s in site_labels]
            ax.scatter(x_2d[:, 0], x_2d[:, 1], c=site_colors, alpha=0.6, s=20)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(f'Site Effects ({self.config.n_sites} sites)')
            handles = [Patch(color=self.CLUSTER_COLORS[i], label=f'Site {i}')
                       for i in range(self.config.n_sites)]
            ax.legend(handles=handles, loc='best', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Single site\n(no batch effects)',
                    ha='center', va='center', fontsize=12)
            ax.set_title('Site Effects')
            ax.axis('off')

        # Top-right: Missing data pattern
        ax = axes[0, 1]
        if self.config.missing_rate > 0:
            missing_mask = np.isnan(view_x)
            n_show = min(50, view_x.shape[0])
            m_show = min(30, view_x.shape[1])
            ax.imshow(missing_mask[:n_show, :m_show], cmap='Greys',
                      aspect='auto', interpolation='nearest')
            ax.set_xlabel('Feature')
            ax.set_ylabel('Sample')
            actual_rate = np.mean(np.isnan(view_x))
            ax.set_title(f'Missing Pattern ({actual_rate:.1%} missing)')
        else:
            ax.text(0.5, 0.5, 'No missing data',
                    ha='center', va='center', fontsize=12)
            ax.set_title('Missing Data')
            ax.axis('off')

        # Bottom-left: Age confound
        ax = axes[1, 0]
        if 'age' in confounds:
            age = np.array(confounds['age'])
            # Use first feature as example
            feat_idx = 0
            feat_vals = view_x_clean[:, feat_idx]
            colors = self._get_cluster_colors(labels)
            ax.scatter(age, feat_vals, c=colors, alpha=0.5, s=20)
            ax.set_xlabel('Age')
            ax.set_ylabel(f'Feature 0 value')
            ax.set_title('Age Confound Effect')
            # Add regression line
            z = np.polyfit(age, feat_vals, 1)
            p = np.poly1d(z)
            ax.plot(np.sort(age), p(np.sort(age)), 'k--', alpha=0.5,
                    label=f'r={np.corrcoef(age, feat_vals)[0,1]:.2f}')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Age confound\nnot included',
                    ha='center', va='center', fontsize=12)
            ax.set_title('Age Confound')
            ax.axis('off')

        # Bottom-right: Sex confound
        ax = axes[1, 1]
        if 'sex' in confounds:
            sex = np.array(confounds['sex'])
            feat_idx = 0
            feat_vals = view_x_clean[:, feat_idx]
            male_vals = feat_vals[sex == 0]
            female_vals = feat_vals[sex == 1]
            bp = ax.boxplot([male_vals, female_vals], labels=['Male', 'Female'],
                            patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightpink')
            ax.set_ylabel(f'Feature 0 value')
            ax.set_title('Sex Confound Effect')
            # Add effect size
            if len(male_vals) > 0 and len(female_vals) > 0:
                effect = (np.mean(female_vals) - np.mean(male_vals)) / np.std(feat_vals)
                ax.text(0.95, 0.95, f"Cohen's d={effect:.2f}",
                        transform=ax.transAxes, ha='right', va='top', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Sex confound\nnot included',
                    ha='center', va='center', fontsize=12)
            ax.set_title('Sex Confound')
            ax.axis('off')

        # Explanation text
        parts = []
        if self.config.n_sites > 1:
            parts.append(f"Sites: {self.config.n_sites} (magnitude={self.config.site_effect_magnitude})")
        if self.config.missing_rate > 0:
            actual = np.mean(np.isnan(view_x))
            parts.append(f"Missing: {self.config.missing_rate:.0%} target, {actual:.1%} actual")
        if self.config.include_age or self.config.include_sex:
            conf_list = []
            if self.config.include_age:
                conf_list.append("age")
            if self.config.include_sex:
                conf_list.append("sex")
            parts.append(f"Confounds: {', '.join(conf_list)} (magnitude={self.config.confound_effect_magnitude})")

        explanation = " | ".join(parts) if parts else "No noise or confounds applied"

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        self._add_explanation(fig, explanation)
        plt.show(block=self.block)

    def show_stage4_summary(self, data: Dict):
        """
        Final quality summary.

        Shows:
        - Combined PCA of both views
        - Silhouette scores by cluster
        - Configuration summary
        - Quality metrics
        """
        fig, axes = self._create_figure("Generation Summary", 2, 2)

        view_x = np.array(data['view_x'])
        view_y = np.array(data['view_y'])
        labels = np.array(data['cluster_labels'])
        colors = self._get_cluster_colors(labels)
        gt = data.get('ground_truth', {})

        # Handle NaN
        view_x_clean = np.nan_to_num(view_x, nan=0)
        view_y_clean = np.nan_to_num(view_y, nan=0)

        # Combine views
        combined = np.hstack([view_x_clean, view_y_clean])

        # Top-left: Combined PCA
        ax = axes[0, 0]
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        ax.scatter(combined_2d[:, 0], combined_2d[:, 1], c=colors, alpha=0.6, s=20)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('Combined View PCA')
        self._add_cluster_legend(ax, self.config.n_clusters)

        # Top-right: Silhouette scores
        ax = axes[0, 1]
        try:
            sil_samples = silhouette_samples(combined, labels)
            sil_avg = silhouette_score(combined, labels)

            y_lower = 0
            for i in range(self.config.n_clusters):
                cluster_sil = sil_samples[labels == i]
                cluster_sil.sort()
                size = len(cluster_sil)
                y_upper = y_lower + size
                ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                                 facecolor=self.CLUSTER_COLORS[i], alpha=0.7)
                ax.text(-0.05, y_lower + 0.5 * size, str(i), fontsize=10)
                y_lower = y_upper + 5

            ax.axvline(sil_avg, color='red', linestyle='--', label=f'Avg: {sil_avg:.2f}')
            ax.set_xlabel('Silhouette Coefficient')
            ax.set_ylabel('Cluster')
            ax.set_title('Silhouette Analysis')
            ax.legend(loc='upper right', fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f'Silhouette error:\n{str(e)[:30]}',
                    ha='center', va='center')
            ax.set_title('Silhouette Analysis')

        # Bottom-left: Configuration summary
        ax = axes[1, 0]
        ax.axis('off')
        config_text = (
            f"CONFIGURATION SUMMARY\n"
            f"{'─' * 30}\n"
            f"Study: {self.config.name}\n"
            f"Samples: {self.config.n_samples}\n"
            f"Clusters: {self.config.n_clusters}\n"
            f"Separation: {self.config.cluster_separation}\n"
            f"Balance: {self.config.cluster_balance}\n"
            f"{'─' * 30}\n"
            f"View X: {self.config.modality_x}\n"
            f"  Features: {self.config.n_features_x}, SNR: {self.config.snr_x}\n"
            f"View Y: {self.config.modality_y}\n"
            f"  Features: {self.config.n_features_y}, SNR: {self.config.snr_y}\n"
            f"{'─' * 30}\n"
            f"Sites: {self.config.n_sites}\n"
            f"Missing rate: {self.config.missing_rate:.0%}\n"
            f"Confounds: {'age ' if self.config.include_age else ''}{'sex' if self.config.include_sex else ''}"
        )
        ax.text(0.1, 0.95, config_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax.set_title('Configuration')

        # Bottom-right: Quality metrics
        ax = axes[1, 1]
        ax.axis('off')

        # Compute quality metrics
        try:
            sil_score = silhouette_score(combined, labels)
        except:
            sil_score = float('nan')

        # Cross-view correlation (first few features)
        n_feat = min(5, view_x.shape[1], view_y.shape[1])
        cross_corrs = [np.corrcoef(view_x_clean[:, i], view_y_clean[:, i])[0, 1]
                       for i in range(n_feat)]
        avg_cross_corr = np.nanmean(cross_corrs)

        # Cluster size balance
        counts = np.bincount(labels, minlength=self.config.n_clusters)
        balance_ratio = counts.min() / counts.max() if counts.max() > 0 else 0

        target_corrs = gt.get('true_canonical_correlations', [])
        if target_corrs is not None and len(target_corrs) > 0:
            target_corrs = np.array(target_corrs)
            cc_text = ', '.join([f'{c:.2f}' for c in target_corrs[:3]])
        else:
            cc_text = 'N/A'

        metrics_text = (
            f"QUALITY METRICS\n"
            f"{'─' * 30}\n"
            f"Silhouette Score: {sil_score:.3f}\n"
            f"  (>0.5 = good separation)\n"
            f"\n"
            f"Cluster Balance: {balance_ratio:.2f}\n"
            f"  (1.0 = perfectly balanced)\n"
            f"\n"
            f"Avg Cross-View Corr: {avg_cross_corr:.3f}\n"
            f"  (shared structure)\n"
            f"\n"
            f"Target Canon. Corrs:\n"
            f"  [{cc_text}]"
        )
        ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.set_title('Quality Metrics')

        # Explanation text
        explanation = (
            f"Generation complete! Silhouette={sil_score:.2f} | "
            f"Balance ratio={balance_ratio:.2f} | "
            f"Cross-view correlation={avg_cross_corr:.2f}"
        )

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        self._add_explanation(fig, explanation)
        plt.show(block=self.block)

    def has_noise(self) -> bool:
        """Check if any noise/confounds are configured."""
        return (self.config.n_sites > 1 or
                self.config.missing_rate > 0 or
                self.config.include_age or
                self.config.include_sex)
