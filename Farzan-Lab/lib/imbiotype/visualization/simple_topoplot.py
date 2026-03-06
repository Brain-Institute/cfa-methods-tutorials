"""
Simple topographic plotting without MNE dependency
Uses standard 10-20 electrode positions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Dict, List, Optional

# Standard 10-20 electrode positions (subset)
STANDARD_1020_POSITIONS = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
    'F7': (-0.7, 0.4), 'F3': (-0.4, 0.4), 'Fz': (0, 0.4), 'F4': (0.4, 0.4), 'F8': (0.7, 0.4),
    'T7': (-0.9, 0), 'C3': (-0.4, 0), 'Cz': (0, 0), 'C4': (0.4, 0), 'T8': (0.9, 0),
    'P7': (-0.7, -0.4), 'P3': (-0.4, -0.4), 'Pz': (0, -0.4), 'P4': (0.4, -0.4), 'P8': (0.7, -0.4),
    'O1': (-0.3, -0.9), 'O2': (0.3, -0.9)
}

def plot_simple_topomap(weights: np.ndarray, 
                       channel_names: List[str],
                       title: str = "Topographic Map",
                       ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Simple topographic plot using standard electrode positions
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    # Match channel names to positions
    matched_positions = []
    matched_weights = []
    
    for i, ch_name in enumerate(channel_names):
        # Try exact match first
        if ch_name in STANDARD_1020_POSITIONS:
            matched_positions.append(STANDARD_1020_POSITIONS[ch_name])
            matched_weights.append(weights[i])
        else:
            # Try substring match
            for std_name, pos in STANDARD_1020_POSITIONS.items():
                if std_name.lower() in ch_name.lower() or ch_name.lower() in std_name.lower():
                    matched_positions.append(pos)
                    matched_weights.append(weights[i])
                    break
    
    if not matched_positions:
        # Fallback: arrange in circle
        n_channels = len(channel_names)
        angles = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
        matched_positions = [(0.7*np.cos(a), 0.7*np.sin(a)) for a in angles]
        matched_weights = weights.tolist()
    
    # Convert to arrays
    positions = np.array(matched_positions)
    weights_array = np.array(matched_weights)
    
    # Create interpolation grid
    xi = np.linspace(-1, 1, 100)
    yi = np.linspace(-1, 1, 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate weights
    zi = griddata(positions, weights_array, (xi_grid, yi_grid), method='cubic')
    
    # Mask outside head circle
    mask = xi_grid**2 + yi_grid**2 > 1
    zi[mask] = np.nan
    
    # Plot interpolated surface
    im = ax.contourf(xi_grid, yi_grid, zi, levels=20, cmap='RdBu_r', extend='both')
    
    # Plot electrode positions
    ax.scatter(positions[:, 0], positions[:, 1], c='black', s=30, zorder=10)
    
    # Draw head outline
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    # Draw nose
    nose_x = [0, 0.1, 0, -0.1, 0]
    nose_y = [1, 1.15, 1.3, 1.15, 1]
    ax.plot(nose_x, nose_y, 'k-', linewidth=2)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8, label='Weight')
    
    return fig