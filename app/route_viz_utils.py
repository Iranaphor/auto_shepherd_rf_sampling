#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# Visualization utilities for search route grid analysis
#
# AI-WRITTEN CODE DECLARATION:
# This code was developed with substantial assistance from AI tools.
#
# @author      James R. Heselden (github: Iranaphor)
# @maintainer  James R. Heselden (github: Iranaphor)
# @datecreated 15th December 2025
# ###########################################################################

import numpy as np


def plot_obstacle_map_with_grid(polygons, grid_points, edge_signature_counts,
                                base_xy, out_path):
    """
    Plot obstacle map with grid overlay showing edge signature diversity.
    Uses render_utils for matplotlib operations.
    """
    from render_utils import (
        get_default_figsize, get_default_dpi, apply_plot_styling, draw_polygons_xy
    )
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    
    fig, ax = plt.subplots(figsize=get_default_figsize('square'), 
                          dpi=get_default_dpi())
    
    draw_polygons_xy(ax, polygons, show_fill=True, show_edges=True)
    
    if edge_signature_counts:
        edges_list = list(edge_signature_counts.keys())
        counts = np.array([edge_signature_counts[e] for e in edges_list])
        
        if counts.max() > counts.min():
            norm_counts = (counts - counts.min()) / (counts.max() - counts.min())
        else:
            norm_counts = np.ones_like(counts) * 0.5
        
        segments = [[(p1[0], p1[1]), (p2[0], p2[1])] for p1, p2 in edges_list]
        
        lc = LineCollection(segments, linewidths=1.5, alpha=0.7)
        lc.set_array(norm_counts)
        lc.set_cmap('coolwarm')
        ax.add_collection(lc)
        
        cbar = plt.colorbar(lc, ax=ax, label='Unique Signatures per Edge')
        cbar.ax.set_ylabel('Unique Signatures per Edge', rotation=270, labelpad=20)
    
    if grid_points:
        xs, ys = zip(*grid_points)
        ax.scatter(xs, ys, c='black', s=20, alpha=0.4, zorder=5, label='Grid Points')
    
    if base_xy:
        ax.scatter([base_xy[0]], [base_xy[1]], c='gold', s=300, 
                  marker='*', edgecolors='black', linewidths=1.5,
                  zorder=10, label='Base Station')
    
    apply_plot_styling(ax, xlabel="X (m)", ylabel="Y (m)", 
                      title="Grid Overlay with Edge Signature Diversity")
    ax.legend(loc='best', fontsize=9)
    ax.axis('equal')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=get_default_dpi(), bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved grid overlay plot to {out_path}")


def plot_grid_points_colored_by_potential(polygons, grid_points, 
                                         point_signature_counts,
                                         base_xy, out_path):
    """
    Plot grid points colored by their data collection potential.
    Uses render_utils for matplotlib operations.
    """
    from render_utils import (
        get_default_figsize, get_default_dpi, apply_plot_styling, draw_polygons_xy
    )
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=get_default_figsize('square'), 
                          dpi=get_default_dpi())
    
    draw_polygons_xy(ax, polygons, show_fill=True, show_edges=False, alpha=0.3)
    
    if grid_points and point_signature_counts:
        xs, ys = zip(*grid_points)
        potentials = [point_signature_counts.get(pt, 0) for pt in grid_points]
        
        scatter = ax.scatter(xs, ys, c=potentials, s=80, alpha=0.8,
                           cmap='viridis', edgecolors='black', linewidth=0.5,
                           zorder=5)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.ax.set_ylabel('Data Potential (unique signature)', 
                          rotation=270, labelpad=20)
    
    if base_xy:
        ax.scatter([base_xy[0]], [base_xy[1]], c='red', s=400, 
                  marker='*', edgecolors='black', linewidths=2,
                  zorder=10, label='Base Station')
    
    apply_plot_styling(ax, xlabel="X (m)", ylabel="Y (m)", 
                      title="Grid Points Colored by Data Collection Potential")
    ax.legend(loc='best', fontsize=9)
    ax.axis('equal')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=get_default_dpi(), bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved point potential plot to {out_path}")


def plot_signature_distribution_histogram(edge_signature_counts, out_path):
    """
    Plot histogram showing distribution of unique signatures per edge.
    Uses render_utils for matplotlib operations.
    """
    from render_utils import get_default_figsize, get_default_dpi, apply_plot_styling
    import matplotlib.pyplot as plt
    
    if not edge_signature_counts:
        print("[VIZ] No edge data to plot histogram")
        return
    
    counts = list(edge_signature_counts.values())
    
    fig, ax = plt.subplots(figsize=get_default_figsize('landscape'), 
                          dpi=get_default_dpi())
    
    ax.hist(counts, bins=max(20, max(counts)), color='steelblue', 
           edgecolor='black', alpha=0.7)
    
    mean_count = np.mean(counts)
    median_count = np.median(counts)
    ax.axvline(mean_count, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_count:.1f}')
    ax.axvline(median_count, color='orange', linestyle='--', linewidth=2,
              label=f'Median: {median_count:.1f}')
    
    apply_plot_styling(ax, xlabel="Unique Signatures per Edge", 
                      ylabel="Frequency",
                      title="Distribution of Edge Information Content")
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=get_default_dpi(), bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved signature distribution histogram to {out_path}")
