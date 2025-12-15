#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# Rendering utilities for obstacle and RF visualization
#
# @author      James R. Heselden (github: Iranaphor)
# @maintainer  James R. Heselden (github: Iranaphor)
# @datecreated 12th December 2025
# @datemodified 12th December 2025
# @credits     Developed with assistance from Claude Sonnet 4.5 and GitHub
#              Copilot.
# ###########################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

from config import CATEGORY_COLORS, OBSTACLE_CODES


def get_default_figsize(aspect='landscape'):
    """Return consistent figure sizes."""
    if aspect == 'square':
        return (8, 8)
    elif aspect == 'wide':
        return (12, 6)
    elif aspect == 'tall':
        return (6, 12)
    else:
        return (10, 6)


def get_default_dpi():
    """Return consistent DPI for all plots."""
    return 200


def apply_plot_styling(ax, xlabel="", ylabel="", title=""):
    """Apply consistent styling to matplotlib axes."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(False)


def get_obstacle_colormap(categories=None):
    """Return categorical colormap for obstacle visualization."""
    if categories is None:
        categories = ["unknown", "open", "lake", "trees", "building"]
    
    colors = [CATEGORY_COLORS.get(cat, "#cccccc") for cat in categories]
    cmap = ListedColormap(colors)
    bounds = np.arange(len(categories) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    
    return cmap, norm, categories


def get_obstacle_legend_patches():
    """Return legend patches for obstacle categories."""
    labels = ["unknown", "open", "lake", "trees", "building"]
    patches = [
        mpatches.Patch(color=CATEGORY_COLORS[lab], label=lab)
        for lab in labels if lab in CATEGORY_COLORS
    ]
    return patches


def compute_xy_extent(polygons, local_points=None, margin_factor=0.05):
    """Compute plot extents from polygons and optional points."""
    xs, ys = [], []

    for poly, _cat in polygons:
        bx_min, by_min, bx_max, by_max = poly.bounds
        xs.extend([bx_min, bx_max])
        ys.extend([by_min, by_max])

    if local_points:
        for x, y, _v in local_points:
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        return (-10, 10, -10, 10)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    dx = x_max - x_min
    dy = y_max - y_min
    mx = dx * margin_factor if dx > 0 else 1.0
    my = dy * margin_factor if dy > 0 else 1.0

    return (x_min - mx, x_max + mx, y_min - my, y_max + my)


def compute_polar_extent(radii_m, angles_deg=None):
    """Compute polar plot extents."""
    if len(radii_m) > 1:
        dr = radii_m[1] - radii_m[0]
    else:
        dr = radii_m[0] * 2.0 if len(radii_m) == 1 else 1.0
    
    r_max = radii_m[-1] + dr / 2.0
    theta_min, theta_max = 0.0, 360.0
    
    return r_max, theta_min, theta_max


def draw_polygons_xy(ax, polygons, show_fill=True, show_edges=True, alpha=0.8):
    """Draw polygons on XY axes."""
    for poly, cat in polygons:
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS.get("unknown", "#cccccc"))
        x, y = poly.exterior.xy

        if show_fill:
            ax.fill(x, y, facecolor=color, edgecolor="none", alpha=alpha)
        if show_edges:
            ax.plot(x, y, color="black", linewidth=0.7)


def draw_rf_scatter(ax, local_points, s=10, alpha=0.8, cmap='viridis'):
    """Draw RF data points as colored scatter."""
    xs = [p[0] for p in local_points]
    ys = [p[1] for p in local_points]
    vs = [p[2] for p in local_points]

    sc = ax.scatter(xs, ys, c=vs, s=s, alpha=alpha, cmap=cmap)
    return sc


def add_colorbar(fig, mappable, ax, label=""):
    """Add a colorbar with consistent styling."""
    cbar = fig.colorbar(mappable, ax=ax)
    if label:
        cbar.set_label(label, fontsize=10)
    return cbar


def add_legend(ax, location="upper right", fontsize="small"):
    """Add obstacle category legend to axes."""
    patches = get_obstacle_legend_patches()
    if patches:
        ax.legend(handles=patches, loc=location, fontsize=fontsize, frameon=True)


def obstacle_grid_to_numeric(obstacle_grid, categories=None):
    """Convert obstacle grid with category strings to numeric grid for plotting."""
    if categories is None:
        categories = ["open", "trees", "building", "lake", "unknown"]
    
    present = {c for c in obstacle_grid.flatten()}
    categories = [c for c in categories if c in present]
    
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    numeric_grid = np.vectorize(cat_to_idx.get)(obstacle_grid)
    
    return numeric_grid, categories


def signature_to_obstacle_grid(unique_sigs, n_r):
    """Convert signature strings to obstacle grid for visualization."""
    num_sigs = len(unique_sigs)
    sig_obs_grid = np.full((num_sigs, n_r), np.nan, dtype=float)

    for i, sig in enumerate(unique_sigs):
        # Skip 's' prefix if present
        sig_str = sig[1:] if sig.startswith('s') else sig
        L = min(len(sig_str), n_r)
        for j in range(L):
            c = sig_str[j]
            if c.isdigit():
                sig_obs_grid[i, j] = int(c)

    return sig_obs_grid


def plot_signature_comparison(before_sigs, after_sigs, before_data, after_data,
                               radii_m, out_path, title_prefix="Dataset"):
    """
    Plot before/after comparison of signature data.
    Shows obstacle patterns and RF data side by side.
    """
    # Determine all unique signatures
    all_sigs = sorted(set(list(before_sigs.keys()) + list(after_sigs.keys())))
    num_sigs = len(all_sigs)
    
    if num_sigs == 0:
        print("[WARNING] No signatures to plot")
        return
    
    # Determine max bins
    max_bins = 0
    for sig in all_sigs:
        if sig in before_data:
            max_bins = max(max_bins, len(before_data[sig]))
        if sig in after_data:
            max_bins = max(max_bins, len(after_data[sig]))
    
    if len(radii_m) > max_bins:
        radii_m = radii_m[:max_bins]
    dr = radii_m[1] - radii_m[0] if len(radii_m) > 1 else 10.0
    r_max = radii_m[-1] + dr / 2.0 if len(radii_m) > 0 else max_bins * 10
    
    # Build obstacle grids
    before_obs_grid = signature_to_obstacle_grid(all_sigs, max_bins)
    
    # Build RF data grids
    before_rf_grid = np.full((num_sigs, max_bins), np.nan)
    after_rf_grid = np.full((num_sigs, max_bins), np.nan)
    
    for i, sig in enumerate(all_sigs):
        if sig in before_data:
            data = before_data[sig]
            for j in range(min(len(data), max_bins)):
                val = data[j]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    before_rf_grid[i, j] = val
        
        if sig in after_data:
            data = after_data[sig]
            for j in range(min(len(data), max_bins)):
                val = data[j]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    after_rf_grid[i, j] = val
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, max(8, 0.3 * num_sigs)))
    
    # Obstacle colormap
    obs_cmap, obs_norm, _ = get_obstacle_colormap(
        ["unknown", "open", "lake", "trees", "building"]
    )
    
    # Top left: obstacles
    im_obs = axes[0, 0].imshow(
        before_obs_grid,
        origin="lower",
        aspect="auto",
        extent=[0, r_max, 0, num_sigs],
        cmap=obs_cmap,
        norm=obs_norm,
        interpolation="nearest"
    )
    apply_plot_styling(axes[0, 0],
                      xlabel="Distance from base (m)",
                      ylabel="Signature index",
                      title="Obstacle Patterns")
    
    # Top right: before RF
    im_before = axes[0, 1].imshow(
        before_rf_grid,
        origin="lower",
        aspect="auto",
        extent=[0, r_max, 0, num_sigs],
        interpolation="nearest"
    )
    apply_plot_styling(axes[0, 1],
                      xlabel="Distance from base (m)",
                      title=f"{title_prefix} - Before")
    cbar_before = fig.colorbar(im_before, ax=axes[0, 1])
    cbar_before.set_label("RF Value")
    
    # Bottom left: signature labels
    axes[1, 0].axis('off')
    sig_text = "\n".join([f"{i}: {sig}" for i, sig in enumerate(all_sigs[:50])])
    if num_sigs > 50:
        sig_text += f"\n... and {num_sigs - 50} more"
    axes[1, 0].text(0.1, 0.5, sig_text, fontsize=6, family='monospace',
                    verticalalignment='center')
    
    # Bottom right: after RF
    im_after = axes[1, 1].imshow(
        after_rf_grid,
        origin="lower",
        aspect="auto",
        extent=[0, r_max, 0, num_sigs],
        interpolation="nearest"
    )
    apply_plot_styling(axes[1, 1],
                      xlabel="Distance from base (m)",
                      title=f"{title_prefix} - After")
    cbar_after = fig.colorbar(im_after, ax=axes[1, 1])
    cbar_after.set_label("RF Value")
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=get_default_dpi())
    plt.close(fig)
    print(f"[PLOT] Saved comparison to {out_path}")
