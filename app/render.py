#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# Unified rendering class for obstacle and RF visualization
#
# @author      James R. Heselden (github: iranaphor)
# @maintainer  James R. Heselden (github: iranaphor)
# @datecreated 12th December 2025
# ###########################################################################

import os
import numpy as np
import matplotlib.pyplot as plt

from render_utils import *
from config import CATEGORY_COLORS, OBSTACLE_CODES


class Render:
    """
    Unified rendering class for obstacle maps and RF data visualization.
    
    Supports both XY plane and polar coordinate rendering with consistent styling.
    """
    
    def __init__(self, obstacle_kml_path=None, rf_data_path=None):
        """
        Initialize renderer with data paths.
        
        Args:
            obstacle_kml_path: Path to KML file with obstacle polygons
            rf_data_path: Optional path to RF data (YAML or other format)
        """
        self.obstacle_kml_path = obstacle_kml_path
        self.rf_data_path = rf_data_path
        
        # Data containers (loaded on demand)
        self.polygons = None
        self.rf_points = None
        self.center_lat = None
        self.center_lon = None
        self.local_rf_points = None
        
        # Cached computed data
        self._angles_deg = None
        self._radii_m = None
        self._obstacle_grid = None
        self._heatmap = None
    
    def load_data(self, center_lat=None, center_lon=None):
        """
        Load obstacle and RF data.
        
        Args:
            center_lat, center_lon: Center coordinates for conversion
        """
        from utils import load_kml_polygons, load_yaml_points, latlon_to_local_xy
        
        if self.obstacle_kml_path and center_lat is not None and center_lon is not None:
            self.polygons = load_kml_polygons(self.obstacle_kml_path, center_lat, center_lon)
            self.center_lat = center_lat
            self.center_lon = center_lon
        
        if self.rf_data_path:
            self.center_lat, self.center_lon, self.rf_points = load_yaml_points(self.rf_data_path)
            
            # Convert to local XY
            self.local_rf_points = []
            for lat, lon, v in self.rf_points:
                x, y = latlon_to_local_xy(lat, lon, self.center_lat, self.center_lon)
                self.local_rf_points.append((x, y, v))
    
    def render(self, obstacle_edge=False, obstacle_fill=False, rf_data=False, 
               use_polar=False, fileout=None):
        """
        Render visualization based on specified options.
        
        Args:
            obstacle_edge: Show obstacle boundaries
            obstacle_fill: Fill obstacle regions with colors
            rf_data: Show RF data overlay
            use_polar: Use polar coordinates (vs XY plane)
            fileout: Output file path (None = return figure)
        
        Returns:
            matplotlib Figure object if fileout is None, else saves to file
        """
        if use_polar:
            return self._render_polar(obstacle_edge, obstacle_fill, rf_data, fileout)
        else:
            return self._render_xy(obstacle_edge, obstacle_fill, rf_data, fileout)
    
    def _render_xy(self, show_edges, show_fill, show_rf, fileout):
        """Render XY plane visualization."""
        if self.polygons is None:
            raise ValueError("No polygon data loaded. Call load_data() first.")
        
        # Determine what to show
        local_points = self.local_rf_points if show_rf else None
        
        # Compute extent
        x_min, x_max, y_min, y_max = compute_xy_extent(self.polygons, local_points)
        
        # Create figure
        fig, ax = plt.subplots(figsize=get_default_figsize('square'))
        
        # Draw polygons
        if show_fill or show_edges:
            draw_polygons_xy(ax, self.polygons, show_fill=show_fill, show_edges=show_edges)
        
        # Draw RF data
        if show_rf and local_points:
            sc = draw_rf_scatter(ax, local_points)
            add_colorbar(fig, sc, ax, label="Value (v)")
        
        # Styling
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        title_parts = []
        if show_fill:
            title_parts.append("Obstacles")
        if show_edges and not show_fill:
            title_parts.append("Boundaries")
        if show_rf:
            title_parts.append("RF Data")
        title = " + ".join(title_parts) + " (XY plane)" if title_parts else "XY plane"
        
        apply_plot_styling(ax, xlabel="X (m, local)", ylabel="Y (m, local)", title=title)
        
        # Add legend for obstacle colors
        if show_fill:
            add_legend(ax)
        
        fig.tight_layout()
        
        if fileout:
            fig.savefig(fileout, dpi=get_default_dpi())
            plt.close(fig)
            print(f"Saved {fileout}")
            return None
        else:
            return fig
    
    def _render_polar(self, show_edges, show_fill, show_rf, fileout):
        """Render polar coordinate visualization."""
        if self._obstacle_grid is None:
            raise ValueError("No polar grid data. Build polar grid first.")
        
        n_theta, n_r = self._obstacle_grid.shape
        r_max, theta_min, theta_max = compute_polar_extent(self._radii_m, self._angles_deg)
        
        fig, ax = plt.subplots(figsize=get_default_figsize())
        
        if show_fill or show_edges:
            # Show obstacle grid
            numeric_grid, categories = obstacle_grid_to_numeric(self._obstacle_grid)
            cmap, norm, _ = get_obstacle_colormap(categories)
            
            im = ax.imshow(
                numeric_grid,
                origin="lower",
                aspect="auto",
                extent=[0, r_max, theta_min, theta_max],
                cmap=cmap,
                norm=norm,
                interpolation="nearest"
            )
            
            if show_fill:
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_ticks(np.arange(len(categories)))
                cbar.set_ticklabels(categories)
        
        if show_rf and self._heatmap is not None:
            # Overlay or show heatmap
            im = ax.imshow(
                self._heatmap,
                origin="lower",
                aspect="auto",
                extent=[0, r_max, theta_min, theta_max],
                interpolation="nearest",
                alpha=0.7 if show_fill else 1.0
            )
            add_colorbar(fig, im, ax, label="Mean value (v)")
        
        apply_plot_styling(ax, xlabel="Distance from base (m)", 
                          ylabel="Angle (deg)", title="Polar visualization")
        
        fig.tight_layout()
        
        if fileout:
            fig.savefig(fileout, dpi=get_default_dpi())
            plt.close(fig)
            print(f"Saved {fileout}")
            return None
        else:
            return fig
    
    def set_polar_grid(self, angles_deg, radii_m, obstacle_grid):
        """Set precomputed polar grid data."""
        self._angles_deg = angles_deg
        self._radii_m = radii_m
        self._obstacle_grid = obstacle_grid
    
    def set_heatmap(self, heatmap):
        """Set precomputed heatmap data."""
        self._heatmap = heatmap
    
    def render_obstacle_polar(self, fileout):
        """Render obstacle-only polar map."""
        return self.render(obstacle_fill=True, obstacle_edge=False, 
                          rf_data=False, use_polar=True, fileout=fileout)
    
    def render_heatmap_polar(self, fileout):
        """Render RF heatmap in polar coordinates."""
        return self.render(obstacle_fill=False, obstacle_edge=False, 
                          rf_data=True, use_polar=True, fileout=fileout)
    
    def render_xy_obstacles(self, fileout):
        """Render XY obstacle map with fill and edges."""
        return self.render(obstacle_fill=True, obstacle_edge=True, 
                          rf_data=False, use_polar=False, fileout=fileout)
    
    def render_xy_boundaries_rf(self, fileout):
        """Render XY boundaries with RF overlay."""
        return self.render(obstacle_fill=False, obstacle_edge=True, 
                          rf_data=True, use_polar=False, fileout=fileout)
    
    def render_xy_obstacles_rf(self, fileout):
        """Render XY obstacles with RF overlay."""
        return self.render(obstacle_fill=True, obstacle_edge=True, 
                          rf_data=True, use_polar=False, fileout=fileout)
    
    def render_signature_slices(self, unique_sigs, mean_slices, fileout):
        """
        Render signature slices visualization (2-panel plot).
        
        Args:
            unique_sigs: List of signature strings
            mean_slices: (num_sigs, n_r) array of mean RF values
            fileout: Output file path
        """
        num_sigs, n_r = mean_slices.shape
        
        if self._radii_m is not None and len(self._radii_m) > 0:
            radii_m = self._radii_m
        else:
            # Fallback if radii not set
            from config import DR_M
            radii_m = np.arange(n_r) * DR_M
        
        if len(radii_m) > 1:
            dr = radii_m[1] - radii_m[0]
        else:
            dr = radii_m[0] * 2.0 if len(radii_m) == 1 else 1.0
        
        r_max = radii_m[-1] + dr / 2.0
        
        # Build obstacle grid from signatures
        sig_obs_grid = signature_to_obstacle_grid(unique_sigs, n_r)
        
        # Obstacle colormap
        obs_colors = [
            CATEGORY_COLORS["unknown"],
            CATEGORY_COLORS["open"],
            CATEGORY_COLORS["lake"],
            CATEGORY_COLORS["trees"],
            CATEGORY_COLORS["building"],
        ]
        obs_cmap, obs_norm, _ = get_obstacle_colormap(
            ["unknown", "open", "lake", "trees", "building"]
        )
        
        # Create figure
        fig, (ax_obs, ax_val) = plt.subplots(
            1, 2,
            figsize=(30, max(4, 0.25 * num_sigs)),
            sharey=True
        )
        
        # Left: obstacle patterns
        im_obs = ax_obs.imshow(
            sig_obs_grid,
            origin="lower",
            aspect="auto",
            extent=[0, r_max, 0, num_sigs],
            cmap=obs_cmap,
            norm=obs_norm,
            interpolation="nearest"
        )
        apply_plot_styling(ax_obs, 
                          xlabel="Distance from base (m)",
                          ylabel="Obstacle signature (sorted index)",
                          title="Obstacle pattern per signature")
        
        # Legend
        add_legend(ax_obs, location="upper right")
        
        # Right: mean RF values
        im_val = ax_val.imshow(
            mean_slices,
            origin="lower",
            aspect="auto",
            extent=[0, r_max, 0, num_sigs],
            interpolation="nearest"
        )
        add_colorbar(fig, im_val, ax_val, label="Mean value (v)")
        apply_plot_styling(ax_val,
                          xlabel="Distance from base (m)",
                          title="Mean RF per signature")
        
        # Y tick labels
        max_labels = 20
        if num_sigs <= max_labels:
            yticks = np.arange(num_sigs) + 0.5
            labels_sig = []
            for s in unique_sigs:
                if len(s) <= 20:
                    labels_sig.append(s)
                else:
                    labels_sig.append(s[:8] + "…" + s[-4:])
            ax_obs.set_yticks(yticks)
            ax_obs.set_yticklabels(labels_sig)
            ax_val.set_yticks(yticks)
            ax_val.set_yticklabels([])
        else:
            yticks = np.arange(num_sigs) + 0.5
            ax_obs.set_yticks(yticks)
            ax_obs.set_yticklabels([str(i) for i in range(num_sigs)])
            ax_val.set_yticks(yticks)
            ax_val.set_yticklabels([])
        
        fig.suptitle("Obstacle signatures and averaged RF slices", y=0.98)
        fig.tight_layout()
        fig.savefig(fileout, dpi=get_default_dpi())
        plt.close(fig)
        print(f"Saved {fileout}")
