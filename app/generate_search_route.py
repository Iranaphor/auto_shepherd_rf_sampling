#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# Generate optimized search route for RF data collection
#
# AI-WRITTEN CODE DECLARATION:
# This code was developed with substantial assistance from AI tools.
#
# @author      James R. Heselden (github: Iranaphor)
# @maintainer  James R. Heselden (github: Iranaphor)
# @datecreated 15th December 2025
# ###########################################################################

import os
import numpy as np

from kml_utils import load_kml_polygons, auto_detect_center
from grid_utils import (
    generate_grid_points,
    generate_8connected_edges,
    compute_grid_bounds_from_polygons,
    compute_obstacle_signature_for_point,
    count_unique_signatures_on_edge
)
from route_viz_utils import (
    plot_obstacle_map_with_grid,
    plot_grid_points_colored_by_potential,
    plot_signature_distribution_histogram
)


def main():
    """
    Main execution function for search route generation.
    
    Process:
    1. Load obstacle map from KML
    2. Generate regular grid across environment
    3. Compute obstacle signatures from each point to base
    4. Generate 8-connected edges with subnodes
    5. Count unique signatures along each edge
    6. Visualize grid with edges colored by information content
    """
    
    # Configuration from environment variables
    data_path = os.getenv('DATA_PATH')
    base_lat = float(os.getenv('CENTER_LAT'))
    base_lon = float(os.getenv('CENTER_LON'))
    grid_spacing = float(os.getenv('GRID_SPACING', '20.0'))  # meters
    num_subnodes = int(os.getenv('NUM_SUBNODES', '5'))  # subnodes per edge
    
    # Polar grid parameters for signature computation
    dtheta_deg = float(os.getenv('ANGULAR_BIN_SIZE', '5.0'))
    dr_m = float(os.getenv('RADIAL_BIN_SIZE', '10.0'))
    max_range_m = float(os.getenv('MAX_RANGE_M', '500.0'))

    # File paths
    kml_path = os.path.join(data_path, "feature_map.kml")
    
    print(f"[INFO] Starting search route generation")
    print(f"[INFO] Data path: {data_path}")
    print(f"[INFO] Grid spacing: {grid_spacing} m")
    print(f"[INFO] Subnodes per edge: {num_subnodes}")
    
    # Load obstacle map from KML using base location as reference origin
    print(f"[KML] Loading obstacle map from {kml_path}")
    print(f"[KML] Using base location as reference origin: {base_lat:.6f}, {base_lon:.6f}")
    
    polygons = load_kml_polygons(kml_path, base_lat, base_lon)
    print(f"[KML] Loaded {len(polygons)} obstacle polygons")
    
    # Base location is at origin (0, 0) since we used it as reference
    base_x, base_y = 0.0, 0.0
    print(f"[BASE] Base location at origin: ({base_x:.1f}, {base_y:.1f}) m")
    
    # Compute grid bounds from obstacles
    x_min, x_max, y_min, y_max = compute_grid_bounds_from_polygons(
        polygons, margin=20.0
    )
    print(f"[GRID] Bounds: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
    
    # Generate grid points
    print(f"[GRID] Generating regular grid...")
    grid_points = generate_grid_points(x_min, x_max, y_min, y_max, grid_spacing)
    print(f"[GRID] Generated {len(grid_points)} grid points")
    
    # Generate 8-connected edges
    print(f"[GRID] Generating 8-connected edges...")
    edges = generate_8connected_edges(grid_points, grid_spacing)
    print(f"[GRID] Generated {len(edges)} edges")
    
    # Setup polar grid parameters for signature computation
    angles_deg = (np.arange(int(360.0 / dtheta_deg)) + 0.5) * dtheta_deg
    radii_m = (np.arange(int(max_range_m / dr_m)) + 0.5) * dr_m
    print(f"[SIGNATURE] Polar grid: {len(angles_deg)} angles x {len(radii_m)} radii")
    
    # Compute obstacle signature for each grid point
    print(f"[SIGNATURE] Computing obstacle signatures for each grid point...")
    point_signatures = {}
    
    for i, (x, y) in enumerate(grid_points):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(grid_points)} points")
        
        sig = compute_obstacle_signature_for_point(
            x, y, base_x, base_y, polygons, angles_deg, radii_m
        )
        point_signatures[(x, y)] = sig
    
    print(f"[SIGNATURE] Computed signatures for {len(point_signatures)} points")
    
    # Count unique signatures per point (for visualization)
    # Here we just store 1 signature per point, but could be extended
    point_signature_counts = {pt: 1 for pt in point_signatures.keys()}
    
    # Count unique signatures along each edge
    print(f"[EDGE] Computing unique signatures along each edge...")
    edge_signature_counts = {}
    
    for i, edge in enumerate(edges):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(edges)} edges")
        
        count = count_unique_signatures_on_edge(
            edge, base_x, base_y, polygons, 
            angles_deg, radii_m, num_subnodes
        )
        edge_signature_counts[edge] = count
    
    print(f"[EDGE] Computed signature counts for {len(edge_signature_counts)} edges")
    
    # Print statistics
    counts = list(edge_signature_counts.values())
    print(f"\n[STATS] Edge signature statistics:")
    print(f"  Min unique signatures: {min(counts)}")
    print(f"  Max unique signatures: {max(counts)}")
    print(f"  Mean unique signatures: {np.mean(counts):.2f}")
    print(f"  Median unique signatures: {np.median(counts):.2f}")
    
    # Generate visualizations
    print(f"\n[VIZ] Generating visualizations...")
    
    # Plot 1: Grid overlay with edges colored by signature diversity
    plot_obstacle_map_with_grid(
        polygons, grid_points, edge_signature_counts,
        (base_x, base_y),
        os.path.join(data_path, "route_grid_overlay.png")
    )
    
    # Plot 2: Grid points colored by data potential
    plot_grid_points_colored_by_potential(
        polygons, grid_points, point_signature_counts,
        (base_x, base_y),
        os.path.join(data_path, "route_point_potential.png")
    )
    
    # Plot 3: Histogram of signature distribution
    plot_signature_distribution_histogram(
        edge_signature_counts,
        os.path.join(data_path, "route_signature_histogram.png")
    )
    
    print(f"\n[COMPLETE] Search route analysis complete!")
    print(f"[COMPLETE] Results saved to {data_path}")


if __name__ == "__main__":
    main()
