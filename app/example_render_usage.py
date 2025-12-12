#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of the new Render class

This demonstrates how to use the Render() class for various visualization tasks
with consistent styling across all plots.
"""

import os
os.environ['ANGULAR_BIN_SIZE'] = '5'
os.environ['RADIAL_BIN_SIZE'] = '10'

from render import Render


def example_basic_usage():
    """Basic usage of Render class with file paths."""
    
    # Initialize with data paths
    renderer = Render(
        obstacle_kml_path="data/feature_map.kml",
        rf_data_path="data/points.yaml"
    )
    
    # Load data (requires center coordinates)
    renderer.load_data(center_lat=51.5, center_lon=-0.1)
    
    # Render different visualizations
    renderer.render(
        obstacle_edge=True,
        obstacle_fill=True,
        rf_data=False,
        use_polar=False,
        fileout="output/xy_obstacles.png"
    )
    
    renderer.render(
        obstacle_edge=True,
        obstacle_fill=False,
        rf_data=True,
        use_polar=False,
        fileout="output/xy_boundaries_rf.png"
    )


def example_convenience_methods():
    """Using convenience methods for common visualizations."""
    
    renderer = Render()
    # Assume data is loaded...
    
    # XY visualizations
    renderer.render_xy_obstacles("output/obstacles.png")
    renderer.render_xy_boundaries_rf("output/boundaries_rf.png")
    renderer.render_xy_obstacles_rf("output/obstacles_rf.png")
    
    # Polar visualizations (requires polar grid data)
    renderer.render_obstacle_polar("output/polar_obstacles.png")
    renderer.render_heatmap_polar("output/polar_heatmap.png")


def example_programmatic_usage():
    """Programmatic usage by setting data directly."""
    import numpy as np
    from shapely.geometry import Polygon
    
    renderer = Render()
    
    # Set polygon data directly
    poly1 = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    poly2 = Polygon([(200, 200), (300, 200), (300, 300), (200, 300)])
    renderer.polygons = [(poly1, "trees"), (poly2, "building")]
    
    # Set RF data
    renderer.local_rf_points = [(50, 50, 0.5), (250, 250, 0.8)]
    
    # Render
    renderer.render_xy_obstacles_rf("output/programmatic.png")
    
    # For polar data
    angles_deg = np.arange(0, 360, 5)
    radii_m = np.arange(0, 100, 10)
    obstacle_grid = np.array([['open']*10 for _ in range(72)])
    
    renderer.set_polar_grid(angles_deg, radii_m, obstacle_grid)
    
    heatmap = np.random.rand(72, 10)
    renderer.set_heatmap(heatmap)
    
    renderer.render_heatmap_polar("output/polar_test.png")


def example_signature_slices():
    """Render signature slices visualization."""
    import numpy as np
    
    renderer = Render()
    
    # Example signature data
    unique_sigs = ['001122', '001133', '002233']
    mean_slices = np.random.rand(3, 10)
    
    # Set radii for proper scaling
    renderer._radii_m = np.arange(0, 100, 10)
    
    renderer.render_signature_slices(
        unique_sigs,
        mean_slices,
        "output/signature_slices.png"
    )


if __name__ == "__main__":
    print("Render class usage examples")
    print("=" * 50)
    print("\n1. Basic usage: render(obstacle_edge, obstacle_fill, rf_data, use_polar, fileout)")
    print("2. Convenience methods: render_xy_obstacles(), render_xy_boundaries_rf(), etc.")
    print("3. Programmatic: Set data directly via renderer.polygons, renderer.local_rf_points")
    print("4. Signature slices: render_signature_slices(unique_sigs, mean_slices, fileout)")
    print("\nAll rendering uses consistent styling from render_utils.py")
