#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# Generate polar obstacle map and RF heatmap from YAML and KML data
#
# @author      James R. Heselden (github: Iranaphor)
# @maintainer  James R. Heselden (github: Iranaphor)
# @datecreated 21st November 2025
# @datemodified 12th December 2025
# @credits     Developed with assistance from Claude Sonnet 4.5 and GitHub
#              Copilot.
# ###########################################################################

import os

from utils import (
    load_yaml_points, generate_points_yaml,
    compute_max_range, build_polar_grid,
    plot_xy_obstacle_map, plot_xy_obstacle_boundaries_with_rf,
    plot_xy_obstacles_with_rf, plot_obstacle_grid,
    build_points_heatmap, plot_heatmap_with_boundaries,
    compute_obstacle_signatures_and_slices, plot_signature_slices,
    compute_signature_knowledge_from_slices,
    plot_sampling_and_signature_coverage_data_map
)
from kml_utils import load_kml_polygons
from find_base import select_top_sampling_locations


def main():
    """Main execution function for RF polar map generation."""
    path = os.getenv('DATA_PATH')

    # File paths
    csv_path = os.path.join(path, "all.csv")
    yaml_path = os.path.join(path, "points.yaml")
    kml_path = os.path.join(path, "feature_map.kml")
    center = [os.getenv('CENTER_LAT'), os.getenv('CENTER_LON')]

    # Construct YAML from CSV data
    print(f"[CVS] Updated yaml to match data in {csv_path}.")
    generate_points_yaml(csv_path, yaml_path, center)

    # Load YAML data
    center_lat, center_lon, points = load_yaml_points(yaml_path)
    print(f"[YAML] Loaded {len(points)} RF points. Center: {center_lat}, {center_lon}")

    # Load KML polygons
    polygons = load_kml_polygons(kml_path, center_lat, center_lon)

    # Determine max range (default: compute from data)
    max_range = compute_max_range(polygons, center_lat, center_lon, points)
    print(f"[INFO] Using max range: {max_range:.1f} m")

    # Default bin sizes
    DTHETA_DEG = 5.0
    DR_M = 10.0

    # Build polar obstacle grid
    angles_deg, radii_m, obstacle_grid = build_polar_grid(
        polygons, max_range, DTHETA_DEG, DR_M
    )

    # Generate XY plane visualizations
    plot_xy_obstacle_map(
        polygons,
        out_path=os.path.join(path, "xy_obstacle_map.png")
    )
    plot_xy_obstacle_boundaries_with_rf(
        polygons,
        points,
        center_lat,
        center_lon,
        out_path=os.path.join(path, "xy_obstacle_boundaries_rf.png")
    )
    plot_xy_obstacles_with_rf(
        polygons,
        points,
        center_lat,
        out_path=os.path.join(path, "xy_obstacles_rf.png")
    )

    # Generate polar obstacle visualization
    plot_obstacle_grid(angles_deg,
        radii_m,
        obstacle_grid,
        os.path.join(path, "obstacles_polar.png")
    )

    # Build and plot RF heatmap
    heatmap = build_points_heatmap(
        points, center_lat, center_lon,
        angles_deg, radii_m, DTHETA_DEG, DR_M, max_range
    )

    plot_heatmap_with_boundaries(
        angles_deg, radii_m, heatmap, obstacle_grid, os.path.join(path, "points_heatmap_polar.png")
    )

    # Compute and plot signature slices
    unique_sigs, mean_slices, counts = compute_obstacle_signatures_and_slices(
        obstacle_grid, heatmap
    )
    plot_signature_slices(
        radii_m, unique_sigs, mean_slices, os.path.join(path, "signature_slices_polar.png")
    )

    # Build knowledge from signature-slice map
    # Build knowledge directly from what the signature-slice map shows
    knowledge = compute_signature_knowledge_from_slices(unique_sigs, mean_slices)

    samples, all_centres = select_top_sampling_locations(
    knowledge = compute_signature_knowledge_from_slices(unique_sigs, mean_slices)

    # Select optimal sampling locations
    samples, all_centres = select_top_sampling_locations(
        polygons,
        angles_deg,
        radii_m,
        knowledge,
        top_k=5,
        min_spacing_m=300.0,
        candidate_grid_step_m=500.0,
        fine_grid_step_m=150.0,
        max_walk_radius_m=200.0,
        include_existing_base=True,
        existing_weight=3.0,
        new_weight=1.0,
        coarse_refine_factor=4,
        max_hamming_frac=0.1,
    )

    # Generate sampling and coverage visualization
    plot_sampling_and_signature_coverage_data_map(
        polygons,
        samples,
        unique_sigs,
        mean_slices,
        radii_m,
        knowledge,
        out_path=os.path.join(path, "sampling_and_signature_coverage.png"),
        max_walk_radius_m=200.0,
        include_new_signatures=True,
        all_centres=all_centres,
    )


if __name__ == "__main__":
    main()