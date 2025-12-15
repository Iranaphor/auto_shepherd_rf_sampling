#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# Extend dataset with new RF measurements
#
# @author      James R. Heselden (github: Iranaphor)
# @maintainer  James R. Heselden (github: Iranaphor)
# @datecreated 12th December 2025
# @datemodified 12th December 2025
# @credits     Developed with assistance from Claude Sonnet 4.5 and GitHub
#              Copilot.
# ###########################################################################

import os

from kml_utils import load_kml_polygons
from utils import (
    load_yaml_points, compute_max_range,
    build_polar_grid, build_points_heatmap
)
from dataset_utils import (
    load_dataset_yaml, save_dataset_yaml,
    build_obstacle_type_map, extract_rf_data_by_signature,
    merge_signatures, prepare_signatures_for_yaml
)
from render_utils import plot_signature_comparison


def load_rf_data(points_yaml, kml_path, dtheta_deg=5.0, dr_m=10.0, max_range_m=None):
    """Load and process RF data into polar coordinates."""
    print("\n=== STEP 1: Loading RF Data ===")
    
    # Load RF points
    center_lat, center_lon, points = load_yaml_points(points_yaml)
    print(f"[RF] Loaded {len(points)} RF points from {points_yaml}")
    
    # Load obstacle polygons
    polygons = load_kml_polygons(kml_path, center_lat, center_lon)
    print(f"[OBSTACLES] Loaded {len(polygons)} polygons from {kml_path}")
    
    # Compute max range if needed
    if max_range_m is None:
        max_range = compute_max_range(polygons, center_lat, center_lon, points)
    else:
        max_range = max_range_m
    
    print(f"[INFO] Using max range: {max_range:.1f} m")
    
    return points, polygons, center_lat, center_lon, max_range


def convert_to_polar_maps(polygons, points, center_lat, center_lon,
                          max_range, dtheta_deg, dr_m):
    """Convert obstacle map and RF data to polar coordinates."""
    print("\n=== STEP 2: Converting to Polar Maps ===")
    
    # Build polar obstacle grid
    angles_deg, radii_m, obstacle_grid = build_polar_grid(
        polygons, max_range, dtheta_deg, dr_m
    )
    print(f"[POLAR] Built obstacle grid: {obstacle_grid.shape}")
    
    # Build RF heatmap
    heatmap = build_points_heatmap(
        points, center_lat, center_lon,
        angles_deg, radii_m, dtheta_deg, dr_m, max_range
    )
    print(f"[POLAR] Built RF heatmap: {heatmap.shape}")
    
    return angles_deg, radii_m, obstacle_grid, heatmap


def associate_signatures(obstacle_grid, heatmap, type_to_tag):
    """Associate obstacle signatures with RF data."""
    print("\n=== STEP 3: Associating Signatures ===")
    
    # Extract RF data by signature
    sig_to_rf = extract_rf_data_by_signature(obstacle_grid, heatmap, type_to_tag)
    
    print(f"[SIGNATURES] Found {len(sig_to_rf)} unique obstacle signatures")
    for sig, rf_arrays in list(sig_to_rf.items())[:5]:
        print(f"  {sig}: {len(rf_arrays)} rays")
    
    return sig_to_rf


def extend_dataset(existing_signatures, new_sig_data, rf_type='ssid-espnow'):
    """Extend existing dataset with new RF measurements."""
    print("\n=== STEP 4: Extending Dataset ===")
    
    before_count = len(existing_signatures)
    
    # Merge signatures
    merged_signatures = merge_signatures(
        existing_signatures,
        new_sig_data,
        rf_type=rf_type
    )
    
    after_count = len(merged_signatures)
    new_count = after_count - before_count
    
    print(f"[EXTEND] Signatures before: {before_count}")
    print(f"[EXTEND] Signatures after: {after_count}")
    print(f"[EXTEND] New signatures added: {new_count}")
    
    # Count extended signatures
    extended_count = 0
    for sig in existing_signatures.keys():
        if sig in merged_signatures:
            before_data_count = len(existing_signatures[sig].get(rf_type, []))
            after_data_count = len(merged_signatures[sig].get(rf_type, []))
            if after_data_count > before_data_count:
                extended_count += 1
    
    print(f"[EXTEND] Existing signatures extended: {extended_count}")
    
    return merged_signatures


def generate_comparison_plots(before_sigs, after_sigs, radii_m, output_path, polygons, points, center_lat, center_lon):
    """Generate before and after visualization."""
    print("\n=== STEP 5: Generating Visualizations ===")
    
    # Extract mean RF values for plotting
    rf_type = 'ssid-espnow'
    
    def extract_mean_values(signatures):
        """Extract mean RF values from signature data."""
        result = {}
        for sig, sig_data in signatures.items():
            if isinstance(sig_data, dict) and rf_type in sig_data:
                rf_arrays = sig_data[rf_type]
                if rf_arrays:
                    # Compute mean across all arrays
                    import numpy as np
                    all_vals = []
                    max_len = max(len(arr) for arr in rf_arrays)
                    for arr in rf_arrays:
                        # Convert values to float, handling strings and None
                        numeric_arr = []
                        for v in arr:
                            try:
                                numeric_arr.append(float(v) if v is not None else np.nan)
                            except (ValueError, TypeError):
                                numeric_arr.append(np.nan)
                        # Pad to max length
                        padded = numeric_arr + [np.nan] * (max_len - len(numeric_arr))
                        all_vals.append(padded)
                    
                    # Convert to numpy and compute mean
                    arr_np = np.array(all_vals, dtype=float)
                    mean_vals = np.nanmean(arr_np, axis=0)
                    result[sig] = mean_vals.tolist()
        return result
    
    before_data = extract_mean_values(before_sigs)
    after_data = extract_mean_values(after_sigs)
    
    # Load obstacles for color mapping
    from dataset_utils import load_dataset_yaml
    obstacles_for_plot, _ = load_dataset_yaml(os.getenv('DATASET_PATH'))
    
    # Generate plot
    plot_signature_comparison(
        before_sigs, after_sigs,
        before_data, after_data,
        radii_m,
        output_path,
        polygons,
        points,
        center_lat,
        center_lon,
        obstacles_for_plot,
        title_prefix="RF Dataset"
    )


def main():
    """Main execution function."""
    # Get configuration from environment variables
    data_path = os.getenv('DATA_PATH')
    dataset_path = os.getenv('DATASET_PATH')
    output_path = os.getenv('OUTPUT_PATH')
    dtheta = float(os.getenv('DTHETA', '5.0'))
    dr = float(os.getenv('DR', '10.0'))
    max_range = float(os.getenv('MAX_RANGE')) if os.getenv('MAX_RANGE') else None
    rf_type = os.getenv('RF_TYPE', 'ssid-espnow')
    plot_output = os.getenv('PLOT_OUTPUT')
    
    # Validate required parameters
    if not data_path:
        raise ValueError("DATA_PATH environment variable is required")
    if not dataset_path:
        raise ValueError("DATASET_PATH environment variable is required")
    
    # Set default output paths
    if output_path is None:
        output_path = dataset_path
    
    if plot_output is None:
        base_dir = os.path.dirname(output_path)
        plot_output = os.path.join(base_dir, 'dataset_comparison.png')
    
    # File paths
    points_yaml = os.path.join(data_path, 'points.yaml')
    kml_path = os.path.join(data_path, 'feature_map.kml')
    
    # Load existing dataset
    print(f"\n=== Loading Existing Dataset ===")
    print(f"[DATASET] Loading from {dataset_path}")
    obstacles, existing_signatures = load_dataset_yaml(dataset_path)
    
    # Build obstacle type mappings
    type_to_tag, tag_to_type = build_obstacle_type_map(obstacles)
    
    # Step 1: Load RF data
    points, polygons, center_lat, center_lon, max_range = load_rf_data(
        points_yaml, kml_path,
        dtheta_deg=dtheta,
        dr_m=dr,
        max_range_m=max_range
    )
    
    # Step 2: Convert to polar maps
    angles_deg, radii_m, obstacle_grid, heatmap = convert_to_polar_maps(
        polygons, points, center_lat, center_lon,
        max_range, dtheta, dr
    )
    
    # Step 3: Associate signatures
    new_sig_data = associate_signatures(obstacle_grid, heatmap, type_to_tag)
    
    # Step 4: Extend dataset
    merged_signatures = extend_dataset(
        existing_signatures,
        new_sig_data,
        rf_type=rf_type
    )
    
    # Step 5: Generate visualizations
    generate_comparison_plots(
        existing_signatures,
        merged_signatures,
        radii_m,
        plot_output,
        polygons,
        points,
        center_lat,
        center_lon
    )
    
    # Prepare and save dataset
    print("\n=== Saving Dataset ===")
    prepared_signatures = prepare_signatures_for_yaml(merged_signatures, radii_m)
    save_dataset_yaml(output_path, obstacles, prepared_signatures)
    
    print(f"\n[SUCCESS] Dataset extended and saved to {output_path}")
    print(f"[SUCCESS] Comparison plot saved to {plot_output}")


if __name__ == "__main__":
    main()
