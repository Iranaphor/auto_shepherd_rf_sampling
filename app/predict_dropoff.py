#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# Interactive RF signal prediction between two points
#
# @author      James R. Heselden (github: Iranaphor)
# @maintainer  James R. Heselden (github: Iranaphor)
# @datecreated 13th December 2025
# @datemodified 13th December 2025
# @credits     Developed with assistance from Claude Sonnet 4.5 and GitHub
#              Copilot.
# ###########################################################################

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from kml_utils import load_kml_polygons, latlon_to_local_xy
from utils import load_yaml_points, build_polar_grid
from dataset_utils import load_dataset_yaml, build_obstacle_type_map
from render_utils import compute_xy_extent, get_default_dpi


class RFDropoffPredictor:
    """Interactive tool for predicting RF signal dropoff between two points."""
    
    def __init__(self, dataset_path, kml_path, points_yaml, rf_type='ssid-espnow'):
        """Initialize the predictor with dataset and obstacle map."""
        print("[INIT] Loading dataset and obstacle map...")
        
        # Load dataset
        self.obstacles, self.signatures = load_dataset_yaml(dataset_path)
        self.rf_type = rf_type
        print(f"[DATASET] Loaded {len(self.signatures)} signatures")
        
        # Build obstacle type mappings
        self.type_to_tag, self.tag_to_type = build_obstacle_type_map(self.obstacles)
        
        # Build obstacle color map from dataset
        self.obstacle_colors = {}
        for obs in self.obstacles:
            obs_type = obs.get('type', 'unknown')
            color = obs.get('rendering_colour', '#cccccc')
            self.obstacle_colors[obs_type] = color
        
        print(f"[DATASET] Obstacle colors: {self.obstacle_colors}")
        
        # Load RF points for center coordinates
        self.center_lat, self.center_lon, points = load_yaml_points(points_yaml)
        
        # Load obstacle polygons
        self.polygons = load_kml_polygons(kml_path, self.center_lat, self.center_lon)
        print(f"[OBSTACLES] Loaded {len(self.polygons)} polygons")
        
        # State for point selection
        self.selected_points = []
        self.arrow_patch = None
        self.coverage_patches = []
        self.point_markers = []  # Track plotted points
        self.signal_lines = []   # Track signal plot lines
        
        # Create figure
        self.setup_figure()
        
    def setup_figure(self):
        """Set up the matplotlib figure with two subplots."""
        self.fig = plt.figure(figsize=(14, 10))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # Top: XY obstacle map
        self.ax_map = self.fig.add_subplot(gs[0])
        self.ax_map.set_title("Click two points to predict RF signal dropoff\n(First click: transmitter, Second click: receiver)")
        self.ax_map.set_xlabel("X (m)")
        self.ax_map.set_ylabel("Y (m)")
        self.ax_map.set_aspect('equal')
        
        # Draw obstacles using colors from dataset YAML
        extent = compute_xy_extent(self.polygons, None)
        self.ax_map.set_xlim(extent[0], extent[1])
        self.ax_map.set_ylim(extent[2], extent[3])
        
        for poly, cat in self.polygons:
            # Use color from dataset YAML file
            # Handle plural/singular mismatch (trees vs tree)
            color = self.obstacle_colors.get(cat)
            if color is None and cat == 'trees':
                color = self.obstacle_colors.get('tree', '#cccccc')
            elif color is None:
                color = '#cccccc'
            x, y = poly.exterior.xy
            self.ax_map.fill(x, y, facecolor=color, edgecolor="black", alpha=0.8, linewidth=0.7)
        
        # Bottom: RF signal strength vs distance
        self.ax_signal = self.fig.add_subplot(gs[1])
        self.ax_signal.set_title("RF Signal Strength vs Distance")
        self.ax_signal.set_xlabel("Distance (m)")
        self.ax_signal.set_ylabel("RSSI (dBm)")
        self.ax_signal.grid(True, alpha=0.3)
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax_map:
            return
        
        x, y = event.xdata, event.ydata
        
        # Add point
        self.selected_points.append((x, y))
        
        # Plot the point
        if len(self.selected_points) == 1:
            # First point: transmitter (green)
            marker = self.ax_map.plot(x, y, 'go', markersize=10, label='Transmitter', zorder=10)
            self.point_markers.extend(marker)
            self.ax_map.legend()
            print(f"[SELECT] Transmitter at ({x:.1f}, {y:.1f})")
            
            # Show areas where signature data is available
            self.show_available_signatures(x, y)
            
        elif len(self.selected_points) == 2:
            # Second point: receiver (red)
            marker = self.ax_map.plot(x, y, 'ro', markersize=10, label='Receiver', zorder=10)
            self.point_markers.extend(marker)
            self.ax_map.legend()
            print(f"[SELECT] Receiver at ({x:.1f}, {y:.1f})")
            
            # Clear coverage patches
            for patch in self.coverage_patches:
                patch.remove()
            self.coverage_patches = []
            
            # Draw arrow between points
            self.draw_connection()
            
            # Predict signal dropoff
            self.predict_signal_dropoff()
            
            print("\n[READY] Click again to reset and start a new prediction")
        
        elif len(self.selected_points) == 3:
            # Third click: reset everything
            print("\n[RESET] Clearing previous selection...")
            
            # Clear point markers
            for marker in self.point_markers:
                marker.remove()
            self.point_markers = []
            
            # Clear legend
            legend = self.ax_map.get_legend()
            if legend:
                legend.remove()
            
            # Clear coverage patches
            for patch in self.coverage_patches:
                patch.remove()
            self.coverage_patches = []
            
            # Clear arrow
            if self.arrow_patch:
                self.arrow_patch.remove()
                self.arrow_patch = None
            
            # Clear signal plot
            self.ax_signal.clear()
            self.ax_signal.set_title("RF Signal Strength vs Distance")
            self.ax_signal.set_xlabel("Distance (m)")
            self.ax_signal.set_ylabel("RSSI (dBm)")
            self.ax_signal.grid(True, alpha=0.3)
            
            # Reset selection
            self.selected_points = []
            print("[READY] Click first point (transmitter)")
        
        self.fig.canvas.draw()
    
    def show_available_signatures(self, x0, y0):
        """Highlight areas where signature data is available from transmitter point."""
        from matplotlib.patches import Wedge
        import matplotlib.patches as mpatches
        
        # Clear previous coverage
        for patch in self.coverage_patches:
            patch.remove()
        self.coverage_patches = []
        
        # Build polar grid to find available signatures
        dtheta_deg = 5.0
        dr_m = 10.0
        max_range = 500  # Check up to 500m
        
        angles_deg = np.arange(0, 360, dtheta_deg)
        n_r = int(max_range / dr_m)
        radii_m = np.arange(dr_m / 2, max_range, dr_m)[:n_r]
        
        from utils import build_polar_obstacle_grid_for_center
        obstacle_grid = build_polar_obstacle_grid_for_center(
            self.polygons, x0, y0, angles_deg, radii_m
        )
        
        # For each angle, check if we have signature data and find max range
        available_directions = {}  # angle -> max_range
        
        for angle_idx, angle in enumerate(angles_deg):
            ray_signature = obstacle_grid[angle_idx, :]
            sig_chars = []
            for cat in ray_signature:
                tag = self.type_to_tag.get(cat, 0) if isinstance(cat, str) else 0
                sig_chars.append(str(tag))
            sig_str = 's' + ''.join(sig_chars)
            
            # Check if this signature exists in dataset and find max data range
            matching_sigs = self.find_matching_signatures(sig_str)
            if matching_sigs:
                # Find the maximum range available in the matching signatures
                max_data_range = 0
                for sig_key in matching_sigs:
                    sig_data = self.signatures[sig_key]
                    if self.rf_type in sig_data:
                        bins = sig_data.get('bins', [])
                        rf_arrays = sig_data[self.rf_type]
                        # Find the furthest distance with actual data
                        for rf_values in rf_arrays:
                            for j in range(len(rf_values) - 1, -1, -1):
                                val = rf_values[j]
                                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                    if j < len(bins):
                                        max_data_range = max(max_data_range, bins[j])
                                    break
                
                if max_data_range > 0:
                    available_directions[angle] = max_data_range
        
        # Draw wedges for angles with available data, extending only to data range
        if available_directions:
            print(f"[COVERAGE] Found data for {len(available_directions)}/{len(angles_deg)} directions")
            
            # Draw wedge for each direction with its specific range
            for angle, data_range in available_directions.items():
                wedge = Wedge((x0, y0), data_range, angle - dtheta_deg/2, angle + dtheta_deg/2,
                            facecolor='green', edgecolor='none', alpha=0.15, zorder=5)
                self.ax_map.add_patch(wedge)
                self.coverage_patches.append(wedge)
        else:
            print(f"[WARNING] No signature data available from this location")
    
    def draw_connection(self):
        """Draw arrow connecting the two selected points."""
        if self.arrow_patch:
            self.arrow_patch.remove()
        
        x0, y0 = self.selected_points[0]
        x1, y1 = self.selected_points[1]
        
        self.arrow_patch = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle='->', mutation_scale=20,
            linewidth=2, color='blue', zorder=9
        )
        self.ax_map.add_patch(self.arrow_patch)
    
    def predict_signal_dropoff(self):
        """Predict RF signal dropoff between two selected points."""
        x0, y0 = self.selected_points[0]
        x1, y1 = self.selected_points[1]
        
        # Compute distance and bearing
        dx = x1 - x0
        dy = y1 - y0
        total_distance = np.sqrt(dx**2 + dy**2)
        bearing_deg = np.degrees(np.arctan2(dx, dy))
        if bearing_deg < 0:
            bearing_deg += 360
        
        print(f"[PREDICT] Distance: {total_distance:.1f}m, Bearing: {bearing_deg:.1f}°")
        
        # Build polar grid centered at first point
        # Use fixed parameters for polar grid
        dtheta_deg = 5.0
        dr_m = 10.0
        max_range = max(total_distance * 1.2, 100)
        
        # Build angle and radius bins
        angles_deg = np.arange(0, 360, dtheta_deg)
        n_r = int(max_range / dr_m)
        radii_m = np.arange(dr_m / 2, max_range, dr_m)[:n_r]
        
        # Build polar grid from this point
        from utils import build_polar_obstacle_grid_for_center
        obstacle_grid = build_polar_obstacle_grid_for_center(
            self.polygons, x0, y0, angles_deg, radii_m
        )
        
        print(f"[POLAR] Built obstacle grid: {obstacle_grid.shape}")
        
        # Find the angle bin closest to the bearing
        angle_idx = int(round(bearing_deg / dtheta_deg)) % len(angles_deg)
        
        # Extract obstacle signature along this ray (convert categories to tags)
        ray_signature = obstacle_grid[angle_idx, :]
        sig_chars = []
        for cat in ray_signature:
            if isinstance(cat, str):
                tag = self.type_to_tag.get(cat, 0)
                sig_chars.append(str(tag))
            else:
                sig_chars.append('0')
        sig_str = 's' + ''.join(sig_chars)
        
        print(f"[SIGNATURE] Ray {angle_idx} ({angles_deg[angle_idx]:.1f}°): {sig_str[:50]}...")
        
        # Collect signatures from nearby angles (within ±5 degree arc)
        # Since dtheta_deg=5, we check ±1 bin = ±5 degrees
        angle_tolerance = 1
        
        all_matching_sigs = set()
        angles_to_check = []
        
        # Check the primary angle and nearby angles (±5°)
        for offset in range(-angle_tolerance, angle_tolerance + 1):
            check_idx = (angle_idx + offset) % len(angles_deg)
            check_angle = angles_deg[check_idx]
            angles_to_check.append(check_angle)
            
            # Build signature for this angle
            ray_sig = obstacle_grid[check_idx, :]
            sig_chars = []
            for cat in ray_sig:
                if isinstance(cat, str):
                    tag = self.type_to_tag.get(cat, 0)
                    sig_chars.append(str(tag))
                else:
                    sig_chars.append('0')
            check_sig_str = 's' + ''.join(sig_chars)
            
            # Find matches for this signature
            matches = self.find_matching_signatures(check_sig_str, max_range_m=total_distance)
            all_matching_sigs.update(matches)
        
        matching_sigs = list(all_matching_sigs)
        angle_range_str = f"{angles_to_check[0]:.0f}°-{angles_to_check[-1]:.0f}°" if len(angles_to_check) > 1 else f"{angles_to_check[0]:.0f}°"
        print(f"[MATCH] Checked {len(angles_to_check)} angles ({angle_range_str}), found {len(matching_sigs)} signature(s)")
        
        # Clear previous plot
        self.ax_signal.clear()
        self.ax_signal.set_title(f"RF Signal Strength vs Distance (Bearing: {bearing_deg:.1f}°)")
        self.ax_signal.set_xlabel("Distance (m)")
        self.ax_signal.set_ylabel("RSSI (dBm)")
        self.ax_signal.grid(True, alpha=0.3)
        
        if matching_sigs:
            print(f"[PLOT] Plotting {len(matching_sigs)} matching signature(s)")
            
            # Plot each matching signature's RF data
            for sig_key in matching_sigs:
                sig_data = self.signatures[sig_key]
                if self.rf_type in sig_data:
                    bins = sig_data.get('bins', [])
                    rf_arrays = sig_data[self.rf_type]
                    
                    # Plot each ray in the signature
                    for i, rf_values in enumerate(rf_arrays):
                        distances = bins[:len(rf_values)]
                        rssi = []
                        dist = []
                        
                        # Collect all available data points
                        for j, val in enumerate(rf_values):
                            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                d = distances[j]
                                rssi.append(float(val))
                                dist.append(d)
                        
                        # If we have data and the click is beyond the data range,
                        # plot all available data. Otherwise trim at click distance.
                        if rssi and dist:
                            max_data_dist = max(dist)
                            
                            if total_distance > max_data_dist:
                                # Click is beyond data range, show all available data
                                plot_dist = dist
                                plot_rssi = rssi
                            else:
                                # Click is within data range, trim to click distance
                                plot_dist = []
                                plot_rssi = []
                                for j, d in enumerate(dist):
                                    if d <= total_distance:
                                        plot_dist.append(d)
                                        plot_rssi.append(rssi[j])
                                    elif len(plot_dist) > 0 and j > 0:
                                        # Interpolate final point at click distance
                                        prev_d = dist[j-1]
                                        prev_rssi_val = rssi[j-1]
                                        curr_rssi_val = rssi[j]
                                        t = (total_distance - prev_d) / (d - prev_d)
                                        interp_rssi = prev_rssi_val + t * (curr_rssi_val - prev_rssi_val)
                                        plot_dist.append(total_distance)
                                        plot_rssi.append(interp_rssi)
                                        break
                        
                            if plot_dist:
                                alpha = 0.7 if len(matching_sigs) == 1 else 0.4
                                self.ax_signal.plot(plot_dist, plot_rssi, '-o', alpha=alpha, 
                                                  markersize=3, linewidth=1.5)
            
            # Mark the selected distance
            self.ax_signal.axvline(total_distance, color='red', linestyle='--', 
                                  linewidth=2, label=f'Selected distance: {total_distance:.1f}m')
            self.ax_signal.legend()
            
        else:
            print(f"[WARNING] No matching signatures found for {sig_str[:50]}...")
            self.ax_signal.text(0.5, 0.5, "No matching signature found in dataset",
                              ha='center', va='center', transform=self.ax_signal.transAxes,
                              fontsize=12, color='red')
        
        self.fig.canvas.draw()
    
    def find_matching_signatures(self, query_sig, max_range_m=None):
        """Find signatures in dataset that match the query signature.
        
        Args:
            query_sig: Signature string to match
            max_range_m: Maximum range in meters (used to compute minimum match length)
        """
        matches = []
        
        # Exact match first
        if query_sig in self.signatures:
            matches.append(query_sig)
            return matches
        
        # Partial match: find signatures with similar prefix
        # If click is beyond data range, be more lenient
        if max_range_m:
            # Compute how many bins we're checking (10m per bin)
            query_bins = len(query_sig) - 1  # -1 for the 's' prefix
            target_bins = int(max_range_m / 10.0)
            
            # If query is longer than target, we're clicking beyond
            if query_bins > target_bins:
                # Match up to the target distance
                min_match_length = max(10, target_bins)
            else:
                # Match most of the signature
                min_match_length = max(10, query_bins * 2 // 3)
        else:
            min_match_length = min(20, len(query_sig) // 2)
        
        for sig_key in self.signatures.keys():
            # Compare character by character
            match_len = 0
            for i in range(min(len(query_sig), len(sig_key))):
                if query_sig[i] == sig_key[i]:
                    match_len += 1
                else:
                    break
            
            # If enough characters match, consider it a match
            if match_len >= min_match_length:
                matches.append(sig_key)
        
        return matches
    
    def run(self):
        """Start the interactive predictor."""
        print("\n" + "="*60)
        print("RF SIGNAL DROPOFF PREDICTOR")
        print("="*60)
        print("Instructions:")
        print("  1. Click on a point to set the transmitter location (green)")
        print("  2. Click on a second point to set the receiver location (red)")
        print("  3. The system will display predicted signal strength")
        print("  4. Click two new points to make another prediction")
        print("="*60 + "\n")
        
        plt.show()


def main():
    """Main execution function."""
    # Get configuration from environment variables
    data_path = os.getenv('DATA_PATH')
    dataset_path = os.getenv('DATASET_PATH')
    rf_type = os.getenv('RF_TYPE', 'ssid-espnow')
    
    # Validate required parameters
    if not data_path:
        raise ValueError("DATA_PATH environment variable is required")
    if not dataset_path:
        raise ValueError("DATASET_PATH environment variable is required")
    
    # File paths
    points_yaml = os.path.join(data_path, 'points.yaml')
    kml_path = os.path.join(data_path, 'feature_map.kml')
    
    # Create and run predictor
    predictor = RFDropoffPredictor(
        dataset_path=dataset_path,
        kml_path=kml_path,
        points_yaml=points_yaml,
        rf_type=rf_type
    )
    
    predictor.run()


if __name__ == "__main__":
    main()
