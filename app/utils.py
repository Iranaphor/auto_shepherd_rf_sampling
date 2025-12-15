#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# Utility functions for RF sampling and obstacle analysis
#
# @author      James R. Heselden (github: iranaphor)
# @maintainer  James R. Heselden (github: iranaphor)
# @datecreated 27th November 2025
# ###########################################################################

import os
import csv
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, to_rgba
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

from shapely.geometry import Point
from collections import defaultdict

from config import *
from kml_utils import latlon_to_local_xy, load_kml_polygons


# =========================
# DATA LOADING
# =========================

def load_yaml_points(yaml_path):
    """Load RF data points from YAML file."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    center_lat, center_lon = data["center"]
    points = data["gps_coords_xyv"]  # list of [lat, lon, value]

    return center_lat, center_lon, points


def generate_points_yaml(csv_path, yaml_path, center):
    """Generate YAML points file from CSV data."""
    if not os.path.exists(csv_path):
        print('[CSV] file does not exist')
        return

    with open(csv_path, newline="") as csvfile, open(yaml_path, "w") as yamlfile:
        reader = csv.reader(csvfile)

        # Header comments
        yamlfile.write("# ----------------------\n")
        yamlfile.write("# YAML\n")
        yamlfile.write("#\n")
        yamlfile.write("# ----------------------\n")

        # Center line
        yamlfile.write(f"center: [{center[0]}, {center[1]}]\n")
        yamlfile.write("gps_coords_xyv:\n")

        # Each row is [v, x, y, d, f]
        for row in reader:
            # skip empty lines
            if not row or all(cell.strip() == "" for cell in row):
                continue

            v = float(row[0])
            x = float(row[1])
            y = float(row[2])

            yamlfile.write(f"  -  [{x}, {y}, {v}]\n")


# =========================
# POLAR GRID OPERATIONS
# =========================

def compute_max_range(polygons, center_lat, center_lon, points):
    """
    Compute max needed range from polygon vertices + points.
    Polygons are already in local XY.
    """
    max_r = 0.0

    # From polygons (already local xy)
    for poly, _cat in polygons:
        for x, y in poly.exterior.coords:
            r = math.hypot(x, y)
            max_r = max(max_r, r)
        for ring in poly.interiors:
            for x, y in ring.coords:
                r = math.hypot(x, y)
                max_r = max(max_r, r)

    # From points (convert to local xy)
    for lat, lon, _v in points:
        x, y = latlon_to_local_xy(lat, lon, center_lat, center_lon)
        r = math.hypot(x, y)
        max_r = max(max_r, r)

    return max_r * 1.05  # small margin


def build_polar_grid(polygons, max_range_m, dtheta_deg, dr_m):
    """
    Build a polar obstacle grid:
        - rows: angle bins (0..360)
        - cols: radius bins (0..max_range_m)

    Returns:
        angles_deg (N_theta,)
        radii_m (N_r,)
        obstacle_grid (N_theta, N_r) with category strings
    """
    n_theta = int(360.0 / dtheta_deg)
    n_r = int(max_range_m / dr_m)

    angles_deg = (np.arange(n_theta) + 0.5) * dtheta_deg
    radii_m = (np.arange(n_r) + 0.5) * dr_m

    obstacle_grid = np.empty((n_theta, n_r), dtype=object)
    obstacle_grid[:] = "unknown"

    shapely_polys = [(poly, cat) for (poly, cat) in polygons]

    for ith, theta_deg in enumerate(angles_deg):
        theta_rad = math.radians(theta_deg)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)

        for ir, r in enumerate(radii_m):
            x = r * cos_t
            y = r * sin_t
            pt = Point(x, y)

            # find first polygon that contains this point
            for poly, cat in shapely_polys:
                if poly.contains(pt):
                    obstacle_grid[ith, ir] = cat
                    break
            # otherwise stays "unknown"

    return angles_deg, radii_m, obstacle_grid


def build_polar_obstacle_grid_for_center(polygons, center_x, center_y,
                                         angles_deg, radii_m):
    """
    Build a polar obstacle grid around an arbitrary XY centre.

    polygons   : list of (shapely Polygon in local XY, category_str)
    center_x,y : base station location in that same XY frame
    angles_deg : list/array of angle bin centres in degrees (0..360)
    radii_m    : list/array of radius bin centres in metres

    Returns:
        obstacle_grid_center : (n_theta, n_r) array of category strings
    """
    n_theta = len(angles_deg)
    n_r = len(radii_m)

    obstacle_grid_center = np.empty((n_theta, n_r), dtype=object)
    obstacle_grid_center[:] = "unknown"

    shapely_polys = [(poly, cat) for (poly, cat) in polygons]

    for ith, theta_deg in enumerate(angles_deg):
        theta_rad = np.deg2rad(theta_deg)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        for jr, r in enumerate(radii_m):
            x = center_x + r * cos_t
            y = center_y + r * sin_t
            pt = Point(x, y)

            for poly, cat in shapely_polys:
                if poly.contains(pt):
                    obstacle_grid_center[ith, jr] = cat
                    break

    return obstacle_grid_center


def build_points_heatmap(points, center_lat, center_lon,
                         angles_deg, radii_m, dtheta_deg, dr_m, max_range_m):
    """
    Create a polar heatmap (mean of v) on the same grid as obstacles.
    """
    n_theta = len(angles_deg)
    n_r = len(radii_m)

    sum_grid = np.zeros((n_theta, n_r), dtype=float)
    count_grid = np.zeros((n_theta, n_r), dtype=int)

    for lat, lon, v in points:
        x, y = latlon_to_local_xy(lat, lon, center_lat, center_lon)
        r = math.hypot(x, y)
        if r <= 0 or r > max_range_m:
            continue

        theta_rad = math.atan2(y, x)
        theta_deg = math.degrees(theta_rad)
        if theta_deg < 0:
            theta_deg += 360.0

        ith = int(theta_deg / dtheta_deg)
        if ith < 0 or ith >= n_theta:
            continue

        ir = int(r / dr_m)
        if ir < 0 or ir >= n_r:
            continue

        sum_grid[ith, ir] += v
        count_grid[ith, ir] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        heatmap = sum_grid / count_grid
    heatmap[count_grid == 0] = np.nan

    return heatmap


# =========================
# SIGNATURE ANALYSIS
# =========================

def compute_obstacle_signatures_and_slices(obstacle_grid, heatmap):
    """
    For each angle row (ray) in obstacle_grid, build a signature string,
    then group rays with the same signature and average their heatmap values.

    Returns:
        unique_sigs : list[str]  # sorted unique signature strings
        mean_slices : (num_sigs, n_r) array of mean values (NaNs where no data)
        counts      : list[int]  # number of rays per signature
    """
    n_theta, n_r = obstacle_grid.shape

    # Build a signature string per ray
    sig_to_indices = {}
    for ith in range(n_theta):
        cats = obstacle_grid[ith, :]
        sig = "".join(OBSTACLE_CODES.get(c, OBSTACLE_CODES["unknown"]) for c in cats)
        if sig not in sig_to_indices:
            sig_to_indices[sig] = []
        sig_to_indices[sig].append(ith)

    # Sort signatures by their string ID
    unique_sigs = sorted(sig_to_indices.keys())
    num_sigs = len(unique_sigs)

    mean_slices = np.full((num_sigs, n_r), np.nan, dtype=float)
    counts = []

    for i, sig in enumerate(unique_sigs):
        idxs = sig_to_indices[sig]
        counts.append(len(idxs))

        vals = heatmap[idxs, :]  # shape: (num_rays_in_group, n_r)
        with np.errstate(invalid="ignore"):
            mean_vals = np.nanmean(vals, axis=0)
        mean_slices[i, :] = mean_vals

    # Some debug prints
    print(f"[SIGNATURES] Obstacle codes: {OBSTACLE_CODES}")
    print(f"[SIGNATURES] Found {num_sigs} unique obstacle signatures.")
    for i, (sig, cnt) in enumerate(zip(unique_sigs[:10], counts[:10])):
        print(f"  Signature row {i}: '{sig}' from {cnt} rays")

    return unique_sigs, mean_slices, counts


def compute_signature_knowledge(obstacle_grid, heatmap):
    """
    From the existing polar obstacle grid + heatmap, build a dict:
        knowledge[sig] -> boolean array of length n_r
    where knowledge[sig][j] == True iff there is at least one ray with
    that signature and a non-NaN measurement at radius bin j.
    """
    n_theta, n_r = obstacle_grid.shape
    sampled_mask = ~np.isnan(heatmap)  # True where we have data

    sig_to_rows = defaultdict(list)
    for ith in range(n_theta):
        cats = obstacle_grid[ith, :]
        sig = "".join(OBSTACLE_CODES.get(c, OBSTACLE_CODES["unknown"]) for c in cats)
        sig_to_rows[sig].append(ith)

    knowledge = {}
    for sig, rows in sig_to_rows.items():
        has_any = sampled_mask[rows, :].any(axis=0)  # any data in any row with this sig
        knowledge[sig] = has_any

    return knowledge


def compute_signature_knowledge_from_slices(unique_sigs, mean_slices):
    """
    Build a knowledge dict directly from the signature-slice map:
        knowledge[sig] -> bool[n_r]
    where knowledge[sig][j] == True iff there is *any* existing data for
    that signature at radius bin j (i.e. mean_slices is not NaN there).
    """
    num_sigs, n_r = mean_slices.shape
    knowledge = {}

    for i, sig in enumerate(unique_sigs):
        row = mean_slices[i, :]        # shape (n_r,)
        knowledge[sig] = ~np.isnan(row)

    return knowledge


# =========================
# PLOTTING FUNCTIONS
# =========================

def plot_signature_slices(radii_m, unique_sigs, mean_slices, out_path):
    """Two-panel plot using Render class."""
    from render import Render
    
    renderer = Render()
    renderer._radii_m = radii_m
    renderer.render_signature_slices(unique_sigs, mean_slices, out_path)


def plot_obstacle_grid(angles_deg, radii_m, obstacle_grid, out_path):
    """Obstacle types in polar space using Render class."""
    from render import Render
    
    renderer = Render()
    renderer.set_polar_grid(angles_deg, radii_m, obstacle_grid)
    renderer.render_obstacle_polar(out_path)


def plot_heatmap_with_boundaries(angles_deg, radii_m, heatmap,
                                 obstacle_grid, out_path):
    """RF heatmap in polar coordinates using Render class."""
    from render import Render
    
    renderer = Render()
    renderer.set_polar_grid(angles_deg, radii_m, obstacle_grid)
    renderer.set_heatmap(heatmap)
    renderer.render_heatmap_polar(out_path)


def plot_xy_obstacle_map(polygons, out_path="xy_obstacle_map.png"):
    """XY-plane obstacle map using Render class."""
    from render import Render
    
    renderer = Render()
    renderer.polygons = polygons
    renderer.render_xy_obstacles(out_path)


def plot_xy_obstacle_boundaries_with_rf(
    polygons, points, center_lat, center_lon,
    out_path="xy_obstacle_boundaries_rf.png"
):
    """XY-plane boundaries with RF overlay using Render class."""
    from render import Render
    
    # Convert points to local XY
    local_points = []
    for lat, lon, v in points:
        x, y = latlon_to_local_xy(lat, lon, center_lat, center_lon)
        local_points.append((x, y, v))
    
    renderer = Render()
    renderer.polygons = polygons
    renderer.local_rf_points = local_points
    renderer.render_xy_boundaries_rf(out_path)


def plot_xy_obstacles_with_rf(
    polygons, points, center_lat, center_lon,
    out_path="xy_obstacles_rf.png"
):
    """XY-plane obstacles with RF overlay using Render class."""
    from render import Render
    
    # Convert points to local XY
    local_points = []
    for lat, lon, v in points:
        x, y = latlon_to_local_xy(lat, lon, center_lat, center_lon)
        local_points.append((x, y, v))
    
    renderer = Render()
    renderer.polygons = polygons
    renderer.local_rf_points = local_points
    renderer.render_xy_obstacles_rf(out_path)


def plot_sampling_and_signature_coverage_data_map(
    polygons,
    samples,
    unique_sigs,
    mean_slices,
    radii_m,
    knowledge,
    out_path="sampling_and_signature_coverage.png",
    max_walk_radius_m=None,
    include_new_signatures=True,
    all_centres=None,
):
    """
    Complex plot showing sampling locations and coverage analysis.
    
    Left: XY obstacle map with sampling locations
    Right: 2x2 grid showing coverage for top 4 samples
    """
    from render_utils import compute_xy_extent
    
    sample_colors = [
        "#e41a1c",  # red
        "#377eb8",  # blue
        "#4daf4a",  # green
        "#ff7f00",  # orange
        "#984ea3",  # purple
    ]
    n_samples = min(len(samples), len(sample_colors))
    n_show = min(n_samples, 4)

    num_sigs_base, n_r = mean_slices.shape

    # radius -> x extent
    if n_r > 1:
        dr_plot = float(radii_m[1] - radii_m[0])
    else:
        dr_plot = 1.0
    r_max_full = radii_m[-1] + dr_plot / 2.0

    # Build extended signature list
    base_sig_set = set(unique_sigs)
    sigs_from_samples = set()
    for s in samples[:n_samples]:
        sigs_from_samples |= set(s["coverage"].keys())

    if include_new_signatures:
        all_sigs = base_sig_set | sigs_from_samples
    else:
        all_sigs = base_sig_set | (sigs_from_samples & set(knowledge.keys()))

    extended_sigs = sorted(all_sigs)
    num_sigs_ext = len(extended_sigs)

    sig_to_row_ext = {s: i for i, s in enumerate(extended_sigs)}
    base_sig_to_row = {s: i for i, s in enumerate(unique_sigs)}

    print("========== [DEBUG] Signature / knowledge summary ==========")
    print(f"[DEBUG] unique_sigs: {len(unique_sigs)}")
    print(f"[DEBUG] knowledge entries: {len(knowledge)}")
    print(f"[DEBUG] coverage sigs from samples: {len(sigs_from_samples)}")
    print(f"[DEBUG] extended_sigs: {num_sigs_ext}")
    if extended_sigs:
        print("[DEBUG] first 5 extended_sigs:")
        for s in extended_sigs[:5]:
            print("   ", s[:50] + ("..." if len(s) > 50 else ""))
    print("===========================================================\n")

    # Build extended data map and knowledge map
    extended_data = np.full((num_sigs_ext, n_r), np.nan, dtype=float)
    knowledge_mat = np.zeros((num_sigs_ext, n_r), dtype=bool)

    for s, row_ext in sig_to_row_ext.items():
        if s in base_sig_to_row:
            row_base = base_sig_to_row[s]
            extended_data[row_ext, :] = mean_slices[row_base, :]
        if s in knowledge:
            knowledge_mat[row_ext, :] = knowledge[s]

    # Figure layout
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(
        2, 3, width_ratios=[1.3, 1.0, 1.0], height_ratios=[1.0, 1.0], figure=fig
    )

    ax_xy = fig.add_subplot(gs[:, 0])
    ax_s0 = fig.add_subplot(gs[0, 1])
    ax_s1 = fig.add_subplot(gs[0, 2])
    ax_s2 = fig.add_subplot(gs[1, 1])
    ax_s3 = fig.add_subplot(gs[1, 2])
    sample_axes = [ax_s0, ax_s1, ax_s2, ax_s3]

    def is_base_sample(sample):
        return abs(sample.get("score", 0.0)) < 1e-9

    # LEFT: XY obstacle + centres + samples using render_utils
    x_min, x_max, y_min, y_max = compute_xy_extent(polygons)
    
    # Extend for samples and centres
    xs_all = [x_min, x_max]
    ys_all = [y_min, y_max]
    
    if all_centres is not None:
        xs_all.extend([c["x"] for c in all_centres])
        ys_all.extend([c["y"] for c in all_centres])
    xs_all.extend([s["x"] for s in samples[:n_samples]])
    ys_all.extend([s["y"] for s in samples[:n_samples]])
    
    if xs_all and ys_all:
        x_min, x_max = min(xs_all), max(xs_all)
        y_min, y_max = min(ys_all), max(ys_all)
        dx = x_max - x_min
        dy = y_max - y_min
        mx = dx * 0.05 if dx > 0 else 1.0
        my = dy * 0.05 if dy > 0 else 1.0
        x_min -= mx
        x_max += mx
        y_min -= my
        y_max += my

    # Draw obstacle polygons
    for poly, cat in polygons:
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS.get("unknown", "#cccccc"))
        x_poly, y_poly = poly.exterior.xy
        ax_xy.fill(
            x_poly,
            y_poly,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.7,
        )

    # All evaluated centres as faint x
    if all_centres is not None and len(all_centres) > 0:
        xs_c = [c["x"] for c in all_centres]
        ys_c = [c["y"] for c in all_centres]
        ax_xy.scatter(
            xs_c,
            ys_c,
            marker="x",
            color="black",
            s=15,
            alpha=0.3,
            zorder=2,
        )

    # Chosen samples as coloured circles + rings
    for idx in range(n_samples):
        s = samples[idx]
        ax_xy.scatter(
            s["x"],
            s["y"],
            color=sample_colors[idx],
            s=60,
            marker="o",
            edgecolor="black",
            linewidths=1.0,
            zorder=5,
        )
        ax_xy.text(
            s["x"],
            s["y"],
            f"{idx+1}",
            color="black",
            fontsize=5,
            ha="center",
            va="center",
            zorder=6,
        )
        if max_walk_radius_m is not None:
            ring = Circle(
                (s["x"], s["y"]),
                radius=max_walk_radius_m,
                edgecolor=sample_colors[idx],
                facecolor="none",
                linewidth=1.0,
                alpha=0.8,
                zorder=4,
            )
            ax_xy.add_patch(ring)

    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlim(x_min, x_max)
    ax_xy.set_ylim(y_min, y_max)
    ax_xy.set_xlabel("X (m, local)")
    ax_xy.set_ylabel("Y (m, local)")
    ax_xy.set_title("Sampling locations (all centres 'x', chosen dots + rings)")

    # Legend
    patches = []
    for lab in ["open", "trees", "building", "lake", "unknown"]:
        if lab in CATEGORY_COLORS:
            patches.append(mpatches.Patch(color=CATEGORY_COLORS[lab], label=lab))
    for idx in range(n_samples):
        patches.append(
            mpatches.Patch(color=sample_colors[idx], label=f"sample {idx+1}")
        )
    if patches:
        ax_xy.legend(handles=patches, loc="upper right",
                     fontsize="small", frameon=True)

    # RIGHT: per-sample coverage maps
    for ax in sample_axes:
        ax.imshow(
            extended_data,
            origin="lower",
            aspect="auto",
            extent=[0, r_max_full, 0, num_sigs_ext],
            interpolation="nearest",
        )

    combined_new = np.zeros((num_sigs_ext, n_r), dtype=bool)

    for s_idx in range(n_show):
        s = samples[s_idx]
        ax = sample_axes[s_idx]
        cov = s["coverage"]

        print(f"\n---------- [DEBUG] Sample {s_idx + 1} ----------")
        print(f"[DEBUG] sample idx={s_idx}, score={s.get('score', 0.0)}")
        print(f"[DEBUG] coverage has {len(cov)} signatures")

        show_cells = np.zeros((num_sigs_ext, n_r), dtype=bool)
        existing_gain = 0
        new_gain = 0

        if is_base_sample(s):
            show_cells = knowledge_mat.copy()
            existing_gain = int(show_cells.sum())
            print(
                f"[COVERAGE] sample {s_idx+1} (base): "
                f"existing_cells={existing_gain}, new_cells=0"
            )
        else:
            for sig, mask in cov.items():
                if sig not in sig_to_row_ext:
                    continue
                row = sig_to_row_ext[sig]

                show_cells[row, :] |= mask

                known_row = knowledge_mat[row, :]
                new_cells = mask & ~known_row
                combined_new[row, :] |= new_cells

                gain_here = int(new_cells.sum())
                if sig in base_sig_set:
                    existing_gain += gain_here
                else:
                    new_gain += gain_here

            print(
                f"[COVERAGE] sample {s_idx+1}: "
                f"existing_gain={existing_gain}, new_gain={new_gain}, "
                f"total_covered_cells={int(show_cells.sum())}"
            )

        # Overlay coverage
        mask_display = np.ma.masked_where(~show_cells, show_cells)
        rgba = to_rgba(sample_colors[s_idx])
        hl_cmap = ListedColormap([(0, 0, 0, 0), rgba])

        ax.imshow(
            mask_display,
            origin="lower",
            aspect="auto",
            extent=[0, r_max_full, 0, num_sigs_ext],
            cmap=hl_cmap,
            interpolation="nearest",
            alpha=0.9,
        )

        if is_base_sample(s):
            ax.set_title("Sample 1 (base)\nexisting coverage")
        else:
            ax.set_title(
                f"Sample {s_idx+1}\n(gain_existing={existing_gain}, gain_new={new_gain})"
            )
        ax.set_xlabel("Distance (m)")
        ax.set_yticks([])

    # Hide unused sample axes
    for idx in range(n_show, 4):
        sample_axes[idx].axis("off")

    # Crop x-axis
    used_cols = np.any(knowledge_mat | combined_new, axis=0)
    if used_cols.any():
        last_col = np.max(np.where(used_cols)[0])
        r_max_plot = (last_col + 1) * dr_plot
        for ax in sample_axes:
            ax.set_xlim(0, r_max_plot)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved {out_path}")
