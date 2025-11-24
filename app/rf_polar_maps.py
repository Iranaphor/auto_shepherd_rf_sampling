#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# Generate a polar obstacle map and RF heatmap from YAML (points) and
# KML (obstacle polygons).
#
# @author      James R. Heselden (github: iranaphor)
# @maintainer  James R. Heselden (github: iranaphor)
# @datecreated 21st November 2025
# @credits     Code structure and implementation were developed by the
#              author with assistance from OpenAI's GPT-5.1 model, used
#              under the author's direction and supervision.
#
# @inputs
#   points.yaml
#   feature_map.kml
#
# @outputs
#   obstacles_polar.png
#   points_heatmap_polar.png
#   signature_slices_polar.png
# ###########################################################################

import os
import csv
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt

from fastkml import kml
from shapely.geometry import Polygon, MultiPolygon, Point
from collections import Counter

# =========================
# CONFIGURATION
# =========================

# Angular bin size in degrees (y-axis resolution)
DTHETA_DEG = 5.0   # e.g. 5° -> 72 rows

# Radial bin size in meters (x-axis resolution)
DR_M = 10.0        # e.g. 10 m bins

# Max range (if None, computed from polygons + points)
MAX_RANGE_M = None

# Obstacle category colors (for plotting)
CATEGORY_COLORS = {
    "open":     "#ffffff",  # white
    "trees":    "#00aa00",  # green
    "building": "#ff9900",  # orange
    "lake":     "#0066ff",  # blue
    "unknown":  "#cccccc"   # light grey background
}
# Map obstacle categories to numeric codes (as characters)
OBSTACLE_CODES = {
    "open": "0",
    "trees": "1",
    "building": "2",
    "lake": "3",
    "unknown": "4",  # fallback
}

# =========================
# HELPER FUNCTIONS
# =========================

def latlon_to_local_xy(lat, lon, lat0, lon0):
    """
    Approximate conversion from lat/lon (degrees) to local XY in meters
    relative to (lat0, lon0) using a simple equirectangular approximation.
    """
    R = 6378137.0  # meters

    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)

    x = R * dlon * math.cos(math.radians(lat0))
    y = R * dlat
    return x, y


def load_yaml_points(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    center_lat, center_lon = data["center"]
    points = data["gps_coords_xyv"]  # list of [lat, lon, value]

    return center_lat, center_lon, points


import xml.etree.ElementTree as ET
from collections import Counter
from shapely.geometry import Polygon

def classify_category(name_str: str) -> str:
    """
    Map a name string to one of our categories based on simple rules.
    Adjust this if your KML uses different words.
    """
    s = (name_str or "").strip().lower()

    # exact matches
    if s in ("open", "open field", "open area", "field", "fields"):
        return "open"
    if s in ("tree", "trees", "wood", "woods", "forest"):
        return "trees"
    if s in ("building", "buildings", "barn", "shed", "house", "houses"):
        return "building"
    if s in ("lake", "lakes", "pond", "ponds", "water", "waterbody", "water body"):
        return "lake"

    # substring fallbacks
    if "tree" in s:
        return "trees"
    if "build" in s or "barn" in s or "shed" in s or "house" in s:
        return "building"
    if "lake" in s or "pond" in s or "water" in s:
        return "lake"
    if "open" in s or "field" in s or "pasture" in s:
        return "open"

    return "unknown"


def load_kml_polygons(kml_path, center_lat, center_lon):
    """
    Load polygons from a Google Earth KML using plain XML:

    - Find all <Placemark>
    - Read <name> for category
    - Read <Polygon>/<outerBoundaryIs>/<LinearRing>/<coordinates>
    - Convert lon,lat to local XY, build shapely Polygons

    Prints one debug line per polygon with name, type, colour, and bounds.
    """

    print(f"[KML] Parsing KML with ElementTree: {kml_path}")
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # Root tag is usually {namespace}kml. We'll just wildcard the namespace: { * }
    ns_any = "{*}"

    polygons = []

    # Loop over all Placemark elements in the document
    for pm in root.findall(".//{*}Placemark"):
        name_el = pm.find(f"{ns_any}name")
        name = (name_el.text.strip() if name_el is not None and name_el.text else "")
        category = classify_category(name)
        color = CATEGORY_COLORS.get(category, "#cccccc")

        # Each Placemark may have one or more Polygon elements
        poly_elems = pm.findall(".//{*}Polygon")
        if not poly_elems:
            # This placemark might be a Point, LineString, etc. – skip it
            continue

        for poly_el in poly_elems:
            coords_el = poly_el.find(".//{*}outerBoundaryIs/{*}LinearRing/{*}coordinates")
            if coords_el is None or not coords_el.text or not coords_el.text.strip():
                print(f"[KML]   Placemark '{name}' has Polygon but no coordinates – skipping")
                continue

            coord_strings = coords_el.text.strip().split()
            xy_coords = []
            for cs in coord_strings:
                parts = cs.split(",")
                if len(parts) < 2:
                    continue
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                except ValueError:
                    continue
                x, y = latlon_to_local_xy(lat, lon, center_lat, center_lon)
                xy_coords.append((x, y))

            if len(xy_coords) < 3:
                print(f"[KML]   Placemark '{name}' polygon has <3 valid coords – skipping")
                continue

            poly = Polygon(xy_coords)
            idx = len(polygons)
            polygons.append((poly, category))

            print(
                f"[KML]   Polygon {idx}: "
                f"name='{name}', "
                f"category='{category}', "
                f"color='{color}', "
                f"bounds={poly.bounds}"
            )

    counts = Counter(cat for _, cat in polygons)
    print(f"[KML] Loaded {len(polygons)} polygons by category: {dict(counts)}")

    return polygons

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


def compute_obstacle_signatures_and_slices(obstacle_grid, heatmap):
    """
    For each angle row (ray) in obstacle_grid, build a signature string
    like '00001110101233' using OBSTACLE_CODES, then group rays with the
    same signature and average their heatmap values over angle.

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

from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_signature_slices(radii_m, unique_sigs, mean_slices, out_path):
    """
    Two-panel plot:
      - Left: obstacle patterns per signature (categorical colours)
      - Right: averaged RF value per signature (heatmap)

    Both share the same y-axis (signature index), so rows are aligned.
    """
    num_sigs, n_r = mean_slices.shape

    if len(radii_m) > 1:
        dr = radii_m[1] - radii_m[0]
    else:
        dr = DR_M
    r_max = radii_m[-1] + dr / 2.0

    # ------------------------------------------------------------------
    # Build obstacle grid from signature strings
    # Each signature is like "00001110101233" (chars are codes 0..4)
    # ------------------------------------------------------------------
    sig_obs_grid = np.full((num_sigs, n_r), np.nan, dtype=float)

    for i, sig in enumerate(unique_sigs):
        # Truncate or pad to the number of radius bins
        L = min(len(sig), n_r)
        for j in range(L):
            c = sig[j]
            if c.isdigit():
                sig_obs_grid[i, j] = int(c)

    # Categorical colormap for obstacles (0=open,1=trees,2=building,3=lake,4=unknown)
    obs_colors = [
        CATEGORY_COLORS["open"],
        CATEGORY_COLORS["trees"],
        CATEGORY_COLORS["building"],
        CATEGORY_COLORS["lake"],
        CATEGORY_COLORS["unknown"],
    ]
    obs_cmap = ListedColormap(obs_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, obs_cmap.N)

    # ------------------------------------------------------------------
    # Create side-by-side subplots
    # ------------------------------------------------------------------
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
        norm=norm,
        interpolation="nearest"
    )
    ax_obs.set_xlabel("Distance from base (m)")
    ax_obs.set_ylabel("Obstacle signature (sorted index)")
    ax_obs.set_title("Obstacle pattern per signature")

    # Build a small legend for obstacle colours
    import matplotlib.patches as mpatches
    labels = ["open", "trees", "building", "lake", "unknown"]
    patches = [
        mpatches.Patch(color=CATEGORY_COLORS[l], label=l)
        for l in labels
    ]
    ax_obs.legend(
        handles=patches,
        loc="upper right",
        fontsize="small",
        frameon=True
    )

    # Right: mean RF values
    im_val = ax_val.imshow(
        mean_slices,
        origin="lower",
        aspect="auto",
        extent=[0, r_max, 0, num_sigs],
        interpolation="nearest"
    )
    cbar = fig.colorbar(im_val, ax=ax_val)
    cbar.set_label("Mean value (v)")

    ax_val.set_xlabel("Distance from base (m)")
    ax_val.set_title("Mean RF per signature")

    # Y tick labels: show signatures if there aren't too many
    max_labels = 20
    if num_sigs <= max_labels:
        yticks = np.arange(num_sigs) + 0.5
        # Truncate long signature strings for readability
        labels_sig = []
        for s in unique_sigs:
            if len(s) <= 20:
                labels_sig.append(s)
            else:
                labels_sig.append(s[:8] + "…" + s[-4:])
        ax_obs.set_yticks(yticks)
        ax_obs.set_yticklabels(labels_sig)
        ax_val.set_yticks(yticks)
        ax_val.set_yticklabels([])  # no duplicate labels on right
    else:
        yticks = np.arange(num_sigs) + 0.5
        ax_obs.set_yticks(yticks)
        ax_obs.set_yticklabels([str(i) for i in range(num_sigs)])
        ax_val.set_yticks(yticks)
        ax_val.set_yticklabels([])

    fig.suptitle("Obstacle signatures and averaged RF slices", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_obstacle_grid(angles_deg, radii_m, obstacle_grid, out_path):
    """
    First image: obstacle types in polar space.
    """
    # Ensure consistent order of categories
    cats = ["open", "trees", "building", "lake", "unknown"]
    present = {c for c in obstacle_grid.flatten()}
    cats = [c for c in cats if c in present]

    cat_to_idx = {c: i for i, c in enumerate(cats)}
    grid_idx = np.vectorize(cat_to_idx.get)(obstacle_grid)

    from matplotlib.colors import ListedColormap
    colors = [CATEGORY_COLORS[c] for c in cats]
    cmap = ListedColormap(colors)

    # Extent for imshow
    if len(radii_m) > 1:
        dr = radii_m[1] - radii_m[0]
    else:
        dr = DR_M
    r_max = radii_m[-1] + dr / 2.0

    theta_min, theta_max = 0.0, 360.0

    plt.figure(figsize=(10, 6))
    plt.imshow(
        grid_idx,
        origin="lower",
        aspect="auto",
        extent=[0, r_max, theta_min, theta_max],
        cmap=cmap,
        interpolation="nearest"
    )
    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(len(cats)) + 0.5)
    cbar.set_ticklabels(cats)

    plt.xlabel("Distance from base (m)")
    plt.ylabel("Angle (deg)")
    plt.title("Obstacle categories in polar space")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_heatmap_with_boundaries(angles_deg, radii_m, heatmap,
                                 obstacle_grid, out_path):
    """
    Second image: RF heatmap with black borders around regions where
    obstacle category changes (boundaries between coloured cells).
    """
    # Category -> index again
    cats = ["open", "trees", "building", "lake", "unknown"]
    present = {c for c in obstacle_grid.flatten()}
    cats = [c for c in cats if c in present]
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    grid_idx = np.vectorize(cat_to_idx.get)(obstacle_grid)

    # Build boundary mask: mark cells that have a neighbour with different category
    boundary = np.zeros_like(grid_idx, dtype=bool)
    # vertical neighbours
    boundary[:-1, :] |= (grid_idx[:-1, :] != grid_idx[1:, :])
    boundary[1:, :]  |= (grid_idx[:-1, :] != grid_idx[1:, :])
    # horizontal neighbours
    boundary[:, :-1] |= (grid_idx[:, :-1] != grid_idx[:, 1:])
    boundary[:, 1:]  |= (grid_idx[:, :-1] != grid_idx[:, 1:])

    # Extent
    if len(radii_m) > 1:
        dr = radii_m[1] - radii_m[0]
    else:
        dr = DR_M
    r_max = radii_m[-1] + dr / 2.0
    theta_min, theta_max = 0.0, 360.0

    plt.figure(figsize=(10, 6))

    # Heatmap of values
    im = plt.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=[0, r_max, theta_min, theta_max],
        interpolation="nearest"
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Mean value (v)")

    # Draw black contours along boundaries
    plt.contour(
        boundary.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=0.5,
        origin="lower",
        extent=[0, r_max, theta_min, theta_max]
    )

    plt.xlabel("Distance from base (m)")
    plt.ylabel("Angle (deg)")
    plt.title("Point heatmap in polar space with obstacle region boundaries")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")



def _points_to_local_xy(points, center_lat, center_lon):
    """Convert [lat, lon, v] points into (x, y, v) in local meters."""
    local = []
    for lat, lon, v in points:
        x, y = latlon_to_local_xy(lat, lon, center_lat, center_lon)
        local.append((x, y, v))
    return local


def _compute_xy_extent(polygons, local_points=None, margin_factor=0.05):
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
        # Fallback
        return (-10, 10, -10, 10)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add a small margin
    dx = x_max - x_min
    dy = y_max - y_min
    mx = dx * margin_factor if dx > 0 else 1.0
    my = dy * margin_factor if dy > 0 else 1.0

    return (x_min - mx, x_max + mx, y_min - my, y_max + my)


def _render_xy_scene(
    polygons,
    local_points,
    show_fill=True,
    show_edges=True,
    show_rf=False,
    title="",
    out_path="xy_plot.png"
):
    """
    Internal renderer for XY plane.

    polygons   : list of (Polygon in local XY, category_str)
    local_points : list of (x, y, v) in local XY
    show_fill  : fill polygons with CATEGORY_COLORS
    show_edges : draw polygon edges
    show_rf    : scatter RF points coloured by v
    """
    # Compute extents
    x_min, x_max, y_min, y_max = _compute_xy_extent(polygons, local_points)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw polygons
    for poly, cat in polygons:
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS.get("unknown", "#cccccc"))
        x, y = poly.exterior.xy

        if show_fill:
            ax.fill(x, y, facecolor=color, edgecolor="none", alpha=0.8)
        if show_edges:
            ax.plot(x, y, color="black", linewidth=0.7)

    # RF overlay
    if show_rf and local_points:
        xs = [p[0] for p in local_points]
        ys = [p[1] for p in local_points]
        vs = [p[2] for p in local_points]

        sc = ax.scatter(xs, ys, c=vs, s=10, alpha=0.8)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Value (v)")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X (m, local)")
    ax.set_ylabel("Y (m, local)")
    ax.set_title(title)

    # Optional legend for obstacle colours
    import matplotlib.patches as mpatches
    labels = ["open", "trees", "building", "lake", "unknown"]
    patches = []
    for lab in labels:
        if lab in CATEGORY_COLORS:
            patches.append(
                mpatches.Patch(color=CATEGORY_COLORS[lab], label=lab)
            )
    if patches:
        ax.legend(handles=patches, loc="upper right", fontsize="small", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")



def plot_xy_obstacle_map(polygons, out_path="xy_obstacle_map.png"):
    """
    XY-plane obstacle map: filled polygons in their obstacle colours.
    """
    _render_xy_scene(
        polygons=polygons,
        local_points=None,
        show_fill=True,
        show_edges=True,
        show_rf=False,
        title="Obstacle map (XY plane)",
        out_path=out_path,
    )

def plot_xy_obstacle_boundaries_with_rf(
    polygons, points, center_lat, center_lon,
    out_path="xy_obstacle_boundaries_rf.png"
):
    """
    XY-plane: polygon boundaries in black, RF points overlaid as coloured scatter.
    """
    local_points = _points_to_local_xy(points, center_lat, center_lon)

    _render_xy_scene(
        polygons=polygons,
        local_points=local_points,
        show_fill=False,   # no fills, just edges
        show_edges=True,
        show_rf=True,
        title="Obstacle boundaries + RF overlay (XY plane)",
        out_path=out_path,
    )

def plot_xy_obstacles_with_rf(
    polygons, points, center_lat, center_lon,
    out_path="xy_obstacles_rf.png"
):
    """
    XY-plane: filled obstacle polygons plus RF points overlaid as coloured scatter.
    """
    local_points = _points_to_local_xy(points, center_lat, center_lon)

    _render_xy_scene(
        polygons=polygons,
        local_points=local_points,
        show_fill=True,
        show_edges=True,
        show_rf=True,
        title="Obstacle map + RF overlay (XY plane)",
        out_path=out_path,
    )



# =========================
# MAIN
# =========================

def generate_points_yaml(csv_path, yaml_path, center):

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

def main():
    path = os.getenv('DATA_PATH')

    # Your fixed file names
    csv_path = os.path.join(path, "all.csv")
    yaml_path = os.path.join(path, "points.yaml")
    kml_path = os.path.join(path, "feature_map.kml")
    center = [os.getenv('CENTER_LAT'), os.getenv('CENTER_LON')]

    # 0) Construct yaml file
    print(f"[CVS] Updated yaml to match data in {csv_path}.")
    generate_points_yaml(csv_path, yaml_path, centre)

    # 1) Load YAML
    center_lat, center_lon, points = load_yaml_points(yaml_path)
    print(f"[YAML] Loaded {len(points)} RF points. Center: {center_lat}, {center_lon}")

    # 2) Load KML polygons (converted to local XY)
    polygons = load_kml_polygons(kml_path, center_lat, center_lon)

    # 3) Determine max range if not set
    if MAX_RANGE_M is None:
        max_range = compute_max_range(polygons, center_lat, center_lon, points)
    else:
        max_range = MAX_RANGE_M

    print(f"[INFO] Using max range: {max_range:.1f} m")

    # 4) Build polar obstacle grid
    angles_deg, radii_m, obstacle_grid = build_polar_grid(
        polygons, max_range, DTHETA_DEG, DR_M
    )

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
        center_lon,
        out_path=os.path.join(path, "xy_obstacles_rf.png")
    )


    # 5) Plot obstacle polar map
    plot_obstacle_grid(angles_deg,
        radii_m,
        obstacle_grid,
        os.path.join(path, "obstacles_polar.png")
    )

    # 6) Build points heatmap in same grid
    heatmap = build_points_heatmap(
        points, center_lat, center_lon,
        angles_deg, radii_m, DTHETA_DEG, DR_M, max_range
    )

    # 7) Plot heatmap with black boundaries around obstacle regions
    plot_heatmap_with_boundaries(
        angles_deg, radii_m, heatmap, obstacle_grid, os.path.join(path, "points_heatmap_polar.png")
    )

    # 8) Build and plot signature-averaged slices image
    unique_sigs, mean_slices, counts = compute_obstacle_signatures_and_slices(
        obstacle_grid, heatmap
    )
    plot_signature_slices(
        radii_m, unique_sigs, mean_slices, os.path.join(path, "signature_slices_polar.png")
    )


if __name__ == "__main__":
    main()
