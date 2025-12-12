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

from utils import *
from config import *

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import to_rgba, ListedColormap
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

def select_top_sampling_locations(
    polygons,
    angles_deg,
    radii_m,
    knowledge,              # dict sig -> bool[n_r], from compute_signature_knowledge_from_slices
    top_k=5,
    min_spacing_m=50.0,
    candidate_grid_step_m=200.0,   # coarse step over site
    fine_grid_step_m=None,         # fine step (defaults to half of coarse)
    max_walk_radius_m=None,
    include_existing_base=False,
    existing_weight=3.0,           # prefer filling existing lines
    new_weight=1.0,                # weight for new lines
    coarse_refine_factor=4,        # how many coarse best to refine: factor * top_k
    max_hamming_frac=0.1,          # <=10% difference still counts as "same" signature
):
    """
    Multi-resolution & weighted selection of sampling locations.

    Returns:
        samples: list of dicts (chosen best centres, possibly with base prepended)
        all_centres: list of dicts {x, y, score} for *all* evaluated centres
                     (coarse + fine), for plotting as 'x' markers.
    """
    # -----------------------
    # Radii / bins
    # -----------------------
    n_r = len(radii_m)
    if n_r > 1:
        dr = float(radii_m[1] - radii_m[0])
    else:
        dr = 1.0

    if max_walk_radius_m is not None:
        max_bins_global = min(n_r - 1, int(max_walk_radius_m / dr))
    else:
        max_bins_global = n_r - 1

    if fine_grid_step_m is None:
        fine_grid_step_m = candidate_grid_step_m / 2.0

    # -----------------------
    # Site bounding box
    # -----------------------
    xs_bounds, ys_bounds = [], []
    for poly, _cat in polygons:
        bx_min, by_min, bx_max, by_max = poly.bounds
        xs_bounds.extend([bx_min, bx_max])
        ys_bounds.extend([by_min, by_max])

    if not xs_bounds or not ys_bounds:
        raise RuntimeError("No polygon bounds found for generating candidate centres.")

    x_min, x_max = min(xs_bounds), max(xs_bounds)
    y_min, y_max = min(ys_bounds), max(ys_bounds)

    # -----------------------
    # Signature matching helpers
    # -----------------------
    known_sigs = list(knowledge.keys())
    sig_match_cache = {}

    def best_matching_existing_signature(sig_raw):
        """
        Map sig_raw to a canonical signature:
          - if exact match exists in knowledge, use it
          - else, find best existing sig by Hamming distance; if
            distance <= max_hamming_frac * len(sig), snap to that
            existing sig;
          - otherwise treat as genuinely new signature (sig_raw itself).
        """
        if sig_raw in sig_match_cache:
            return sig_match_cache[sig_raw]

        if sig_raw in knowledge:
            sig_match_cache[sig_raw] = sig_raw
            return sig_raw

        if not known_sigs:
            sig_match_cache[sig_raw] = sig_raw
            return sig_raw

        L = len(sig_raw)
        max_d = int(max_hamming_frac * L)

        best_sig = None
        best_dist = L + 1

        for ks in known_sigs:
            if len(ks) != L:
                continue
            d = sum(ch1 != ch2 for ch1, ch2 in zip(sig_raw, ks))
            if d < best_dist:
                best_dist = d
                best_sig = ks
                if best_dist == 0:
                    break

        if best_sig is not None and best_dist <= max_d:
            canonical = best_sig
        else:
            canonical = sig_raw

        sig_match_cache[sig_raw] = canonical
        return canonical

    # -----------------------
    # Helper to score a single XY centre
    # -----------------------
    def score_candidate_center(x_c, y_c):
        """
        Build polar obstacle grid around (x_c, y_c), compute signature coverage
        and the resulting scores + coverage.
        """
        obs_grid_c = build_polar_obstacle_grid_for_center(
            polygons, x_c, y_c, angles_deg, radii_m
        )

        cov = defaultdict(lambda: np.zeros(n_r, dtype=bool))
        n_theta = obs_grid_c.shape[0]

        for ith in range(n_theta):
            cats_row = obs_grid_c[ith, :]
            raw_sig = "".join(
                OBSTACLE_CODES.get(c, OBSTACLE_CODES["unknown"]) for c in cats_row
            )
            canon_sig = best_matching_existing_signature(raw_sig)

            mask = np.zeros(n_r, dtype=bool)
            mask[: max_bins_global + 1] = True
            cov[canon_sig] |= mask

        score_existing = 0
        score_new = 0

        for sig, mask in cov.items():
            if sig in knowledge:   # mapped to an existing line
                known = knowledge[sig]
                new_cells = mask & ~known
                score_existing += int(new_cells.sum())
            else:
                # genuinely new signature line
                score_new += int(mask.sum())

        score_weighted = existing_weight * score_existing + new_weight * score_new

        return {
            "x": x_c,
            "y": y_c,
            "score_existing": score_existing,
            "score_new": score_new,
            "score": float(score_weighted),
            "coverage": dict(cov),
        }

    all_centres = []  # for plotting every evaluated centre

    # -----------------------
    # 1) Coarse grid search
    # -----------------------
    xs_coarse = np.arange(x_min, x_max + candidate_grid_step_m * 0.5,
                          candidate_grid_step_m)
    ys_coarse = np.arange(y_min, y_max + candidate_grid_step_m * 0.5,
                          candidate_grid_step_m)

    coarse_candidates = []
    for x_c in xs_coarse:
        for y_c in ys_coarse:
            cand = score_candidate_center(x_c, y_c)
            print(
                f"[COARSE CAND] centre=({x_c:.1f},{y_c:.1f}) "
                f"score={cand['score']:.0f} "
                f"(existing={cand['score_existing']}, new={cand['score_new']})"
            )
            all_centres.append({"x": x_c, "y": y_c, "score": cand["score"]})
            if cand["score"] <= 0:
                continue
            coarse_candidates.append(cand)

    if not coarse_candidates:
        print("[INFO] No beneficial coarse candidates found.")
        samples = []
        if include_existing_base:
            base_cov = {sig: knowledge[sig].copy() for sig in knowledge}
            samples.insert(0, {
                "x": 0.0,
                "y": 0.0,
                "score": 0.0,
                "score_existing": 0.0,
                "score_new": 0.0,
                "coverage": base_cov,
            })
        return samples, all_centres

    coarse_candidates.sort(key=lambda d: d["score"], reverse=True)

    n_refine = min(len(coarse_candidates), coarse_refine_factor * top_k)
    coarse_to_refine = coarse_candidates[:n_refine]
    print(f"[INFO] Coarse grid produced {len(coarse_candidates)} candidates, "
          f"refining top {n_refine}.")

    # -----------------------
    # 2) Fine grid refinement
    # -----------------------
    fine_candidates = []
    seen_xy = set()

    for cc in coarse_to_refine:
        x_c, y_c = cc["x"], cc["y"]

        x_start = x_c - candidate_grid_step_m
        x_end   = x_c + candidate_grid_step_m
        y_start = y_c - candidate_grid_step_m
        y_end   = y_c + candidate_grid_step_m

        xs_fine = np.arange(x_start, x_end + fine_grid_step_m * 0.5, fine_grid_step_m)
        ys_fine = np.arange(y_start, y_end + fine_grid_step_m * 0.5, fine_grid_step_m)

        for xf in xs_fine:
            if xf < x_min or xf > x_max:
                continue
            for yf in ys_fine:
                if yf < y_min or yf > y_max:
                    continue

                key = (round(xf, 3), round(yf, 3))
                if key in seen_xy:
                    continue
                seen_xy.add(key)

                cand = score_candidate_center(xf, yf)
                print(
                    f"[FINE CAND] centre=({xf:.1f},{yf:.1f}) "
                    f"score={cand['score']:.0f} "
                    f"(existing={cand['score_existing']}, new={cand['score_new']})"
                )
                all_centres.append({"x": xf, "y": yf, "score": cand["score"]})
                if cand["score"] <= 0:
                    continue
                fine_candidates.append(cand)

    final_candidates = fine_candidates if fine_candidates else coarse_candidates
    if not fine_candidates:
        print("[INFO] No beneficial fine candidates found; falling back to coarse.")

    # -----------------------
    # 3) Greedy selection with min spacing
    # -----------------------
    final_candidates.sort(key=lambda d: d["score"], reverse=True)

    chosen = []
    chosen_xy = []
    min_spacing_sq = min_spacing_m ** 2

    for cand in final_candidates:
        if len(chosen) >= top_k:
            break

        x_c, y_c = cand["x"], cand["y"]
        ok = True
        for (x0, y0) in chosen_xy:
            dx = x_c - x0
            dy = y_c - y0
            if dx * dx + dy * dy < min_spacing_sq:
                ok = False
                break

        if not ok:
            continue

        chosen.append(cand)
        chosen_xy.append((x_c, y_c))

    print(
        f"[INFO] Selected {len(chosen)} sampling locations "
        f"(requested {top_k}, min spacing {min_spacing_m} m)."
    )
    for k, c in enumerate(chosen):
        print(
            f"  #{k:02d}: x={c['x']:.1f}, y={c['y']:.1f}, "
            f"score={c['score']:.0f}, "
            f"existing={c['score_existing']}, new={c['score_new']}"
        )

    samples = chosen
    if include_existing_base:
        base_cov = {sig: knowledge[sig].copy() for sig in knowledge}
        base_sample = {
            "x": 0.0,    # adjust to your actual base XY if needed
            "y": 0.0,
            "score": 0.0,
            "score_existing": 0.0,
            "score_new": 0.0,
            "coverage": base_cov,
        }
        samples = [base_sample] + samples

    return samples, all_centres

from matplotlib.colors import ListedColormap, to_rgba
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt


def main():
    path = os.getenv('DATA_PATH')

    # Your fixed file names
    csv_path = os.path.join(path, "all.csv")
    yaml_path = os.path.join(path, "points.yaml")
    kml_path = os.path.join(path, "feature_map.kml")
    center = [os.getenv('CENTER_LAT'), os.getenv('CENTER_LON')]

    # 0) Construct yaml file
    print(f"[CVS] Updated yaml to match data in {csv_path}.")
    generate_points_yaml(csv_path, yaml_path, center)

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


    # Build knowledge directly from what the signature-slice map shows
    knowledge = compute_signature_knowledge_from_slices(unique_sigs, mean_slices)

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
        all_centres=all_centres,   # <- NEW
    )






if __name__ == "__main__":
    main()
