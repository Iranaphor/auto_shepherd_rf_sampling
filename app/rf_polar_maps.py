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


def plot_sampling_and_signature_coverage_data_map(
    polygons,
    samples,          # list from select_top_sampling_locations
    unique_sigs,      # existing data signatures
    mean_slices,      # existing data map (num_sigs x n_r)
    radii_m,
    knowledge,        # dict[sig] -> bool[n_r]
    out_path="sampling_and_signature_coverage.png",
    max_walk_radius_m=None,
    include_new_signatures=True,
    all_centres=None, # list of {x,y,score} for all evaluated centres
):
    """
    Left: XY obstacle map
      - All evaluated centres (all_centres) as faint black 'x'
      - Chosen samples as coloured circles + rings

    Right: 2x2 grid (top 4 samples):
      - Background: coloured extended data map (mean_slices, extended)
      - Sample 1 (if base, score ~ 0): show existing coverage (knowledge_mat)
      - Other samples: show **full coverage** (where coverage[sig] is True)
        in that sample's colour, on any rows (existing or new).
      - Console prints info gain per sample (existing vs new rows), based on
        new_cells = coverage & ~knowledge_mat, plus extra debug info.
    """
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

    # ------------------------------------------------------------------
    # Build extended signature list so that **all coverage sigs** exist
    # ------------------------------------------------------------------
    base_sig_set = set(unique_sigs)

    # Always include all coverage keys from all samples, regardless of flag
    sigs_from_samples = set()
    for s in samples[:n_samples]:
        sigs_from_samples |= set(s["coverage"].keys())

    if include_new_signatures:
        all_sigs = base_sig_set | sigs_from_samples
    else:
        # If flag is False, we still need coverage keys present or nothing shows.
        # So restrict to existing + any sigs that mapped to existing knowledge.
        all_sigs = base_sig_set | (sigs_from_samples & set(knowledge.keys()))

    extended_sigs = sorted(all_sigs)
    num_sigs_ext = len(extended_sigs)

    sig_to_row_ext = {s: i for i, s in enumerate(extended_sigs)}
    base_sig_to_row = {s: i for i, s in enumerate(unique_sigs)}

    # --- DEBUG: high-level signature info ---
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

    # Build extended data map and extended knowledge map
    extended_data = np.full((num_sigs_ext, n_r), np.nan, dtype=float)
    knowledge_mat = np.zeros((num_sigs_ext, n_r), dtype=bool)

    for s, row_ext in sig_to_row_ext.items():
        if s in base_sig_to_row:
            row_base = base_sig_to_row[s]
            extended_data[row_ext, :] = mean_slices[row_base, :]
        if s in knowledge:
            knowledge_mat[row_ext, :] = knowledge[s]
        else:
            knowledge_mat[row_ext, :] = False

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
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
        # treat the prepended base sample as score ~ 0.0
        return abs(sample.get("score", 0.0)) < 1e-9

    # ------------------------------------------------------------------
    # LEFT: XY obstacle + centres + samples
    # ------------------------------------------------------------------
    xs_all, ys_all = [], []
    for poly, _cat in polygons:
        bx_min, by_min, bx_max, by_max = poly.bounds
        xs_all.extend([bx_min, bx_max])
        ys_all.extend([by_min, by_max])

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
    else:
        x_min, x_max, y_min, y_max = -10, 10, -10, 10

    # obstacle polygons
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

    # all evaluated centres as faint x
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

    # chosen samples as coloured circles + rings
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

    # ------------------------------------------------------------------
    # RIGHT: per-sample coverage maps
    # ------------------------------------------------------------------
    # Base coloured data background on each axis
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
        cov = s["coverage"]  # dict sig -> bool[n_r]

        print("\n---------- [DEBUG] Sample", s_idx + 1, "----------")
        print(f"[DEBUG] sample idx={s_idx}, score={s.get('score', 0.0)}")
        print(f"[DEBUG] coverage has {len(cov)} signatures")

        # quick coverage summary
        total_cov_cells = 0
        known_cov_sigs = 0
        new_cov_sigs = 0
        for sig, mask in cov.items():
            if sig in knowledge:
                known_cov_sigs += 1
            else:
                new_cov_sigs += 1
            total_cov_cells += int(mask.sum())

        print(f"[DEBUG] coverage sigs: known={known_cov_sigs}, new_like={new_cov_sigs}")
        print(f"[DEBUG] total coverage cells (raw mask sum)={total_cov_cells}")

        # sample first few sigs for mapping
        print("[DEBUG] first up to 5 coverage signatures & row mapping:")
        for i, sig in enumerate(list(cov.keys())[:5]):
            row = sig_to_row_ext.get(sig, None)
            print("   ", i, "row=", row, "sig=", sig[:50] + ("..." if len(sig) > 50 else ""))

        show_cells = np.zeros((num_sigs_ext, n_r), dtype=bool)
        existing_gain = 0
        new_gain = 0

        if is_base_sample(s):
            # base sample: show existing coverage
            show_cells = knowledge_mat.copy()
            existing_gain = int(show_cells.sum())
            new_gain = 0
            print(
                f"[COVERAGE] sample {s_idx+1} (base): "
                f"existing_cells={existing_gain}, new_cells={new_gain}"
            )
        else:
            # other samples: show **full coverage**, compute info gain vs knowledge
            for sig, mask in cov.items():
                if sig not in sig_to_row_ext:
                    print(f"[WARN] coverage signature not in extended_sigs (skipped): {sig[:60]}...")
                    continue
                row = sig_to_row_ext[sig]

                # visual: full coverage
                show_cells[row, :] |= mask

                # info-gain: which of those are newly uncovered
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

        total_show = int(show_cells.sum())
        if total_show == 0:
            print(f"[DEBUG] sample {s_idx+1}: show_cells has 0 True entries (no overlay).")

        # Overlay full coverage for this sample
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

    # Hide unused sample axes if < 4
    for idx in range(n_show, 4):
        sample_axes[idx].axis("off")

    # Crop x-axis to region where there is either existing or new coverage
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
