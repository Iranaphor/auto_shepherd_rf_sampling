#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# Sampling location selection and optimization utilities
#
# @author      James R. Heselden (github: Iranaphor)
# @maintainer  James R. Heselden (github: Iranaphor)
# @datecreated 12th December 2025
# @datemodified 12th December 2025
# @credits     Developed with assistance from Claude Sonnet 4.5 and GitHub
#              Copilot.
# ###########################################################################

from collections import defaultdict

import numpy as np

from config import OBSTACLE_CODES
from utils import build_polar_obstacle_grid_for_center


def select_top_sampling_locations(
    polygons,
    angles_deg,
    radii_m,
    knowledge,
    top_k=5,
    min_spacing_m=50.0,
    candidate_grid_step_m=200.0,
    fine_grid_step_m=None,
    max_walk_radius_m=None,
    include_existing_base=False,
    existing_weight=3.0,
    new_weight=1.0,
    coarse_refine_factor=4,
    max_hamming_frac=0.1,
):
    """
    Multi-resolution weighted selection of optimal sampling locations.
    Returns chosen samples and all evaluated centres for visualization.
    """
    # Radii and bin setup
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

    # Compute site bounding box
    xs_bounds, ys_bounds = [], []
    for poly, _cat in polygons:
        bx_min, by_min, bx_max, by_max = poly.bounds
        xs_bounds.extend([bx_min, bx_max])
        ys_bounds.extend([by_min, by_max])

    if not xs_bounds or not ys_bounds:
        raise RuntimeError("No polygon bounds found for generating candidate centres.")

    x_min, x_max = min(xs_bounds), max(xs_bounds)
    y_min, y_max = min(ys_bounds), max(ys_bounds)

    # Signature matching helpers
    known_sigs = list(knowledge.keys())
    sig_match_cache = {}

    def best_matching_existing_signature(sig_raw):
        """Map raw signature to canonical signature using Hamming distance."""
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
        sig_match_cache[sig_raw] = canonical
        return canonical

    def score_candidate_center(x_c, y_c):
        """Build polar grid around centre and compute coverage scores.""" cats_row = obs_grid_c[ith, :]
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
        score_weighted = existing_weight * score_existing + new_weight * score_new

        return {
            "x": x_c,
            "y": y_c,
            "score_existing": score_existing,
            "score_new": score_new,
            "score": float(score_weighted),
            "coverage": dict(cov),
        }

    all_centres = []

    # Coarse grid search
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
    print(f"[INFO] Coarse grid produced {len(coarse_candidates)} candidates, "
          f"refining top {n_refine}.")

    # Fine grid refinement
    fine_candidates = []
    seen_xy = set()e_to_refine:
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

    final_candidates = fine_candidates if fine_candidates else coarse_candidates
    if not fine_candidates:
        print("[INFO] No beneficial fine candidates found; falling back to coarse.")

    # Greedy selection with minimum spacing
    final_candidates.sort(key=lambda d: d["score"], reverse=True)

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
