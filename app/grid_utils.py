#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# Grid generation utilities for search route optimization
#
# AI-WRITTEN CODE DECLARATION:
# This code was developed with substantial assistance from AI tools.
#
# @author      James R. Heselden (github: Iranaphor)
# @maintainer  James R. Heselden (github: Iranaphor)
# @datecreated 15th December 2025
# ###########################################################################

import numpy as np
from kml_utils import latlon_to_local_xy
from utils import build_polar_obstacle_grid_for_center, OBSTACLE_CODES


def generate_grid_points(x_min, x_max, y_min, y_max, grid_spacing):
    """
    Generate a regular grid of points across the environment.
    
    Returns list of (x, y) tuples.
    """
    xs = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    ys = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    
    points = []
    for x in xs:
        for y in ys:
            points.append((x, y))
    
    return points


def generate_8connected_edges(grid_points, grid_spacing):
    """
    Generate 8-connected edges between grid points.
    
    Returns list of edge tuples: ((x1, y1), (x2, y2))
    """
    edges = []
    point_set = set(grid_points)
    
    # 8 directions: right, up, diag-up-right, diag-up-left
    deltas = [
        (grid_spacing, 0),           # right
        (0, grid_spacing),           # up
        (grid_spacing, grid_spacing),   # diag up-right
        (-grid_spacing, grid_spacing),  # diag up-left
    ]
    
    for x, y in grid_points:
        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if (nx, ny) in point_set:
                # Add edge in both directions (undirected graph)
                edge1 = ((x, y), (nx, ny))
                edge2 = ((nx, ny), (x, y))
                if edge1 not in edges and edge2 not in edges:
                    edges.append(edge1)
    
    return edges


def compute_grid_bounds_from_polygons(polygons, margin=10.0):
    """
    Compute bounding box from polygons with margin.
    
    Returns (x_min, x_max, y_min, y_max).
    """
    x_coords = []
    y_coords = []
    
    for poly, _cat in polygons:
        x_min, y_min, x_max, y_max = poly.bounds
        x_coords.extend([x_min, x_max])
        y_coords.extend([y_min, y_max])
    
    return (
        min(x_coords) - margin,
        max(x_coords) + margin,
        min(y_coords) - margin,
        max(y_coords) + margin
    )


def compute_obstacle_signature_for_point(x, y, base_x, base_y, polygons, 
                                        angles_deg, radii_m):
    """
    Compute obstacle signature from a grid point to base location.
    
    Returns signature string (e.g., "001112223") where each character
    represents the obstacle category at each radial bin.
    """
    # Build polar grid centered at this point
    obstacle_grid = build_polar_obstacle_grid_for_center(
        polygons, x, y, angles_deg, radii_m
    )
    
    # Compute angle from point to base
    dx = base_x - x
    dy = base_y - y
    angle_to_base = np.degrees(np.arctan2(dy, dx))
    if angle_to_base < 0:
        angle_to_base += 360
    
    # Find closest angle bin
    angle_idx = int(np.round(angle_to_base / (angles_deg[1] - angles_deg[0]))) % len(angles_deg)
    
    # Extract signature along this ray
    cats = obstacle_grid[angle_idx, :]
    signature = "".join(OBSTACLE_CODES.get(c, OBSTACLE_CODES["unknown"]) for c in cats)
    
    return signature


def generate_subnodes_on_edge(p1, p2, num_subnodes):
    """
    Generate equally-spaced subnodes along an edge.
    
    Returns list of (x, y) positions including endpoints.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    subnodes = []
    for i in range(num_subnodes + 1):
        t = i / num_subnodes
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        subnodes.append((x, y))
    
    return subnodes


def count_unique_signatures_on_edge(edge, base_x, base_y, polygons,
                                   angles_deg, radii_m, num_subnodes=5):
    """
    Count unique obstacle signatures along an edge by sampling subnodes.
    
    Returns integer count of unique signatures.
    """
    p1, p2 = edge
    subnodes = generate_subnodes_on_edge(p1, p2, num_subnodes)
    
    signatures = set()
    for x, y in subnodes:
        sig = compute_obstacle_signature_for_point(
            x, y, base_x, base_y, polygons, angles_deg, radii_m
        )
        signatures.add(sig)
    
    return len(signatures)
