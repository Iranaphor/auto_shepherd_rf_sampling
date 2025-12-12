#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# KML parsing and loading utilities
#
# @author      James R. Heselden (github: iranaphor)
# @maintainer  James R. Heselden (github: iranaphor)
# @datecreated 12th December 2025
# ###########################################################################

import math
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from config import CATEGORY_COLORS


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
    - Skip placemarks with <visibility>0</visibility>
    - Read <name> for category
    - Read <Polygon>/<outerBoundaryIs>/<LinearRing>/<coordinates>
    - Convert lon,lat to local XY, build shapely Polygons

    Prints one debug line per polygon with name, type, colour, and bounds.
    
    Returns:
        List of (shapely.Polygon in local XY, category_str) tuples
    """

    print(f"[KML] Parsing KML with ElementTree: {kml_path}")
    tree = ET.parse(kml_path)
    root = tree.getroot()

    ns_any = "{*}"  # wildcard namespace
    polygons = []

    for pm in root.findall(".//{*}Placemark"):
        # --- visibility check ---
        vis_el = pm.find(f"{ns_any}visibility")
        if vis_el is not None and vis_el.text is not None:
            vis_text = vis_el.text.strip()
            # Google Earth uses 0/1; treat '0' as hidden
            if vis_text == "0":
                pm_name_el = pm.find(f"{ns_any}name")
                pm_name = pm_name_el.text.strip() if (pm_name_el is not None and pm_name_el.text) else ""
                print(f"[KML]   Skipping Placemark '{pm_name}' due to visibility=0")
                continue

        # Name / category
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
            if coords_el is None or coords_el.text is None:
                continue

            coord_text = coords_el.text.strip()
            points = []
            for line in coord_text.split():
                parts = line.split(",")
                if len(parts) >= 2:
                    lon_deg = float(parts[0])
                    lat_deg = float(parts[1])
                    x, y = latlon_to_local_xy(lat_deg, lon_deg, center_lat, center_lon)
                    points.append((x, y))

            if len(points) >= 3:
                poly = Polygon(points)
                polygons.append((poly, category))

                bx_min, by_min, bx_max, by_max = poly.bounds
                print(
                    f"[KML]   Polygon: name='{name}', "
                    f"type={category}, color={color}, "
                    f"bounds=({bx_min:.1f}, {by_min:.1f}) to ({bx_max:.1f}, {by_max:.1f})"
                )

    print(f"[KML] Loaded {len(polygons)} polygons")
    return polygons
