#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
#
# @author      James R. Heselden (github: iranaphor)
# @maintainer  James R. Heselden (github: iranaphor)
# @datecreated 27st November 2025
# @credits     Code structure and implementation were developed by the
#              author with assistance from OpenAI's GPT-5.1 model, used
#              under the author's direction and supervision.
#
# ###########################################################################

import os

# =========================
# CONFIGURATION
# =========================

# Angular bin size in degrees (y-axis resolution)
DTHETA_DEG = float(os.getenv('ANGULAR_BIN_SIZE'))   # e.g. 5° -> 72 rows

# Radial bin size in meters (x-axis resolution)
DR_M = float(os.getenv('RADIAL_BIN_SIZE'))        # e.g. 10 m bins

# Max range (if None, computed from polygons + points)
MAX_RANGE_M = None

# Obstacle category colors (for plotting)
CATEGORY_COLORS = {
    "unknown":  "#cccccc",  # light grey background
    "open":     "#ffffff",  # white
    "lake":     "#0066ff",  # blue
    "trees":    "#00aa00",  # green
    "building": "#ff9900"   # orange
}
# Map obstacle categories to numeric codes (as characters)
OBSTACLE_CODES = {
    "unknown": "0",
    "open": "1",
    "lake": "2",
    "trees": "3",
    "building": "4"
}
