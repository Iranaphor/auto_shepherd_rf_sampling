#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###########################################################################
# Configuration settings for RF sampling and obstacle analysis
#
# @author      James R. Heselden (github: Iranaphor)
# @maintainer  James R. Heselden (github: Iranaphor)
# @datecreated 27th November 2025
# @datemodified 12th December 2025
# @credits     Developed with assistance from Claude Sonnet 4.5 and GitHub
#              Copilot.
# ###########################################################################

import os


# Angular bin size in degrees (y-axis resolution)
DTHETA_DEG = float(os.getenv('ANGULAR_BIN_SIZE'))

# Radial bin size in meters (x-axis resolution)
DR_M = float(os.getenv('RADIAL_BIN_SIZE'))

# Max range (if None, computed from polygons + points)
MAX_RANGE_M = None

# Obstacle category colors for plotting
CATEGORY_COLORS = {
    "unknown":  "#cccccc",
    "open":     "#ffffff",
    "lake":     "#0066ff",
    "trees":    "#00aa00",
    "building": "#ff9900"
}

# Map obstacle categories to numeric codes
OBSTACLE_CODES = {
    "unknown": "0",
    "open": "1",
    "lake": "2",
    "trees": "3",
    "building": "4"
}
