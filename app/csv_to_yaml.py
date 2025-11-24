#!/usr/bin/env python3
import csv

CENTER = [53.26831, -0.52984]

csv_path = "RIH-all.csv"
yaml_path = "points.yaml"

with open(csv_path, newline="") as csvfile, open(yaml_path, "w") as yamlfile:
    reader = csv.reader(csvfile)

    # Header comments
    yamlfile.write("# ----------------------\n")
    yamlfile.write("# YAML\n")
    yamlfile.write("#\n")
    yamlfile.write("# ----------------------\n")

    # Center line
    yamlfile.write(f"center: [{CENTER[0]}, {CENTER[1]}]\n")
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
