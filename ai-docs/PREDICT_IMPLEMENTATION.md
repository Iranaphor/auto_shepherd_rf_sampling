# RF Signal Dropoff Predictor - Implementation Summary

## Overview

Created an interactive tool for predicting peer-to-peer RF signal dropoff between two points based on learned obstacle signatures from the extended dataset.

## Files Created/Modified

### 1. `/app/predict_dropoff.py` (NEW)
Main interactive prediction script with the following features:

#### Key Components:
- **RFDropoffPredictor Class**: Main predictor with interactive matplotlib interface
  - `__init__()`: Loads dataset, obstacles, builds color maps
  - `setup_figure()`: Creates 2-subplot figure (map + signal graph)
  - `on_click()`: Handles mouse clicks on obstacle map
  - `show_coverage()`: Displays transmitter coverage zones
  - `draw_connection()`: Draws arrow between selected points
  - `predict_signal_dropoff()`: Main prediction logic
  - `find_matching_signatures()`: Matches query signatures to dataset

#### Workflow:
1. Load extended dataset and obstacle map
2. Display interactive XY obstacle map with color-coded polygons
3. Wait for user to click transmitter location (green marker)
4. Show coverage zones from transmitter
5. Wait for user to click receiver location (red marker)
6. Build polar obstacle grid centered at transmitter
7. Extract obstacle signature along bearing to receiver
8. Find matching signatures in dataset
9. Plot all matching RF signal vs. distance curves
10. Mark selected distance with vertical line
11. Reset for next prediction

#### Technical Details:
- **Polar Grid**: 5° angular resolution, 10m radial resolution
- **Signature Matching**: Exact match first, then prefix matching (≥20 chars)
- **Visualization**: TkAgg backend for interactive display
- **Reuses Functions**: 
  - `load_dataset_yaml()` from dataset_utils
  - `load_kml_polygons()` from kml_utils
  - `load_yaml_points()` from utils
  - `build_polar_obstacle_grid_for_center()` from utils
  - `build_obstacle_type_map()` from dataset_utils
  - `compute_xy_extent()` from render_utils

### 2. `/docker-compose.predict-dropoff.yml` (MODIFIED)
Updated configuration for interactive X11 display:

#### Changes:
- Added X11 socket mount: `/tmp/.X11-unix:/tmp/.X11-unix`
- Added `network_mode: host` for X11 connectivity
- Added `DISPLAY` environment variable
- Removed `PLOT_OUTPUT` (not needed for interactive mode)
- Set command to run `predict_dropoff.py`

#### Environment Variables:
```yaml
- DATA_PATH=/data/riseholme_lake
- DATASET_PATH=/data/dataset_extended.yaml
- RF_TYPE=ssid-espnow
- DISPLAY=${DISPLAY}
```

### 3. `/app/README_PREDICT.md` (NEW)
Comprehensive documentation including:
- Feature description
- Usage instructions
- X11 setup requirements
- Environment variable configuration
- Interactive workflow explanation
- Technical implementation details
- Troubleshooting guide

## Usage

```bash
# 1. Enable X11 access
xhost +local:docker

# 2. Run the predictor
docker compose -f docker-compose.predict-dropoff.yml up

# 3. Interact with the display:
#    - Click transmitter location (green)
#    - Click receiver location (red)
#    - View predicted signal dropoff
#    - Click two new points to repeat

# 4. Clean up
xhost -local:docker
```

## Key Features Implemented

### ✅ Interactive Obstacle Map
- Color-coded polygons using dataset colors
- Click to select transmitter/receiver
- Shows coverage zones from transmitter
- Draws connection arrow between points

### ✅ Polar Grid Construction
- Centered at transmitter location
- Extracts obstacle signature along bearing
- Converts categories to numeric tags
- Handles arbitrary point positions

### ✅ Signature Matching
- Exact match prioritized
- Fallback to partial prefix matching
- Displays multiple matching signatures
- Handles case where no match exists

### ✅ Signal Prediction Visualization
- Plots RSSI vs. distance for all matching rays
- Multiple overlaid curves if multiple matches
- Marks selected point-to-point distance
- Clear axis labels and grid

### ✅ Reusable Components
- Leverages existing utility functions
- Uses dataset color definitions
- Maintains consistent styling
- Follows established code patterns

## Technical Architecture

```
predict_dropoff.py
├── Load dataset & obstacles
├── Setup matplotlib figure (2 subplots)
├── Event loop (click handler)
│   ├── First click: Set transmitter
│   │   └── Show coverage zones
│   └── Second click: Set receiver
│       ├── Build polar grid at transmitter
│       ├── Extract signature along bearing
│       ├── Match signature to dataset
│       └── Plot RF signal predictions
└── Reset for next iteration
```

## Dependencies

- matplotlib (TkAgg backend)
- numpy
- Existing utility modules:
  - kml_utils
  - utils
  - dataset_utils
  - render_utils

## Known Limitations

1. **X11 Required**: Needs X11 display server (not available on headless systems)
2. **Signature Coverage**: Predictions limited to signatures in training dataset
3. **Matching Tolerance**: Partial matching may not always find ideal signature
4. **Single Ray**: Uses single ray along bearing (doesn't consider multipath)
5. **Static Obstacles**: Assumes obstacle map doesn't change during prediction

## Future Enhancements

- [ ] Add confidence intervals based on variance in signature data
- [ ] Support for headless mode with saved output images
- [ ] Heatmap mode showing signal coverage for entire area
- [ ] Multiple receiver mode for multi-point communication
- [ ] Export predictions to CSV/JSON
- [ ] Real-time comparison with actual measurements
- [ ] Fresnel zone visualization
