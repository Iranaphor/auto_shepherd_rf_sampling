# RF Signal Dropoff Predictor

Interactive tool for predicting peer-to-peer RF signal strength between two points based on obstacle signatures.

## Features

- **Interactive Map**: Click on the obstacle map to select transmitter and receiver locations
- **Real-time Prediction**: Visualizes expected RF signal strength based on learned signatures
- **Coverage Visualization**: Shows approximate signal coverage from the transmitter
- **Multiple Signatures**: When multiple signature matches exist, displays all of them

## Usage

### Prerequisites

1. Extended dataset with RF signatures (`dataset_extended.yaml`)
2. Obstacle map (KML file)
3. RF measurement points (YAML file with center coordinates)
4. X11 display for interactive visualization

### Running the Tool

```bash
# Allow X11 connections from docker
xhost +local:docker

# Run the predictor
docker compose -f docker-compose.predict-dropoff.yml up

# When done, revoke X11 access
xhost -local:docker
```

### Environment Variables

Configure in `docker-compose.predict-dropoff.yml`:
- `DATA_PATH`: Path to data directory containing KML and points.yaml
- `DATASET_PATH`: Path to extended dataset YAML file
- `RF_TYPE`: RF measurement type (default: `ssid-espnow`)
- `DISPLAY`: X11 display (automatically set from host)

### How It Works

1. **First Click (Green)**: 
   - Sets the transmitter location
   - Shows approximate coverage zones as green circles
   - Waits for receiver location

2. **Second Click (Red)**:
   - Sets the receiver location
   - Draws blue arrow connecting transmitter to receiver
   - Builds polar obstacle grid centered at transmitter
   - Extracts obstacle signature along the bearing to receiver
   - Finds matching signatures in the dataset
   - Plots predicted RF signal strength vs. distance

3. **Signal Prediction Graph**:
   - X-axis: Distance from transmitter (meters)
   - Y-axis: RSSI signal strength (dBm)
   - Multiple lines: Different measurement rays with the same signature
   - Red dashed line: Shows the selected point-to-point distance

4. **Click Again**: Select two new points for another prediction

## Technical Details

### Signature Matching

- First attempts exact signature match
- Falls back to partial prefix matching (minimum 20 characters or half signature length)
- Displays all matching signature data overlaid on the same graph

### Obstacle Grid

- Uses 5° angular resolution (72 bins around 360°)
- Uses 10m radial resolution
- Extends to 120% of point-to-point distance or 100m minimum
- Categories: unknown, open, lake, trees, building

### Visualization

- Top subplot: XY obstacle map with colored polygons based on dataset
- Bottom subplot: RF signal dropoff graph with all matching signatures
- Colors match those defined in dataset YAML file

## Example

```
[INIT] Loading dataset and obstacle map...
[DATASET] Loaded 46 signatures
[OBSTACLES] Loaded 8 polygons

Click transmitter location → Shows coverage zones
Click receiver location → Predicts signal dropoff

[SELECT] Transmitter at (50.0, 100.0)
[SELECT] Receiver at (150.0, -50.0)
[PREDICT] Distance: 180.3m, Bearing: 123.7°
[POLAR] Built obstacle grid: (72, 18)
[SIGNATURE] Ray 25 (125.0°): s11111144000000000000000000000000000000000000...
[MATCH] Found 1 matching signature(s)
```

## Troubleshooting

### X11 Display Issues

If you get "cannot connect to X server":
```bash
# Check DISPLAY is set
echo $DISPLAY

# Allow docker to access X11
xhost +local:docker

# If still failing, try
export DISPLAY=:0
```

### No Matching Signatures

If predictions show "No matching signature found":
- The obstacle pattern between your points may not exist in the training dataset
- Try selecting points in areas where RF measurements were collected
- Check that `DATASET_PATH` points to an extended dataset with signatures

### Interactive Display Not Showing

- Ensure matplotlib is using TkAgg backend (configured in script)
- Check X11 socket is mounted: `/tmp/.X11-unix:/tmp/.X11-unix`
- Verify network_mode is set to `host` in docker-compose
