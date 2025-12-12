# Extended Dataset Script

## Overview

The `extend_dataset.py` script extends an existing RF signature dataset with new measurements. It:

1. **Loads RF data** from `points.yaml` and obstacle map from `feature_map.kml`
2. **Converts** obstacle map and RF data into polar coordinate maps
3. **Associates** obstacle signatures with RF signatures
4. **Extends** the existing dataset by:
   - Adding new signatures where they don't exist
   - Extending data within existing signatures
   - Padding with NaN values when new bins are added
5. **Generates** before/after comparison visualizations

## Usage

```bash
python extend_dataset.py \
    --data-path data/riseholme_lake \
    --dataset dataset.yaml \
    --output dataset_updated.yaml \
    --plot-output comparison.png
```

### Required Arguments

- `--data-path`: Path to directory containing `points.yaml` and `feature_map.kml`
- `--dataset`: Path to existing dataset YAML file

### Optional Arguments

- `--output`: Output path for updated dataset (default: overwrite input)
- `--plot-output`: Path for before/after comparison plot (default: `dataset_comparison.png`)
- `--dtheta`: Angular bin size in degrees (default: 5.0)
- `--dr`: Radial bin size in meters (default: 10.0)
- `--max-range`: Maximum range in meters (default: auto-compute)
- `--rf-type`: RF data type key (default: `ssid-espnow`)

## Dataset Format

The dataset YAML format:

```yaml
obstacles:
- type: unknown
  signature_tag: 0
  rendering_colour: '#cccccc'
- type: open
  signature_tag: 1
  rendering_colour: '#ffffff'
# ...

signatures:
  s000000000000:
    bins: [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    ssid-espnow: 
    - [-85, -88, -90, NaN, -95, NaN, -102, NaN, -96, NaN, NaN, -87]
    - [-85, -88, -90, -92, -95, -100, -102, -98, -96, -93, -89, -87]
  # ...
```

## File Organization

### Main Script
- `extend_dataset.py` - Main execution script with clean function calls

### Utility Files
- `dataset_utils.py` - Dataset I/O and signature management
- `render_utils.py` - Visualization functions including comparison plots
- `utils.py` - Polar grid operations and RF data processing
- `kml_utils.py` - KML file parsing

### Example Workflow

1. Collect RF data in a new location and save as `points.yaml`
2. Use the same obstacle map `feature_map.kml`
3. Run the script to extend the dataset
4. Review the before/after comparison plot
5. Use the updated dataset for RF propagation modeling

## Output

The script produces:

1. **Updated dataset YAML** with extended/new signatures
2. **Comparison plot** showing:
   - Top left: Obstacle patterns
   - Top right: RF data before extension
   - Bottom left: Signature labels
   - Bottom right: RF data after extension
