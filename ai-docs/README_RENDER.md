# Render Class Documentation

## Quick Start

```python
from render import Render

# Initialize with data
renderer = Render(
    obstacle_kml_path="data/feature_map.kml",
    rf_data_path="data/points.yaml"
)

# Load the data
renderer.load_data(center_lat=51.5, center_lon=-0.1)

# Render various visualizations
renderer.render(
    obstacle_edge=True,
    obstacle_fill=True,
    rf_data=True,
    use_polar=False,
    fileout="output/visualization.png"
)
```

## Class Overview

The `Render` class provides a unified interface for creating obstacle maps and RF data visualizations with consistent styling.

### Key Features
- ✅ Unified rendering interface
- ✅ Consistent styling across all plots
- ✅ Support for both XY and polar coordinates
- ✅ Backward compatible with existing code
- ✅ Flexible data loading (file-based or programmatic)

## API Reference

### Constructor

```python
Render(obstacle_kml_path=None, rf_data_path=None)
```

**Parameters:**
- `obstacle_kml_path` (str, optional): Path to KML file with obstacle polygons
- `rf_data_path` (str, optional): Path to RF data YAML file

### Main Method

```python
render(obstacle_edge=False, obstacle_fill=False, rf_data=False, use_polar=False, fileout=None)
```

**Parameters:**
- `obstacle_edge` (bool): Draw obstacle boundaries
- `obstacle_fill` (bool): Fill obstacle regions with category colors
- `rf_data` (bool): Show RF data as scatter/heatmap overlay
- `use_polar` (bool): Use polar coordinates (False = XY plane)
- `fileout` (str, optional): Output file path; if None, returns figure object

**Returns:**
- `matplotlib.figure.Figure` if `fileout` is None
- `None` if `fileout` is provided (saves to file)

### Data Loading Methods

```python
load_data(center_lat=None, center_lon=None)
```
Load data from file paths specified in constructor.

```python
set_polar_grid(angles_deg, radii_m, obstacle_grid)
```
Set precomputed polar grid data.
- `angles_deg`: Array of angle bin centers (degrees)
- `radii_m`: Array of radius bin centers (meters)
- `obstacle_grid`: (n_theta, n_r) array of category strings

```python
set_heatmap(heatmap)
```
Set precomputed heatmap data (n_theta, n_r) array.

### Convenience Methods

#### XY Plane Visualizations

```python
render_xy_obstacles(fileout)
```
Render filled obstacles with edges in XY plane.

```python
render_xy_boundaries_rf(fileout)
```
Render obstacle boundaries with RF data overlay.

```python
render_xy_obstacles_rf(fileout)
```
Render filled obstacles with RF data overlay.

#### Polar Visualizations

```python
render_obstacle_polar(fileout)
```
Render obstacle categories in polar coordinates.

```python
render_heatmap_polar(fileout)
```
Render RF heatmap in polar coordinates.

```python
render_signature_slices(unique_sigs, mean_slices, fileout)
```
Render 2-panel signature analysis plot.
- `unique_sigs`: List of signature strings
- `mean_slices`: (num_sigs, n_r) array of mean RF values
- `fileout`: Output file path

## Usage Examples

### Example 1: Basic XY Visualization

```python
from render import Render

renderer = Render("obstacles.kml", "rf_data.yaml")
renderer.load_data(center_lat=51.5, center_lon=-0.1)

# Just obstacles
renderer.render(
    obstacle_fill=True,
    obstacle_edge=True,
    fileout="obstacles.png"
)

# Obstacles + RF data
renderer.render(
    obstacle_fill=True,
    obstacle_edge=True,
    rf_data=True,
    fileout="obstacles_with_rf.png"
)
```

### Example 2: Polar Visualization

```python
import numpy as np
from render import Render

renderer = Render()

# Set polar grid data
angles_deg = np.arange(0, 360, 5)
radii_m = np.arange(0, 1000, 10)
obstacle_grid = build_polar_grid(...)  # your polar grid

renderer.set_polar_grid(angles_deg, radii_m, obstacle_grid)

# Render obstacle polar map
renderer.render(
    obstacle_fill=True,
    use_polar=True,
    fileout="polar_obstacles.png"
)

# Add heatmap
heatmap = build_points_heatmap(...)
renderer.set_heatmap(heatmap)

renderer.render(
    rf_data=True,
    use_polar=True,
    fileout="polar_heatmap.png"
)
```

### Example 3: Programmatic Data Setting

```python
from render import Render
from shapely.geometry import Polygon

renderer = Render()

# Set polygon data directly
poly1 = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
poly2 = Polygon([(200, 0), (300, 0), (300, 100), (200, 100)])

renderer.polygons = [
    (poly1, "trees"),
    (poly2, "building")
]

# Set RF points (x, y, value)
renderer.local_rf_points = [
    (50, 50, 0.5),
    (150, 50, 0.7),
    (250, 50, 0.9)
]

# Render
renderer.render_xy_obstacles_rf("output.png")
```

### Example 4: Using Convenience Methods

```python
from render import Render

renderer = Render("obstacles.kml", "rf_data.yaml")
renderer.load_data(center_lat=51.5, center_lon=-0.1)

# Multiple visualizations with consistent styling
renderer.render_xy_obstacles("01_obstacles.png")
renderer.render_xy_boundaries_rf("02_boundaries_rf.png")
renderer.render_xy_obstacles_rf("03_obstacles_rf.png")
```

### Example 5: Signature Slices

```python
import numpy as np
from render import Render

renderer = Render()

# Example signature data
unique_sigs = ['000011', '000122', '001233']
mean_slices = np.random.rand(3, 100)
radii_m = np.arange(0, 1000, 10)

renderer._radii_m = radii_m
renderer.render_signature_slices(
    unique_sigs,
    mean_slices,
    "signature_analysis.png"
)
```

## Styling Customization

All styling is centralized in `render_utils.py`. To customize:

### Change Figure Sizes

```python
from render_utils import get_default_figsize

# Modify in render_utils.py
def get_default_figsize(aspect='landscape'):
    if aspect == 'square':
        return (10, 10)  # Changed from (8, 8)
    # ...
```

### Change DPI

```python
def get_default_dpi():
    return 300  # Changed from 200
```

### Change Colors

Colors are defined in `config.py`:

```python
CATEGORY_COLORS = {
    "unknown":  "#cccccc",
    "open":     "#ffffff",
    "lake":     "#0066ff",
    "trees":    "#00aa00",
    "building": "#ff9900"
}
```

## Integration with Existing Code

The refactoring maintains backward compatibility. All existing functions in `utils.py` work as before:

```python
from utils import (
    plot_obstacle_grid,
    plot_heatmap_with_boundaries,
    plot_signature_slices,
    plot_xy_obstacle_map,
    plot_xy_obstacle_boundaries_with_rf,
    plot_xy_obstacles_with_rf
)

# These all work exactly as before
plot_obstacle_grid(angles_deg, radii_m, obstacle_grid, "output.png")
plot_xy_obstacle_map(polygons, "output.png")
# etc.
```

Internally, these functions now use the `Render` class, ensuring consistent styling.

## Render Utils Functions

The `render_utils.py` module provides helper functions for custom rendering:

### Styling
- `get_default_figsize(aspect)` - Get consistent figure sizes
- `get_default_dpi()` - Get consistent DPI
- `apply_plot_styling(ax, xlabel, ylabel, title)` - Apply standard styling

### Colormaps
- `get_obstacle_colormap(categories)` - Get obstacle categorical colormap
- `get_obstacle_legend_patches()` - Get legend patches for obstacles

### Extent Computation
- `compute_xy_extent(polygons, local_points, margin_factor)` - Compute XY extents
- `compute_polar_extent(radii_m, angles_deg)` - Compute polar extents

### Drawing
- `draw_polygons_xy(ax, polygons, show_fill, show_edges, alpha)` - Draw polygons
- `draw_rf_scatter(ax, local_points, s, alpha, cmap)` - Draw RF scatter
- `add_colorbar(fig, mappable, ax, label)` - Add colorbar
- `add_legend(ax, location, fontsize)` - Add legend

### Grid Conversion
- `obstacle_grid_to_numeric(obstacle_grid, categories)` - Convert to numeric
- `signature_to_obstacle_grid(unique_sigs, n_r)` - Convert signatures to grid

## Error Handling

The `Render` class validates data before rendering:

```python
renderer = Render()

# This will raise ValueError
try:
    renderer.render_xy_obstacles("output.png")
except ValueError as e:
    print(f"Error: {e}")
    # Error: No polygon data loaded. Call load_data() first.
```

Always load or set data before rendering:

```python
# Option 1: Load from files
renderer.load_data(center_lat=51.5, center_lon=-0.1)

# Option 2: Set directly
renderer.polygons = [...]
renderer.local_rf_points = [...]
```

## Performance Considerations

- The `Render` class caches data to avoid recomputation
- Polar grid computation can be expensive for large grids
- Use `set_polar_grid()` to avoid recomputing grids
- File I/O happens only in `load_data()` method

## Testing

Run the included examples:

```bash
cd app
ANGULAR_BIN_SIZE=5 RADIAL_BIN_SIZE=10 python3 example_render_usage.py
```

Test imports:

```bash
cd app
ANGULAR_BIN_SIZE=5 RADIAL_BIN_SIZE=10 python3 -c "from render import Render; print('OK')"
```

## Dependencies

Required packages:
- `numpy` - Array operations
- `matplotlib` - Plotting
- `shapely` - Geometry operations
- `fastkml` - KML parsing (for file loading)
- `pyyaml` - YAML parsing (for file loading)

## Troubleshooting

### Import Error: float() argument must be a string

**Problem:** Environment variables not set.

**Solution:**
```bash
export ANGULAR_BIN_SIZE=5
export RADIAL_BIN_SIZE=10
```

Or set in Python:
```python
import os
os.environ['ANGULAR_BIN_SIZE'] = '5'
os.environ['RADIAL_BIN_SIZE'] = '10'
```

### ValueError: No polygon data loaded

**Problem:** Forgot to load or set data.

**Solution:**
```python
renderer.load_data(center_lat=51.5, center_lon=-0.1)
# or
renderer.polygons = [...]
```

### AttributeError: 'Render' object has no attribute '_radii_m'

**Problem:** Trying to use polar rendering without setting grid.

**Solution:**
```python
renderer.set_polar_grid(angles_deg, radii_m, obstacle_grid)
```

## See Also

- `REFACTORING_SUMMARY.md` - Details on the refactoring process
- `example_render_usage.py` - Complete usage examples
- `config.py` - Configuration and color definitions
- `render_utils.py` - Rendering utility functions
