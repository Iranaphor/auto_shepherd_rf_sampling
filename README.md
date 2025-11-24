# Simple run:
First cd into this directory then run `docker compose up`.

# Run on your own data:
To run this on your own data, first create a new subdirectory in `./data`.

## CSV Convertor
If you have a csv file with data stored as [value, lat, lon, _, _],
then you can add this directly into the `./data/mydataset/all.csv`.
This data will be converted automatically into the correct format for use.

You will need to set the global lat/lon centre point, in the `docker-compose.yml`.

```yml
    environment:
      - DATA_PATH=/data/mydataset
      - CENTER_LAT=53.26831
      - CENTER_LON=-0.52984
```

## Points
Format your recieved rf points as:
```
center: [53.26831, -0.52984]
gps_coords_xyv:
  -  [53.2683631, -0.5298106, 58.0]
  -  [53.2683639, -0.5298107, 57.0]
```
Where the cente: denoted the gps position of the static point communicated to. 
The `gps_coords_xyv` contains a list of [lat, long, data] sampled manually in field.

## Feature Map
Alongside this points.yaml, include you should include a kml file named `feature_map.kml`
This KML should consist of a set of polygons (no nesting).
Each polygon should be named on of the following: `building`, `trees`, `lake`, `open`.
This can be created using GoogleEarth and exported directly as a local KML.
