# Mesh Tally DataFrame Schema

Mesh tally files parsed with `msht_parser.parse_msht` or loaded through
`MeshTallyView` produce a `pandas.DataFrame` with the following columns:

| Column      | Type  | Description                       |
|-------------|-------|-----------------------------------|
| `x`         | float | X-coordinate of the mesh element |
| `y`         | float | Y-coordinate of the mesh element |
| `z`         | float | Z-coordinate of the mesh element |
| `result`    | float | Tally result value               |
| `rel_error` | float | Relative error of the result     |
| `volume`    | float | Volume of the voxel              |
| `result_vol`| float | `result * volume`                |

All values are represented as floating point numbers. Future analysis
functions can rely on these column names and types when consuming mesh
tally data.
