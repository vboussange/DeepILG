# Artificial-coordinate fixture

`artificial_coordinates.csv` is a deterministic software fixture for all 68
public population identifiers and five region keys. Coordinates lie on small
regular grids around arbitrary centres in Montana, Arizona, South Dakota,
Nevada, and Idaho. They were created independently of genetic values and of
all exact, relative, or obscured sampling locations.

These are not ecological observations, anonymized coordinates, or plausible
reconstructions. Every product derived from this fixture must retain:

```text
geolocation_status = artificial
scientific_use = false
```

Outputs are suitable only for exercising the software pipeline and must not be
interpreted, compared with the manuscript results, or published as scientific
findings.
