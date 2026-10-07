# Diamond NAgg Map

Calculates nitrogen concentration and aggregation state across FTIR maps of diamond plates. Each spectrum in the map gets a baseline correction, a thickness normalization and a least-squares fit to reference spectra, and the results are saved as maps.

## How it works

1. Reads a Thermo Fisher Nicolet OMNIC `.map` file with [FTIR_OMNIC_MAP_Reader](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader) into an xarray dataset.
2. Removes the baseline of each spectrum with asymmetric least squares (pybaselines).
3. Normalizes each spectrum for thickness by comparing it with a type IIa reference spectrum over 1800-2313 and 2390-2670 cm<sup>-1</sup>.
4. Fits the nitrogen region (up to 1400 cm<sup>-1</sup>) with a bounded least-squares combination of the C, A, X, B and D center reference spectra plus a constant and a linear term. The reference spectra come from the QUIDDIT program.
5. Converts the A and B fits to ppm (factors 16.5 and 79.4) and calculates total nitrogen and %B for every point.
6. Saves maps of A ppm, B ppm, %B, total N, C and D as PNG and TIFF, plus example plots of the baseline, type IIa and nitrogen fits on a grid of points across the map.

## Install and run

Requires Python 3.13 or later.

```bash
uv tool install git+https://github.com/whtowbin/Diamond_NAgg_Map
diamond-nagg-map path/to/sample.map --output Results
```

Version 0.8.0 on PyPI has a broken `diamond-nagg-map` command; use the GitHub install until 0.9.0 is published.

Options:

- `--output`: folder for the maps and plots (default `Results`).
- `--lam`, `--p`: baseline parameters passed to `pybaselines.whittaker.asls` (defaults 2e7 and 1e-6).
- `--n_examples`: grid of points to plot example fits for, either `N` for an N x N grid or `NX NY` (default 5). Each point produces three plots.

From Python:

```python
from diamond_nagg_map.DiamondNAggMap import process_map

process_map("sample.map", "Results", baseline_lam=1e7, baseline_p=1e-5, n_examples=(3, 3))
```

## Limitations

- Reads only OMNIC `.map` files from Nicolet FTIR microscopes.
- Fits type IaA/B diamonds. Type Ib support is not finished.
- Single-spectrum processing lives in [diamond-ftir-package](https://github.com/whtowbin/diamond-ftir-package).

## License

MIT. See [LICENSE](LICENSE).
