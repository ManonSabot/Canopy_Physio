# TractLSM (a Tractable simplified Land Surface Model)

The **TractLSM** is a LSM framework that integrates a suite of competing plant
optimality principles that operate at different functional levels and over
various timescales.

The model is organised as follows:

```bash
TractLSM
├── run_homogeneous_surface.py
├── run_utils.py
├── CH2OCoupler
│   ├── ProfitMax.py
│   ├── USO.py
├── SPAC
│   ├── canatm.py
│   ├── hydraulics.py
│   ├── leaf.py
│   ├── soil.py
└── Utils
    ├── build_final_forcings.py
    ├── calculate_solar_geometry.py
    ├── constants_and_conversions.py
    ├── default_params.py
    ├── drivers_site_level.py
    ├── general_utils.py
    └── weather_generator.py
```

&nbsp;

`run_homogeneous_surface.py` is where the forcing is read, the main routines
called, and the output written. `run_utils.py` contains functions to support
these actions.

&nbsp;

The `CH2OCoupler/` is where you can find the `ProfitMax.py` approach, which is
derived/adapted from the work of
[Sperry et al. (2017)](https://doi.org/10.1111/pce.12852).
A more standard flux coupling method can be found in `USO.py`; it uses the
[Medlyn et al. (2011)](https://doi.org/10.1111/j.1365-2486.2010.02375.x) model.

&nbsp;

The model's biogeophysical routines can be found in the `SPAC/` repository,
ranging from micrometeorology (`canatm.py`) to plant hydraulics
(`hydraulics.py`).

&nbsp;

All support routines (automating the format of the input files, etc.) can be
found in `Utils/`.

&nbsp;
