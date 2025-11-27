# GEEODE: A Google Earth Engine Implementation of Optimization by Differential Evolution

![NDVI Curve Fitting](docs/graphics/ee-chart.gif "curve_fitting")

## 🌍🕰️ Summary 📈📉

Do you commonly use time series data in Google Earth Engine? Are you interested in modelling these time series using mathematical models of arbitrary forms (e.g., linear, exponential, logarithmic, etc.)? If so, consider GEEODE as an option for your task.

With GEEODE you can optimize any arbitrary close-formed alegbraic model on a time series image collection using a process called differential evolution. Various options exist to fine-tune the analysis, and accompanying statistics measuring the degree of optimization (i.e., convergence to a final model) can also be produced.

[Here's](https://uzh-eoas.github.io/geeode/) the documentation.

For questions please contact: [Devin Routh](mailto:devin.routh@uzh.ch?subject=GEEODE%20Request) and/or [Claudia Röösli](mailto:claudia.roeoesli@geo.uzh.ch?subject=GEEODE%20Request)

### Repo Description

- `geeode.js`: the Google Earth Engine Code Editor functions; also importable via `require('users/uzheoas/geeode:geeode.js')`
- `src`: the Python implementation of the functions
- `paper`: the directory containing all of the materials for the manuscript
    - `figure_generation.py` creates the images within `docs/graphics`
- `docs`: the directory containing documentation files
- `README.md`: the document you're reading
- `requirements.txt`: a minimal description of the core GEEODE dependencies
- `requirements-tests.txt`: the packages required to serve the run PyTests
- `requirements-docs.txt`: the packages required to serve the documentation
- `LICENSE`: GEEODE's license file
- `pyproject.toml`: GEEODE's Python configuration file

## Installation

### Javascript

Using `geeode` from the [Google Earth Engine Javascript Playground](https://developers.google.com/earth-engine/guides/playground) is as simple as adding the following line to your script of interest:

```javascript
var geeode = require('users/uzheoas/geeode:geeode.js')
```

This functionality makes use of Earth Engines [script module](https://developers.google.com/earth-engine/guides/playground#script_modules) system.

### Python

To make use of GEEODE's workflow functionality via Python, you can install the module from source via:

```python
git clone https://github.com/uzh-eoas/geeode.git
cd geeode
pip install .
```

## Tests

GEEODE is equipped with a PyTest framework for affirming algorithmic fidelity. To run the tests:

```python
git clone https://github.com/uzh-eoas/geeode.git
cd geeode
pip install -r requirements-tests.txt
cd tests
pytest --capture=no
```

- Use the `--capture=no` flag to watch the progress via the test's printed statements
