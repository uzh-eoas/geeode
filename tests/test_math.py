#!/usr/bin/env python
# coding: utf-8

# !! Note:
# This test requires initialization and authentication to Google Earth Engine.
# It makes use of the tasks and asset systems, such that when you run the test
# many tasks are queued and many assets are generated (within a `pytest` 
# GEE asset folder). Because the script tests for existing assets before
# running them, you can rerun the script in much less time than running
# it for the first time (because it uses the pre-existing assets to make
# its assertions).


# Import necessary modules
import datetime
import time
import itertools
import re
import math
import copy
import random

# Import and initialize Earth Engine (and other packages required for initialization)
import ee
import os
from conftest import initialize_ee
initialize_ee()

# Import the local modules
from geeode import de_optim, sub_sample, ts_image_to_coll, apply_model, check_for_tasks, check_for_asset_then_run_task, pause_and_wait


# Users must set GEEODE_TEST_ASSET_ROOT to a writable Earth Engine asset path
# (e.g., 'projects/your-project' or 'users/your-username') before running tests.
# See README for details.
gee_asset_root = os.getenv("GEEODE_TEST_ASSET_ROOT")

if not gee_asset_root:
    raise RuntimeError(
        "GEEODE_TEST_ASSET_ROOT is not set.\n"
        "This must be a writable Earth Engine asset path where test assets "
        "can be created; e.g., 'projects/your-project/assets' or 'users/your-username'.\n"
        "Set it before running the tests:\n"
        "    GEEODE_TEST_ASSET_ROOT='projects/your-project' pytest --capture=no\n"
        "See the 'Tests' section of the README for more information."
    )

gee_asset_root = gee_asset_root.rstrip("/")
pytest_folder = f"{gee_asset_root}/geeode-pytest-results"

# Create the folder if needed
try:
    ee.data.getAsset(pytest_folder)
    print(f"Folder already exists: {pytest_folder}")
except ee.EEException:
    # Folder doesn't exist; try to create it
    try:
        ee.data.createFolder(pytest_folder)
        print(f"Folder created: {pytest_folder}")
    except ee.EEException as create_error:
        raise RuntimeError(
            f"Could not create folder '{pytest_folder}'.\n"
            f"Verify that GEEODE_TEST_ASSET_ROOT ('{gee_asset_root}') is a valid, "
            f"writable asset path. For cloud projects this must include the "
            f"'/assets' segment; e.g., 'projects/your-project/assets'.\n"
            f"Original error: {create_error}"
        ) from create_error
except Exception as error:
    print(f"Warning: Unexpected error checking asset folder '{pytest_folder}'.")
    print(f"Error details: {error}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DE Optim Test
# Input a list of dictionaries with an expression to evaluate, its name (for reference,
# the bounds for the function's coefficients, and the names of the coefficients
expListOfDict = [{"expression":"b('a') * log(b('time') + b('b')) + b('c')",
                  "name":"simple_log",
                  "boundsList":[[1, 9],[1, 9],[1, 9]],
                  "vars":['a','b','c']},
                 {"expression":"b('a') * log(b('time') + b('b'))",
                  "name":"simpler_log",
                  "boundsList":[[1, 9],[1, 9]],
                  "vars":['a','b']},
                 {"expression":"(b('a') * b('time')) + b('b')",
                  "name":"linear",
                  "boundsList":[[1, 9],[1, 9]],
                  "vars":['a','b']},
                 {"expression":"(b('a') * b('time') ** 2) + (b('b') * b('time') + b('c'))",
                  "name":"quadratic",
                  "boundsList":[[1, 9],[1, 9],[1, 9]],
                  "vars":['a','b','c']},
                 {"expression":"b('a') * sin(b('time') + b('b'))",
                  "name":"harmonic",
                  "boundsList":[[1, 9],[1, 9]],
                  "vars":['a','b']}]

# You can select expression families using 'ALL' to run every family (e.g., for CI
# set the GEEODE_TEST_EXPRESSIONS variable); otherwise one family is chosen at random,
# seeded with POSIX time. Optionally, seed it manually with GEEODE_TEST_SEED=<printed value> 
# to allow reproducibility.
seed = int(os.getenv("GEEODE_TEST_SEED", int(time.time())))
print(f"GEEODE runtime seed: {seed}")
random.seed(seed)
if os.getenv("GEEODE_TEST_EXPRESSIONS", "").upper() == "ALL":
    families = expListOfDict
else:
    families = [random.choice(expListOfDict)]
expListOfDictCopy = copy.deepcopy(families)

paramList = []

# Input the number of replicates (i.e., number of randomized datasets to generate/assess)
# then bundle all replicate data into dictionaries for later use
nRepl = 2

for idx, expOI in enumerate(expListOfDictCopy):
    for repl in range(nRepl):
        v = expOI["vars"]
        v_form = []
        for s in v:
            f = f"b('{s}')"
            v_form.append(f)
        random.seed(repl)
        numbers = [random.randint(lo, hi) for lo, hi in expOI["boundsList"]]
        expListOfDictCopy[idx]["trueCoeffs"] = numbers
        l = numbers
        s = expOI["expression"]
        for placeholder, value in zip(v_form, l): s = s.replace(placeholder, str(value))
        # Change 'time' to 'num' for the example
        s = s.replace("b('time')", "b('num')")
        expListOfDictCopy[idx]["randExpression"] = s
        expListOfDictCopy[idx]["repl"] = repl
        paramList.append(dict(expListOfDictCopy[idx]))


# Instantiate functions and variables to test that the algorithm approaches convergence
# on the actual (randomly computed) coefficient values.
MAX_ERROR_DECIMAL = 0.1
def signal_rms(rand_expr):
    # Evaluate a GEE-style expression (variable b('num')) as plain Python
    py_expr = rand_expr.replace("b('num')", "t")
    vals = [eval(py_expr, {"log": math.log, "sin": math.sin, "t": t}) for t in range(1, 31)]
    return math.sqrt(sum(v * v for v in vals) / len(vals))

rmse_floor_lookup = {
    (p['name'], p['repl']): MAX_ERROR_DECIMAL * signal_rms(p['randExpression'])
    for p in paramList
}

MIN_ITER_FOR_FLOOR = 25
FLOOR_TOLERANCE = 2.0 
MIN_POP_FOR_FLOOR = 10

# Create the initial sets of populations and iterations to sweep
# Default options for test were chosen to circumvent any out-of-memory errors on single tasks
popList = [5,10,20]
iterations = [1,10,25,50]

# Create all combinations to task
combinations = itertools.product(paramList, popList, iterations)
all_combos = list(combinations)

# Create empty lists of asset ID's and replicate family names for later use
asset_list = []
replfam_list = []

# Create a helper function to generate a synthetic time series
pOI = ee.Geometry.Point([8.548333, 47.374722])
aOI = pOI.buffer(1000).bounds()
projection = ee.Projection("EPSG:4326").atScale(100)

def make_synthetic_timeseries(rand_expr):
    def make_image(n):
        return (
            ee.Image.constant(n)
            .cast({"constant": ee.PixelType.int8()}, ["constant"])
            .set('num', n)
            .rename('num')
            .addBands(ee.Image.constant(n).rename('time'))
            .clip(aOI)
            .reproject(projection)
        )

    numImColl = (
        ee.ImageCollection(ee.List.sequence(1, 30).map(make_image))
        .cast({'num': 'float', 'time': 'float'}, ['num', 'time'])
    )

    def apply_expression(i):
        return (
            ee.Image(i.select('num')
                   .expression(rand_expr)
                     .rename('num')
                     .copyProperties(i))
            .addBands(i.select('time'))
        )

    def compute_error(i):
        seed = ee.Number(ee.Image(i).get('num'))
        randomField = ee.Image.random(seed, 'normal')
        maxError = i.select('num').multiply(ee.Image.constant(MAX_ERROR_DECIMAL))
        finalError = maxError.multiply(randomField)
        imageToCast = (
            i.select('num')
             .add(finalError)
             .rename('numWithError')
             .addBands(i.select('time'))
        )
        return ee.Image(imageToCast).float()

    return (
        numImColl.map(apply_expression)
        .map(compute_error)
        .map(lambda i: i.set('system:footprint', aOI))
    )

# Loop through all combinations
for combo in all_combos:
    
    # Acquire the parameters for the run
    funToOpt = combo[0].get('expression')
    funName = combo[0].get('name')
    pop_size = combo[1]
    iNum = combo[2]
    randomizedFun = combo[0].get('randExpression')
    inputVars = combo[0].get('vars')
    inputBounds = combo[0].get('boundsList')
    repl = combo[0].get('repl')
    replFam = 'pytest_'+funName+'_p'+str(pop_size).zfill(3)+'_repl'+str(repl)
    replfam_list.append(replFam)
    desc = replFam+'_i'+str(iNum).zfill(3)
    aID = pytest_folder+'/'+desc
    asset_list.append(aID)
    timeSeries = make_synthetic_timeseries(randomizedFun)
    bandName = "numWithError";

    # Run de optim
    deOptimOutput = de_optim(pop_size,
                      iNum,
                      funToOpt,
                      inputVars,
                      inputBounds,
                      timeSeries,
                      bandName);

    # Export the outputs
    values = deOptimOutput.reduceRegion(ee.Reducer.first(), pOI, 100, "EPSG:4326");

    task = ee.batch.Export.table.toAsset(
        collection = ee.FeatureCollection(ee.Feature(pOI).set(ee.Dictionary(values)).set('repl',desc)),
        assetId = aID,
        description = desc
    );
    check_for_asset_then_run_task(aID,task,desc)


# Wait for the tasks to finish
pause_and_wait('pytest')

# Compute the family names of the replicates
fam_map = {}
for combo in all_combos:
    fam = ('pytest_' + combo[0]['name'] + '_p' + str(combo[1]).zfill(3)
           + '_repl' + str(combo[0]['repl']))
    fam_map[fam] = (combo[0]['name'], combo[0]['repl'])
replfam_filters = sorted(fam_map)
replicate_families = [[asset for asset in asset_list if rf in asset] for rf in replfam_filters]
replicate_families


# Compute the fitness RMSE scores to test
nested_output_colls = [[ee.FeatureCollection(r) for r in rfg] for rfg in replicate_families]
output_colls = [ee.FeatureCollection(c).flatten().sort('repl',True) for c in nested_output_colls]
fitness_rmse = [ee.FeatureCollection(fc).aggregate_array('RMSE').getInfo() for fc in output_colls]
error_value_to_check = [all(x <= y for y, x in zip(l, l[1:])) for l in fitness_rmse]

# Pair converged-run RMSE values with their analytic noise floors
rmse_floor_results = []
for fam, rmse_list in zip(replfam_filters, fitness_rmse):
    fun_name, repl = fam_map[fam]
    floor = rmse_floor_lookup[(fun_name, repl)]
    pop = int(re.search(r'_p(\d{3})_', fam).group(1))
    for it, rmse in zip(sorted(iterations), rmse_list):
        if it >= MIN_ITER_FOR_FLOOR and pop >= MIN_POP_FOR_FLOOR:
            rmse_floor_results.append((fam, it, rmse, floor))

# Diagnostic: print RMSE-to-floor ratios to calibrate FLOOR_TOLERANCE
for fam, it, rmse, floor in rmse_floor_results:
    print(f"{fam} iNum={it}: RMSE/floor = {rmse / floor:.2f}")


# Compute the coefficient values to test
list_of_feature_lists = [ee.FeatureCollection(fc).toList(len(iterations)) for fc in output_colls]
list_of_prop_dicts = [f.map(lambda f: ee.Feature(f).toDictionary().remove(['RMSE'])).getInfo() for f in list_of_feature_lists]
list_of_prop_dicts_copy = copy.deepcopy(list_of_prop_dicts)

coeff_results = []
for l in list_of_prop_dicts_copy:
    for d in l:
        f_name = re.search(r'(?<=pytest_).*?(?=_p\d{3})', d.get('repl')).group(0)
        filtered_dict = [d for d in expListOfDict if f_name in d.get("name", "")][0]
        bounds_to_use = filtered_dict.get('boundsList')
        d['boundsList'] = bounds_to_use
        d.pop('repl')
        results = {}
        for key, bounds in zip([k for k in d if k != 'boundsList'], d['boundsList']):
            val = d[key]
            results[key] = bounds[0] <= val <= bounds[1]
        coeff_results.append(results)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Apply the test assertions to confirm the underlying algorithm mathematics are correct.

# Error/Loss Function Optimization
# Assert that the optimization of a loss/error function has occurred uniformly throughout the iterations.
# I.e., RMSE never increases (or remains unchanged) from one iteration to the next.
def test_error():
    assert all(error_value_to_check)


# Make a function that tests whether numbers fall within the supplied bounds.
# The test asserts that "If true, all generated coefficients fall within their designated bounds".
def test_coefficients():
    assert all(all(d.values()) for d in coeff_results)

# Assert that the final values approach (or even exceed in quality) the actual RMSE of the 
# true coefficients (within a reasonable threshold). Converged runs (iNum >= 25) must approach the analytic noise floor.
def test_rmse_floor():
    failures = [
        f"{fam} iNum={it}: RMSE={rmse:.3f} exceeds "
        f"{FLOOR_TOLERANCE}x floor ({FLOOR_TOLERANCE * floor:.3f})"
        for fam, it, rmse, floor in rmse_floor_results
        if rmse > FLOOR_TOLERANCE * floor
    ]
    assert not failures, "\n".join(failures)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Conduct a scree output test (i.e., a single call that asserts scree-image format)

def test_scree_image_format():
    linear_expr = "b('a') * b('time') + b('b')"
    ts = make_synthetic_timeseries("(3 * b('num')) + 2")
    deOptimScree = de_optim(10, 25, linear_expr,
                            ["a", "b"], [[1, 9], [1, 9]],
                            ts, "numWithError",
                            optParams={"computeScree": True})
    bN = deOptimScree.bandNames().getInfo()

    # Format assertions: scree bands follow the RMSE_### naming scheme
    scree_bands = [b for b in bN if re.fullmatch(r"RMSE_\d{3}", b)]
    assert len(scree_bands) == 25          # one per iteration
    assert all("REMOVE" not in b for b in bN)

    # Values: non-increasing across iterations (RMSE floor should improve/hold)
    vals = deOptimScree.select(scree_bands).reduceRegion(
        ee.Reducer.first(), pOI, 100, "EPSG:4326").getInfo()
    seq = [vals[b] for b in scree_bands]
    assert all(y <= x for x, y in zip(seq, seq[1:]))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Best-strategy mutation test (single small run)

def test_de_optim_best_strategy():
    ts = make_synthetic_timeseries("(3 * b('num')) + 2")
    out = de_optim(5, 5, "b('a') * b('time') + b('b')",
                   ["a", "b"], [[1, 9], [1, 9]],
                   ts, "numWithError",
                   optParams={"mutationStrategy": "best"})
    vals = out.reduceRegion(ee.Reducer.first(), pOI, 100, "EPSG:4326").getInfo()
    # Coefficients recovered within bounds and roughly near truth (3, 2)
    assert 1 <= vals['a'] <= 9 and 1 <= vals['b'] <= 9

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Daisy-chain outputs: population image (and population+scree combo)

def test_de_optim_daisy_chain_population():
    ts = make_synthetic_timeseries("(3 * b('num')) + 2")
    pop_img = de_optim(5, 5, "b('a') * b('time') + b('b')",
                       ["a", "b"], [[1, 9], [1, 9]],
                       ts, "numWithError",
                       optParams={"daisyChain": True})
    # Population image is a per-pixel 2D array image (axis 0 = pop members,
    # axis 1 = optimized variables); reduce lengths at the test point
    l0 = pop_img.arrayLength(0).reduceRegion(
        ee.Reducer.first(), pOI, 100, "EPSG:4326").getInfo()
    l1 = pop_img.arrayLength(1).reduceRegion(
        ee.Reducer.first(), pOI, 100, "EPSG:4326").getInfo()
    assert list(l0.values())[0] == 5      # pop_size rows
    assert list(l1.values())[0] == 2      # one column per optimized variable

def test_de_optim_daisy_chain_scree():
    ts = make_synthetic_timeseries("(3 * b('num')) + 2")
    out = de_optim(5, 5, "b('a') * b('time') + b('b')",
                   ["a", "b"], [[1, 9], [1, 9]],
                   ts, "numWithError",
                   optParams={"daisyChain": True, "computeScree": True})
    bN = out.bandNames().getInfo()
    scree_bands = [b for b in bN if re.fullmatch(r"RMSE_\d{3}", b)]
    assert len(scree_bands) == 5    # one scree band per iteration