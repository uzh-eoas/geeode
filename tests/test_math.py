#!/usr/bin/env python
# coding: utf-8

# !! Note:
# This test requires initiatliation and authentication to Google Earth Engine.
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
import random
import copy

# Import and initialize Earth Engine
import ee

# !! If you choose to run this test, you must update this project to your own!
ee.Initialize(project='uzheoas')

# Import the local modules
from ..src.geeode.geeode import *

# !! Input your GEE username, which should match the project initialized above
gee_username = 'uzheoas'

# Create a folder to situate the output of the pytest runs
pytest_folder = 'users/'+gee_username+'/pytest'

try:
    ee.data.getAsset(pytest_folder)
    print(f'Folder already exists: {pytest_folder}')
except ee.EEException:
    ee.data.createFolder(pytest_folder)
    print(f'Folder created: {pytest_folder}')
except Exception as error:
    print(f'Something went wrong when checking for the asset: {asset_id_to_test}')
    print(error)


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
                  "name":"exponential",
                  "boundsList":[[1, 9],[1, 9],[1, 9]],
                  "vars":['a','b','c']},
                 {"expression":"b('a') * sin(b('time') + b('b'))",
                  "name":"harmonic",
                  "boundsList":[[1, 9],[1, 9]],
                  "vars":['a','b']}]


paramList = []
expListOfDictCopy = expListOfDict

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
        l = numbers
        s = expOI["expression"]
        for placeholder, value in zip(v_form, l): s = s.replace(placeholder, str(value))
        # Change 'time' to 'num' for the example
        s = s.replace("b('time')", "b('num')")
        expListOfDictCopy[idx]["randExpression"] = s
        expListOfDictCopy[idx]["repl"] = repl
        paramList.append(dict(expListOfDictCopy[idx]))


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
    
    # Define an arbitrary area of interest and create the imagery there
    pOI = ee.Geometry.Point([8.548333, 47.374722])
    aOI = pOI.buffer(1000).bounds()
    projection = ee.Projection("EPSG:4326").atScale(100)

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

    # Apply the algebraic expression to the image collection
    def apply_expression(i):
        return (
            ee.Image(i.select('num')
                   .expression(randomizedFun)
                     .rename('num')
                     .copyProperties(i))
            .addBands(i.select('time'))
        )

    expressionApplied = numImColl.map(apply_expression)

    # Apply some random error to the series so it's ready to model // as an example dataset
    maxErrorDecimal = 0.1

    def computeError(i):
        seed = ee.Number(ee.Image(i).get('num'))
        randomField = ee.Image.random(seed, 'normal')
        maxError = i.select('num').multiply(ee.Image.constant(maxErrorDecimal))
        finalError = maxError.multiply(randomField)
        imageToCast = (
            i.select('num')
             .add(finalError)
             .rename('numWithError')
             .addBands(i.select('time'))
        )
        return ee.Image(imageToCast).float()

    timeSeries = (
        expressionApplied
        .map(computeError)
        .map(lambda i: i.set('system:footprint', aOI))
    )

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
replfam_filters = list(set(replfam_list))
replicate_families = [[asset for asset in asset_list if rf in asset] for rf in replfam_filters]
replicate_families


# Compute the fitness RMSE scores to test
nested_output_colls = [[ee.FeatureCollection(r) for r in rfg] for rfg in replicate_families]
output_colls = [ee.FeatureCollection(c).flatten().sort('repl',True) for c in nested_output_colls]
fitness_rmse = [ee.FeatureCollection(fc).aggregate_array('RMSE').getInfo() for fc in output_colls]
error_value_to_check = [all(x <= y for y, x in zip(l, l[1:])) for l in fitness_rmse]


# Compute the coefficients values to test
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


# Apply the tests

# Error/Loss Function Optimization
# Assert that the optimization of a loss/error function has occurred uniformly throughout the iterations.
# The test asserts that "If true, all replicates pass".
def test_error():
    """Return True if seq never increases (or remains unchanged) from one element to the next."""
    assert all(error_value_to_check)


# Make a function that tests whether numbers fall within the supplied bounds.
# The test asserts that "If true, all replicates pass".
def test_coefficients():
    """Return True if all generated coefficients fall within their designated bounds."""
    assert all(all(d.values()) for d in coeff_results)



