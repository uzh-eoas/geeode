#!/usr/bin/env python
# coding: utf-8

# Import necessary modules
import statistics

# Import and initialize Earth Engine (and other packages required for initialization)
import ee
from conftest import initialize_ee
initialize_ee()

# Import the local modules
from geeode import de_optim, sub_sample, ts_image_to_coll, apply_model, check_for_tasks, check_for_asset_then_run_task, pause_and_wait

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a synthetic time series locally and test it against the function

# UZH
aOI = ee.Geometry.Point([8.548333, 47.374722]).buffer(500).bounds()
pOI = ee.Geometry.Point([8.548333, 47.374722])

TIMES = [0, 2, 4, 10, 15, 40, 55, 57, 59, 100]
EXPECTED_DENSITIES = [5, 5, 5, 5, 5, 2, 4, 3, 3, 1]


# Make a synhetic timeseries using a list of artificial "times"
def make_synthetic_coll(times):
    imgs = [
        ee.Image.constant(i).rename('num')
        .addBands(ee.Image.constant(float(t)).rename('time'))
        .set('system:footprint', aOI).clip(aOI).float()
        for i, t in enumerate(times)
    ]
    return ee.ImageCollection.fromImages(imgs)

# Make a Python version of sub_sample's temporal-density calculation
def density_replica(times, n_st_d=0.5):
    w = n_st_d * statistics.pstdev(times)
    return [sum(1 for t2 in times if abs(t2 - t) < w) for t in times]


# Make a wrapper to handle the function
def run_sub_sample(s_type, n_keep, **opt_params):
    img = sub_sample(iC=make_synthetic_coll(TIMES), nKeep=n_keep,
                     sType=s_type, bandName='num', optParams=opt_params)
    vals = img.reduceRegion(ee.Reducer.first(), pOI, 100, 'EPSG:4326').getInfo()
    pairs = []
    for k in range(1, n_keep + 1):
        v = vals.get(f'b{k:02d}_num')
        t = vals.get(f'b{k:02d}_time')
        if v is not None and t is not None:
            pairs.append((float(v), float(t)))
    return pairs

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Apply the test assertions

# Check the local Python version against the known densities
def test_density_replica():
    assert density_replica(TIMES) == EXPECTED_DENSITIES


# Test each of the various sampling strategies acccording to a known synthetic time series.
# Also ensure that legitimate 0 time values remain

def test_subsample_bulk():
    assert sorted(run_sub_sample('bulk', 2, randomizeSort=False)) == [(5.0, 40.0), (9.0, 100.0)]
    assert sorted(run_sub_sample('bulk', 4, randomizeSort=False)) == [(5.0, 40.0), (7.0, 57.0), (8.0, 59.0), (9.0, 100.0)]
    assert sorted(run_sub_sample('bulk', 5, randomizeSort=False)) == [(5.0, 40.0), (6.0, 55.0), (7.0, 57.0), (8.0, 59.0), (9.0, 100.0)]
    pairs = run_sub_sample('bulk', 8)
    assert (0.0, 0.0) in pairs
    assert len(pairs) == 8


# Assure there are no extra padded zeroes (that should be masked)
def test_subsample_bulk_padding():
    pairs = run_sub_sample('bulk', 12, randomizeSort=False)
    assert len(pairs) == 10
    assert (0.0, 0.0) in pairs


def test_subsample_leapfrog():
    assert sorted(run_sub_sample('leapfrog', 8, randomizeSort=False)) == [
        (0.0, 0.0), (1.0, 2.0), (3.0, 10.0), (5.0, 40.0),
        (6.0, 55.0), (7.0, 57.0), (8.0, 59.0), (9.0, 100.0)]
    pairs = run_sub_sample('leapfrog', 5, randomizeSort=False)
    valid_inputs = {(float(i), float(t)) for i, t in enumerate(TIMES)}
    # Ensure the outputted values are genuine
    assert set(pairs) <= valid_inputs
    assert len(pairs) == 5
    assert [t for _, t in pairs] == sorted(t for _, t in pairs)


def test_subsample_splitshuffle_structure():
    assert sorted(run_sub_sample('splitshuffle', 4, randomizeSort=False)) == [
        (3.0, 10.0), (5.0, 40.0), (6.0, 55.0), (9.0, 100.0)]
    pairs = run_sub_sample('splitshuffle', 5, randomizeSort=False)
    assert (0.0, 0.0) in pairs
    valid_inputs = {(float(i), float(t)) for i, t in enumerate(TIMES)}
    # Ensure the outputted values are genuine
    assert set(pairs) <= valid_inputs
    assert len(pairs) == 5
    assert [t for _, t in pairs] == sorted(t for _, t in pairs)


# Ensure the randomization does not yield the same result as raw density weights
def test_randomize_sort_actually_randomizes():
    run_ordered = run_sub_sample(s_type='bulk', n_keep=8, randomizeSort=False)
    run_randomized = run_sub_sample(s_type='bulk', n_keep=8, randomizeSort=True, seedNum=7)
    assert run_ordered != run_randomized


