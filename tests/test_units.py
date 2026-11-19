#!/usr/bin/env python
# coding: utf-8

# !! Note:
# These tests expand coverage beyond the math and subsampling verification.

# Import necessary modules
import pytest

# Import and initialize Earth Engine (and other packages required for initialization)
import ee
from conftest import initialize_ee
initialize_ee()

# Import the local modules
from geeode import de_optim, sub_sample, ts_image_to_coll, apply_model, check_for_tasks, check_for_asset_then_run_task, pause_and_wait

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Test de_optim input validations

def test_de_optim_missing_required_args():
    with pytest.raises(ValueError, match="Missing required argument"):
        de_optim(funToOpt=None, inputVars=["a"], inputBounds=[[1, 9]],
                 timeSeries="fake", bandName="num")
    with pytest.raises(ValueError, match="Missing required argument"):
        de_optim(funToOpt="b('a')", inputVars=["a"], inputBounds=[[1, 9]],
                 timeSeries="fake", bandName=None)


def test_de_optim_bad_expression_format():
    with pytest.raises(ValueError, match="appropriate format"):
        de_optim(funToOpt="a * time", inputVars=["a"],
                 inputBounds=[[1, 9]], timeSeries="fake", bandName="num")


def test_de_optim_var_list_mismatch():
    with pytest.raises(ValueError, match="match"):
        de_optim(funToOpt="b('a') * b('time')", inputVars=["a", "b"],
                 inputBounds=[[1, 9], [1, 9]], timeSeries="fake", bandName="num")


def test_de_optim_bounds_length_mismatch():
    with pytest.raises(ValueError, match="numeric bounds"):
        de_optim(funToOpt="b('a') * b('time')", inputVars=["a"],
                 inputBounds=[[1, 9], [1, 9]], timeSeries="fake", bandName="num")


def test_de_optim_invalid_M_and_cr():
    with pytest.raises(ValueError, match="M must be"):
        de_optim(funToOpt="b('a') * b('time')", inputVars=["a"],
                 inputBounds=[[1, 9]], timeSeries="fake", bandName="num",
                 optParams={"M": 0})
    with pytest.raises(ValueError, match="cr must be"):
        de_optim(funToOpt="b('a') * b('time')", inputVars=["a"],
                 inputBounds=[[1, 9]], timeSeries="fake", bandName="num",
                 optParams={"cr": 1.0})


def test_de_optim_invalid_mutation_strategy():
    with pytest.raises(ValueError, match="suggested strategies"):
        de_optim(funToOpt="b('a') * b('time')", inputVars=["a"],
                 inputBounds=[[1, 9]], timeSeries="fake", bandName="num",
                 optParams={"mutationStrategy": "nonsense"})


# Test the sub_sample input validation

def test_sub_sample_missing_required_args():
    with pytest.raises(ValueError, match="Missing required argument"):
        sub_sample(iC="fake", nKeep=5, sType=None, bandName="num")
    with pytest.raises(ValueError, match="Missing required argument"):
        sub_sample(sType="bulk", bandName="num")


# Test the ts_image_to_coll function input validation

def test_ts_image_to_coll_requires_args():
    with pytest.raises(ValueError):
        ts_image_to_coll()
    with pytest.raises(ValueError):
        ts_image_to_coll(band_name="num", ts_length=2)
    with pytest.raises(ValueError):
        ts_image_to_coll(ee.Image(0), "num")


def test_ts_image_to_coll_band_naming():
    # With ts_length omitted
    with pytest.raises(ValueError, match="ts_length"):
        ts_image_to_coll(ee.Image(0), "num")


# Test the apply_model function (requires an EE connection as it makes a getInfo call)

def test_apply_model_adds_predicted():
    pOI = ee.Geometry.Point([8.548333, 47.374722])
    i1 = (ee.Image.constant(1).rename('num')
          .addBands(ee.Image.constant(0.5).rename('time')))
    coll = ee.ImageCollection.fromImages([i1])
    coeffs = (ee.Image.constant(2).rename('a')
              .addBands(ee.Image.constant(3).rename('b')))
    out = apply_model(coll, coeffs, "b('num') * b('a') + b('b') + b('time')")
    img = ee.Image(out.first())
    assert 'predicted' in img.bandNames().getInfo()
    pred = img.reduceRegion(ee.Reducer.first(), pOI, 100).getInfo()['predicted']
    assert pred == pytest.approx(5.5)  # (1*2) + 3 + 0.5


# Test the task/asset helper functions in a mocked-up way with no EE connection

class FakeTask:
    def __init__(self, s):
        self._s = s
        self.started = False
    def __str__(self):
        return self._s
    def start(self):
        self.started = True


def test_check_for_tasks(monkeypatch):
    tasks = [FakeTask("export_pytest_bulk (RUNNING)"),
             FakeTask("export_pytest_bulk (FAILED)"),
             FakeTask("export_other (COMPLETED)")]
    monkeypatch.setattr(ee.batch.Task, "list", lambda: tasks)
    assert check_for_tasks("pytest_bulk") is True
    assert check_for_tasks("other") is False      # matched but not RUNNING/READY
    assert check_for_tasks("nonexistent") is False


def _asset_missing(monkeypatch, queued):
    def boom(aid):
        raise ee.EEException("not found")
    monkeypatch.setattr(ee.data, "getAsset", boom)
    if queued:
        monkeypatch.setattr(ee.batch.Task, "list",
                            lambda: [FakeTask("export_pytest_x (READY)")])
    else:
        monkeypatch.setattr(ee.batch.Task, "list", lambda: [])


def test_asset_exists_task_not_started(monkeypatch, capsys):
    monkeypatch.setattr(ee.data, "getAsset", lambda aid: {})
    task = FakeTask("export_pytest_x")
    check_for_asset_then_run_task("users/u/a", task, "pytest_x")
    assert task.started is False
    assert "already exists" in capsys.readouterr().out


def test_asset_missing_task_queued(monkeypatch, capsys):
    _asset_missing(monkeypatch, queued=True)
    task = FakeTask("export_pytest_x")
    check_for_asset_then_run_task("users/u/a", task, "pytest_x")
    assert task.started is False
    assert "already queued" in capsys.readouterr().out


def test_asset_missing_task_started(monkeypatch, capsys):
    _asset_missing(monkeypatch, queued=False)
    task = FakeTask("export_pytest_x")
    check_for_asset_then_run_task("users/u/a", task, "pytest_x")
    assert task.started is True
    assert "started" in capsys.readouterr().out


def test_pause_and_wait_rejects_nonpositive_wait():
    with pytest.raises(ValueError):
        pause_and_wait("x", wait_time=0)


def test_pause_and_wait_exits_when_no_tasks(monkeypatch):
    monkeypatch.setattr(ee.batch.Task, "list", lambda: [])
    pause_and_wait("pytest_x", wait_time=1)