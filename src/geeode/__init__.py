# Import all functions upon initialization

from .geeode import de_optim
from .geeode import sub_sample
from .geeode import ts_image_to_coll
from .geeode import apply_model
from .geeode import pause_and_wait
from .geeode import check_for_tasks
from .geeode import check_for_asset_then_run_task

# Define __all__ so “from geeode import *” imports functions explicitly
__all__ = [
    "de_optim",
    "sub_sample",
    "ts_image_to_coll",
    "apply_model",
    "pause_and_wait",
    "check_for_tasks",
    "check_for_asset_then_run_task",
]

