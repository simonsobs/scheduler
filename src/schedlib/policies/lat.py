"""A production-level implementation of the LAT policy

"""
import numpy as np
import yaml
import os.path as op
from functools import partial
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

import datetime as dt
from typing import List, Union, Optional, Dict, Any, Tuple
import jax.tree_util as tu
from functools import reduce

from . import tel
from .tel import State, CalTarget
from .. import config as cfg, core, source as src, rules as ru
from .. import commands as cmd, instrument as inst, utils as u
from ..thirdparty import SunAvoidance
from .stages import get_build_stage
from ..commands import SchedMode

logger = u.init_logger(__name__)


# ----------------------------------------------------
#                  Register operations
# ----------------------------------------------------
# Note: to avoid naming collisions. Use appropriate prefixes
# whenver necessary. For example, all lat specific
# operations should start with `lat`.
#
# Registered operations can be three kinds of functions:
#
# 1. for operations with static duration, it can be defined as a function
#    that returns a list of commands, with the static duration specified in
#    the decorator
# 2. for operations with dynamic duration, meaning the duration is determined
#    at runtime, it can be defined as a function that returns a tuple of
#    duration and commands; the decorator should be informed with the option
#    `return_duration=True`
# 3. for operations that depends and/or modifies the state, the operation
#    function should take the state as the first argument (no renaming allowed)
#    and return a new state before the rest of the return values
#
# For example the following are all valid definitions:
#  @cmd.operation(name='my-op', duration=10)
#  def my_op():
#      return ["do something"]
#
#  @cmd.operation(name='my-op', return_duration=True)
#  def my_op():
#      return 10, ["do something"]
#
#  @cmd.operation(name='my-op')
#  def my_op(state):
#      return state, ["do something"]
#
#  @cmd.operation(name='my-op', return_duration=True)
#  def my_op(state):
#      return state, 10, ["do something"]

@cmd.operation(name="lat.preamble", duration=0)
def preamble():
    return tel.preamble()

@cmd.operation(name='lat.ufm_relock', return_duration=True)
def ufm_relock(state, commands=None):
    return tel.ufm_relock(state)

@cmd.operation(name='lat.det_setup', return_duration=True)
def det_setup(state, block, commands=None, apply_boresight_rot=True, iv_cadence=None):
    return tel.det_setup(state, block, commands, apply_boresight_rot, iv_cadence)

@cmd.operation(name='lat.setup_boresight', return_duration=True)
def setup_boresight(state, block, apply_boresight_rot=True):
    commands = []
    duration = 0
    if apply_boresight_rot and (
            state.boresight_rot_now is None or state.boresight_rot_now != block.boresight_angle
        ):

        commands = [f"run.acu.set_boresight({block.boresight_angle})"]
        state = state.replace(boresight_rot_now=block.boresight_angle)
        duration = 1*u.minute

    return state, duration, commands

@cmd.operation(name='lat.bias_step', return_duration=True)
def bias_step(state, block, bias_step_cadence=None):
    return tel.bias_step(state, block, bias_step_cadence)

@cmd.operation(name='lat.cmb_scan', return_duration=True)
def cmb_scan(state, block):
    return tel.cmb_scan(state, block)

@cmd.operation(name="lat.source_scan", return_duration=True)
def source_scan(state, block):
    return tel.source_scan(state, block)

@cmd.operation(name="move_to", return_duration=True)
def move_to(state, az, el, force=False):
    if not force and (state.az_now == az and state.el_now == el):
        return state, 0, []

    cmd = [
        f"run.acu.move_to(az={round(az, 3)}, el={round(el, 3)})",
    ]
    state = state.replace(az_now=az, el_now=el)

    return state, 0, cmd


# ----------------------------------------------------
#         setup LAT specific configs
# ----------------------------------------------------

def make_geometry():
    # These are just the median of the wafers and an ~overestimated rad rn
    # To be updated later
    return {
        "i1_ws0": {
            "center": [1.3516076803207397, 0.5679303407669067],
            "radius": 0.03,
        },
        "i1_ws1": {
            "center": [1.363024353981018, 1.2206860780715942],
            "radius": 0.03,
        },
        "i1_ws2": {
            "center": [1.9164373874664307, 0.9008757472038269],
            "radius": 0.03,
        },
        "i6_ws0": {
            "center": [1.3571038246154785, -1.2071731090545654],
            "radius": 0.03,
        },
        "i6_ws1": {
            "center": [1.3628365993499756, -0.5654135942459106],
            "radius": 0.03,
        },
        "i6_ws2": {
            "center": [1.9065929651260376, -0.8826764822006226],
            "radius": 0.03,
        },
    }

def make_cal_target(
    source: str, 
    boresight: float, 
    elevation: float, 
    focus: str, 
    allow_partial=False,
    drift=True,
    az_branch=None,
    az_speed=None,
    az_accel=None,
) -> CalTarget:

    array_focus = {
        'all' : 'i1_ws0,i1_ws1,i1_ws2,i6_ws0,i6_ws1,i6_ws2'
    }

    boresight = float(boresight)
    elevation = float(elevation)
    focus = focus.lower()

    focus_str = None
    if int(boresight) not in array_focus:
        logger.warning(
            f"boresight not in {array_focus.keys()}, assuming {focus} is a wafer string"
        )
        focus_str = focus
    else:
        focus_str = array_focus[int(boresight)].get(focus, focus)

    assert source in src.SOURCES, f"source should be one of {src.SOURCES.keys()}"

    if az_branch is None:
        az_branch = 180.

    return CalTarget(
        source=source, 
        array_query=focus_str, 
        el_bore=elevation, 
        boresight_rot=boresight, 
        tag=focus_str,
        allow_partial=allow_partial,
        drift=drift,
        az_branch=az_branch,
        az_speed=az_speed,
        az_accel=az_accel,
    )

def make_blocks(master_file):
    return {
        'baseline': {
            'cmb': {
                'type': 'toast',
                'file': master_file
            }
        },
        'calibration': {
            'saturn': {
                'type' : 'source',
                'name' : 'saturn',
            },
            'jupiter': {
                'type' : 'source',
                'name' : 'jupiter',
            },
            'moon': {
                'type' : 'source',
                'name' : 'moon',
            },
            'uranus': {
                'type' : 'source',
                'name' : 'uranus',
            },
            'neptune': {
                'type' : 'source',
                'name' : 'neptune',
            },
            'mercury': {
                'type' : 'source',
                'name' : 'mercury',
            },
            'venus': {
                'type' : 'source',
                'name' : 'venus',
            },
            'mars': {
                'type' : 'source',
                'name' : 'mars',
            }
        },
    }

def make_operations(
    az_speed, az_accel, iv_cadence=4*u.hour, bias_step_cadence=0.5*u.hour,
    disable_hwp=False, apply_boresight_rot=True, home_at_end=False, run_relock=False
):

    pre_session_ops = [
        { 'name': 'lat.preamble'        , 'sched_mode': SchedMode.PreSession},
        { 'name': 'start_time'          , 'sched_mode': SchedMode.PreSession},
        { 'name': 'set_scan_params'     , 'sched_mode': SchedMode.PreSession, 'az_speed': az_speed, 'az_accel': az_accel, },
    ]
    if run_relock:
        pre_session_ops += [
            { 'name': 'lat.ufm_relock'      , 'sched_mode': SchedMode.PreSession, }
        ]
    cal_ops = [
        { 'name': 'lat.setup_boresight' , 'sched_mode': SchedMode.PreCal, 'apply_boresight_rot': apply_boresight_rot, },
        { 'name': 'lat.det_setup'       , 'sched_mode': SchedMode.PreCal, 'apply_boresight_rot': apply_boresight_rot, 'iv_cadence':iv_cadence },
        { 'name': 'lat.source_scan'     , 'sched_mode': SchedMode.InCal, },
        { 'name': 'lat.bias_step'       , 'sched_mode': SchedMode.PostCal, 'bias_step_cadence': bias_step_cadence},
    ]
    cmb_ops = [
        { 'name': 'lat.setup_boresight' , 'sched_mode': SchedMode.PreObs, 'apply_boresight_rot': apply_boresight_rot, },
        { 'name': 'lat.det_setup'       , 'sched_mode': SchedMode.PreObs, 'apply_boresight_rot': apply_boresight_rot, 'iv_cadence':iv_cadence},
        { 'name': 'lat.bias_step'       , 'sched_mode': SchedMode.PreObs, 'bias_step_cadence': bias_step_cadence},
        { 'name': 'lat.cmb_scan'        , 'sched_mode': SchedMode.InObs, },
    ]

    return pre_session_ops + cal_ops + cmb_ops

def make_config(
    master_file,
    az_speed,
    az_accel,
    iv_cadence,
    bias_step_cadence,
    max_cmb_scan_duration,
    cal_targets,
    az_stow=None,
    el_stow=None,
    boresight_override=None,
    **op_cfg
):
    blocks = make_blocks(master_file)
    geometries = make_geometry()
    operations = make_operations(
        az_speed, az_accel,
        iv_cadence, bias_step_cadence,
        **op_cfg
    )

    sun_policy = {
        'min_angle': 41,
        'min_sun_time': 1980,
        'min_el': 48,
    }

    if az_stow is None or el_stow is None:
        stow_position = {}
    else:
        stow_position = {
            'az_stow': az_stow,
            'el_stow': el_stow,
        }

    az_range = {
        'trim': False,
        'az_range': [-45, 405]
    }

    config = {
        'blocks': blocks,
        'geometries': geometries,
        'rules': {
            'min-duration': {
                'min_duration': 600
            },
            'sun-avoidance': sun_policy,
            'az-range': az_range,
        },
        'operations': operations,
        'cal_targets': cal_targets,
        'scan_tag': None,
        'boresight_override': boresight_override,
        'az_speed': az_speed,
        'az_accel': az_accel,
        'iv_cadence': iv_cadence,
        'bias_step_cadence': bias_step_cadence,
        'max_cmb_scan_duration': max_cmb_scan_duration,
        'stages': {
            'build_op': {
                'plan_moves': {
                    'stow_position': stow_position,
                    'sun_policy': sun_policy,
                    'az_step': 0.5,
                    'az_limits': az_range['az_range'],
                }
            }
        }
    }
    return config


@dataclass
class LATPolicy(tel.TelPolicy):
    """a more realistic LAT policy.

    Parameters
    ----------
    state_file : string
        optional path to the state file.

    state_file: Optional[str] = None
    """

    @classmethod
    def from_config(cls, config: Union[Dict[str, Any], str]):
        """
        Constructs a policy object from a YAML configuration file, a YAML string, or a dictionary.

        Parameters
        ----------
        config : Union[dict, str]
            The configuration to populate the policy object.

        Returns
        -------
        The constructed policy object.
        """
        if isinstance(config, str):
            loader = cfg.get_loader()
            if op.isfile(config):
                with open(config, "r") as f:
                    config = yaml.load(f.read(), Loader=loader)
            else:
                config = yaml.load(config, Loader=loader)
        return cls(**config)

    def init_seqs(self, t0: dt.datetime, t1: dt.datetime) -> core.BlocksTree:
        """
        Initialize the sequences for the scheduler to process.

        Parameters
        ----------
        t0 : datetime.datetime
            The start time of the sequences.
        t1 : datetime.datetime
            The end time of the sequences.

        Returns
        -------
        BlocksTree (nested dict / list of blocks)
            The initialized sequences
        """

        # construct seqs by traversing the blocks definition dict
        blocks = tu.tree_map(
            partial(self.construct_seq, t0=t0, t1=t1),
            self.blocks,
            is_leaf=lambda x: isinstance(x, dict) and 'type' in x
        )

        # by default add calibration blocks specified in cal_targets if not already specified
        for cal_target in self.cal_targets:
            if isinstance(cal_target, tel.CalTarget):
                source = cal_target.source
                if source not in blocks['calibration']:
                    blocks['calibration'][source] = src.source_gen_seq(source, t0, t1)

        # trim to given time range
        blocks = core.seq_trim(blocks, t0, t1)

        # ok to drop Nones
        blocks = tu.tree_map(
            lambda x: [x_ for x_ in x if x_ is not None],
            blocks,
            is_leaf=lambda x: isinstance(x, list)
        )

        return blocks

    @classmethod
    def from_defaults(cls, master_file, az_speed=0.8, az_accel=1.5,
        iv_cadence=4*u.hour, bias_step_cadence=0.5*u.hour,
        max_cmb_scan_duration=1*u.hour, cal_targets=None,
        az_stow=None, el_stow=None, boresight_override=None,
        state_file=None, **op_cfg
    ):
        if cal_targets is None:
            cal_targets = []

        x = cls(**make_config(
            master_file, az_speed, az_accel, iv_cadence,
            bias_step_cadence, max_cmb_scan_duration,
            cal_targets, az_stow, el_stow, boresight_override,
            **op_cfg
        ))
        x.state_file=state_file
        return x

    def add_cal_target(self, *args, **kwargs):
        self.cal_targets.append(make_cal_target(*args, **kwargs))

    def init_state(self, t0: dt.datetime) -> State:
        """customize typical initial state for lat, if needed"""
        if self.state_file is not None:
            logger.info(f"using state from {self.state_file}")
            state = State.load(self.state_file)
            if state.curr_time < t0:
                logger.info(
                    f"Loaded state is at {state.curr_time}. Updating time to"
                    f" {t0}"
                )
                state = state.replace(curr_time = t0)
            return state

        return State(
            curr_time=t0,
            az_now=180,
            el_now=48,
            boresight_rot_now=None,
        )