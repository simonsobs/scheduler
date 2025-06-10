"""A production-level implementation of the LAT policy

"""
import numpy as np
import yaml
import os.path as op
from dataclasses import dataclass, field, replace
from dataclasses_json import dataclass_json

import datetime as dt
from typing import List, Union, Optional, Dict, Any, Tuple
import jax.tree_util as tu
from functools import reduce, partial

from .. import config as cfg, core, source as src, rules as ru
from .. import commands as cmd, instrument as inst, utils as u
from ..thirdparty import SunAvoidance
from .stages import get_build_stage
from .stages.build_op import get_parking
from . import tel
from .tel import State, make_blocks
from ..instrument import CalTarget

logger = u.init_logger(__name__)

COROTATOR_DURATION = 1*u.minute
STIMULATOR_DURATION = 15*u.minute

def boresight_to_corotator(el, boresight):
    if el <= 90:
        el_ref = 60
    else:
        el_ref = 240
        el = 180 - el
    return np.round(el - el_ref + boresight, 4)
def corotator_to_boresight(el, corotator):
    if el <= 90:
        el_ref = 60
    else:
        el_ref = 240
        el = 180 - el
    return np.round(-1*(el - el_ref - corotator), 4)

@dataclass_json
@dataclass(frozen=True)
class State(tel.State):
    """
    State relevant to LAT operation scheduling. Inherits other fields:
    (`curr_time`, `az_now`, `el_now`, `az_speed_now`, `az_accel_now`)
    from the base State defined in `schedlib.commands`. And others from
    `tel.State`

    Parameters
    ----------
    corotator_now : int
        The current corotator state.
    """
    corotator_now: float = 0

    def get_boresight(self):
        return corotator_to_boresight(self.el_now, self.corotator_now)

@dataclass(frozen=True)
class StimulatorTarget:
    hour: int
    el_target: float
    az_target: float = 180
    duration: float = STIMULATOR_DURATION

class SchedMode(tel.SchedMode):
    """
    Enumerate different options for scheduling operations in LATPolicy.

    Attributes
    ----------
    Stimulator : str
        'stimulator'; Stimulator observations scheduled between block.t0 and block.t1
    """
    Stimulator = 'stimulator'


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
def preamble(open_shutter=False):
    cmd = tel.preamble()
    cmd += ["acu.clear_faults()"]
    if open_shutter:
        cmd += ["acu.stop_and_clear()",
                "acu.set_shutter(action='open')"
            ]
    return cmd

@cmd.operation(name='lat.wrap_up', duration=0)
def wrap_up(state, block, close_shutter=False):
    state, cmd = tel.wrap_up(state, block)
    if close_shutter:
        cmd += ["acu.set_shutter(action='close')"]
    return state, cmd

@cmd.operation(name='lat.ufm_relock', return_duration=True)
def ufm_relock(state, commands=None, relock_cadence=24*u.hour):
    return tel.ufm_relock(state, commands, relock_cadence)

# per block operation: block will be passed in as parameter
@cmd.operation(name='lat.det_setup', return_duration=True)
def det_setup(state, block, commands=None, apply_corotator_rot=True, iv_cadence=None, det_setup_duration=20*u.minute):
    return tel.det_setup(state, block, commands, apply_corotator_rot, iv_cadence, det_setup_duration)

@cmd.operation(name='lat.cmb_scan', return_duration=True)
def cmb_scan(state, block):
    return tel.cmb_scan(state, block)

@cmd.operation(name='lat.source_scan', return_duration=True)
def source_scan(state, block):
    return tel.source_scan(state, block)


@cmd.operation(name='lat.setup_corotator', return_duration=True)
def setup_corotator(state, block, apply_corotator_rot=True, cryo_stabilization_time=180*u.second, corotator_offset=0.):
    commands = []
    duration = 0

    if apply_corotator_rot and (
            state.corotator_now is None or state.corotator_now != block.corotator_angle
        ):

        assert np.abs(block.corotator_angle) <= 45, f"corotator angle {block.corotator_angle} not within [-45, 45] range"

        ## the ACU command is the one place where boresight=corotator
        ## everywhere else (particularly for math) corotator != boresight
        commands += [
            f"# Set corotator angle to {block.corotator_angle + corotator_offset} degrees",
            f"run.acu.set_boresight(target={block.corotator_angle + corotator_offset})",
        ]
        state = state.replace(corotator_now=block.corotator_angle)
        duration += COROTATOR_DURATION

        if cryo_stabilization_time > 0:
            commands += [f"time.sleep({cryo_stabilization_time})"]
            duration += cryo_stabilization_time

    return state, duration, commands


# passthrough any arguments, to be used in any sched-mode
@cmd.operation(name='lat.bias_step', return_duration=True)
def bias_step(state, block, bias_step_cadence=None):
    return tel.bias_step(state, block, bias_step_cadence)

@cmd.operation(name='lat.stimulator', duration=STIMULATOR_DURATION)
def stimulator(state):
    return state, [
        "run.stimulator.calibrate(continuous=False, elevation_check=True, boresight_check=False, temperature_check=False)"
    ]

@cmd.operation(name="move_to", return_duration=True)
def move_to(state, az, el, az_offset=0, el_offset=0, min_el=0, force=False):
    if not force and (state.az_now == az and state.el_now == el):
        return state, 0, []

    cmd = [
        f"run.acu.move_to(az={round(az + az_offset, 3)}, el={round(el + el_offset, 3)})",
    ]
    state = state.replace(az_now=az, el_now=el)

    return state, 0, cmd

# ----------------------------------------------------
#         setup LAT specific configs
# ----------------------------------------------------

def make_geometry(xi_offset=0., eta_offset=0.):
    logger.info(f"making geometry with xi offset={xi_offset}, eta offset={eta_offset}")
    return {
        "c1_ws0": {"center": [-0.3710+xi_offset, 0+eta_offset], "radius": 0.3,},
        "c1_ws1": {"center": [ 0.1815+xi_offset, 0.3211+eta_offset], "radius": 0.3,},
        "c1_ws2": {"center": [ 0.1815+xi_offset,-0.3211+eta_offset], "radius": 0.3,},
        "i1_ws0": {"center": [-1.9112+xi_offset,-0.9052+eta_offset], "radius": 0.3,},
        "i1_ws1": {"center": [-1.3584+xi_offset,-0.5704+eta_offset], "radius": 0.3,},
        "i1_ws2": {"center": [-1.3587+xi_offset,-1.2133+eta_offset], "radius": 0.3,},
        "i3_ws0": {"center": [ 1.1865+xi_offset,-0.8919+eta_offset], "radius": 0.3,},
        "i3_ws1": {"center": [ 1.7326+xi_offset,-0.5705+eta_offset], "radius": 0.3,},
        "i3_ws2": {"center": [ 1.7333+xi_offset,-1.2135+eta_offset], "radius": 0.3,},
        "i4_ws0": {"center": [ 1.1732+xi_offset, 0.9052+eta_offset], "radius": 0.3,},
        "i4_ws1": {"center": [ 1.7332+xi_offset, 1.2135+eta_offset], "radius": 0.3,},
        "i4_ws2": {"center": [ 1.7326+xi_offset, 0.5705+eta_offset], "radius": 0.3,},
        "i5_ws0": {"center": [-0.3655+xi_offset, 1.7833+eta_offset], "radius": 0.3,},
        "i5_ws1": {"center": [ 0.1879+xi_offset, 2.1045+eta_offset], "radius": 0.3,},
        "i5_ws2": {"center": [ 0.1867+xi_offset, 1.4620+eta_offset], "radius": 0.3,},
        "i6_ws0": {"center": [-1.9082+xi_offset, 0.8920+eta_offset], "radius": 0.3,},
        "i6_ws1": {"center": [-1.3577+xi_offset, 1.2133+eta_offset], "radius": 0.3,},
        "i6_ws2": {"center": [-1.3584+xi_offset, 0.5854+eta_offset], "radius": 0.3,},
    }

def make_cal_target(
    source: str,
    elevation: float,
    focus: str,
    corotator: float=None,
    allow_partial=False,
    drift=True,
    az_branch=None,
    az_speed=None,
    az_accel=None,
    source_direction=None,
) -> CalTarget:

    ## focus = 'all' will concatenate all of the tubes
    array_focus = {
        'c1' : 'c1_ws0,c1_ws1,c1_ws2',
        'i1' : 'i1_ws0,i1_ws1,i1_ws2',
        'i3' : 'i3_ws0,i3_ws1,i3_ws2',
        'i4' : 'i4_ws0,i4_ws1,i4_ws2',
        'i5' : 'i5_ws0,i5_ws1,i5_ws2',
        'i6' : 'i6_ws0,i6_ws1,i6_ws2',
    }
    elevation = float(elevation)
    if corotator is None:
        if elevation <= 90:
            boresight = 0
        else:
            boresight = 180
        corotator = boresight_to_corotator(elevation, boresight)
    boresight = corotator_to_boresight(elevation,float(corotator))

    focus = focus.lower()

    focus_str = None
    if focus == 'all':
        focus_str = ','.join( [v for k,v in array_focus.items()] )
    elif focus in array_focus.keys():
        focus_str = array_focus[focus]
    else:
        focus_str = focus

    sources = src.get_source_list()
    assert source in sources, f"source should be one of {sources.keys()}"

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
        source_direction=source_direction,
        from_table=False,
    )

def make_operations(
    az_speed,
    az_accel,
    az_motion_override,
    iv_cadence=4*u.hour,
    bias_step_cadence=0.5*u.hour,
    det_setup_duration=20*u.minute,
    apply_corotator_rot=True,
    cryo_stabilization_time=180*u.second,
    corotator_offset=0.,
    open_shutter=False,
    close_shutter=False,
    relock_cadence=24*u.hour
):

    pre_session_ops = [
        { 'name': 'lat.preamble'        , 'sched_mode': SchedMode.PreSession, 'open_shutter': open_shutter},
        { 'name': 'start_time'          , 'sched_mode': SchedMode.PreSession},
        { 'name': 'set_scan_params'     , 'sched_mode': SchedMode.PreSession, 'az_speed': az_speed, 'az_accel': az_accel, 'az_motion_override': az_motion_override},
    ]

    cal_ops = []
    cmb_ops = []

    if relock_cadence is not None:
        pre_session_ops += [
            { 'name': 'lat.ufm_relock'      , 'sched_mode': SchedMode.PreSession, 'relock_cadence': relock_cadence}
        ]
        cal_ops += [
            { 'name': 'lat.ufm_relock'      , 'sched_mode': SchedMode.PreCal, 'relock_cadence': relock_cadence}
        ]
        cmb_ops += [
            { 'name': 'lat.ufm_relock'      , 'sched_mode': SchedMode.PreObs, 'relock_cadence': relock_cadence}
        ]

    cal_ops += [
        { 'name': 'lat.setup_corotator' , 'sched_mode': SchedMode.PreCal, 'apply_corotator_rot': apply_corotator_rot,
        'cryo_stabilization_time': cryo_stabilization_time, 'corotator_offset': corotator_offset},
        { 'name': 'lat.det_setup'       , 'sched_mode': SchedMode.PreCal, 'apply_corotator_rot': apply_corotator_rot, 'iv_cadence':iv_cadence },
        { 'name': 'lat.source_scan'     , 'sched_mode': SchedMode.InCal, },
        { 'name': 'lat.bias_step'       , 'sched_mode': SchedMode.PostCal, 'bias_step_cadence': bias_step_cadence},
    ]
    cmb_ops += [
        { 'name': 'lat.setup_corotator' , 'sched_mode': SchedMode.PreObs, 'apply_corotator_rot': apply_corotator_rot,
        'cryo_stabilization_time': cryo_stabilization_time, 'corotator_offset': corotator_offset},
        { 'name': 'lat.det_setup'       , 'sched_mode': SchedMode.PreObs, 'apply_corotator_rot': apply_corotator_rot, 'iv_cadence':iv_cadence},
        { 'name': 'lat.bias_step'       , 'sched_mode': SchedMode.PreObs, 'bias_step_cadence': bias_step_cadence},
        { 'name': 'lat.cmb_scan'        , 'sched_mode': SchedMode.InObs, },
    ]

    post_session_ops = [
        { 'name': 'lat.wrap_up'   , 'sched_mode': SchedMode.PostSession, 'close_shutter': close_shutter},
    ]

    return pre_session_ops + cal_ops + cmb_ops + post_session_ops

def make_config(
    master_file,
    state_file,
    az_speed,
    az_accel,
    iv_cadence,
    bias_step_cadence,
    max_cmb_scan_duration,
    cal_targets,
    elevations_under_90=False,
    remove_targets=[],
    az_stow=None,
    el_stow=None,
    az_offset=0.,
    el_offset=0.,
    xi_offset=0.,
    eta_offset=0.,
    corotator_override=None,
    az_motion_override=False,
    az_branch_override=None,
    allow_partial_override=False,
    drift_override=True,
    **op_cfg
):
    blocks = make_blocks(master_file, 'lat-cmb')
    geometries = make_geometry(xi_offset, eta_offset)

    det_setup_duration = 20*u.minute

    operations = make_operations(
        az_speed, az_accel,
        az_motion_override,
        iv_cadence, bias_step_cadence,
        det_setup_duration,
        **op_cfg
    )

    sun_policy = {
        'min_angle': 21,
        'min_sun_time': 1801,
        'min_el': 0,
        'max_el': 180,
        'min_az': -180+10,
        'max_az': 360-10
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
        'az_range': [-180+10, 360-10],
    }

    el_range = {
        'el_range': [0, 180]
    }

    config = {
        'state_file': state_file,
        'blocks': blocks,
        'geometries': geometries,
        'rules': {
            'min-duration': {
                'min_duration': 60
            },
            'sun-avoidance': sun_policy,
            'az-range': az_range,
        },
        'operations': operations,
        'cal_targets': cal_targets,
        'scan_tag': None,
        'corotator_override': corotator_override,
        'elevations_under_90': elevations_under_90,
        'az_motion_override': az_motion_override,
        'remove_targets': remove_targets,
        'az_speed': az_speed,
        'az_accel': az_accel,
        'az_offset': az_offset,
        'el_offset': el_offset,
        'iv_cadence': iv_cadence,
        'bias_step_cadence': bias_step_cadence,
        'max_cmb_scan_duration': max_cmb_scan_duration,
        'az_branch_override': az_branch_override,
        'allow_partial_override': allow_partial_override,
        'drift_override': drift_override,
        'stages': {
            'build_op': {
                'plan_moves': {
                    'stow_position': stow_position,
                    'sun_policy': sun_policy,
                    'az_step': 0.5,
                    'az_limits': az_range['az_range'],
                    'el_limits': el_range['el_range'],
                }
            }
        }
    }
    return config

@dataclass
class LATPolicy(tel.TelPolicy):
    """a more realistic LAT policy.
    """
    corotator_override: Optional[float] = None
    elevations_under_90: Optional[bool] = False
    remove_targets: Optional[Tuple] = ()

    def apply_overrides(self, blocks):

        if self.elevations_under_90:
            def fix_block(b):
                if b.alt > 90:
                    return b.replace(alt=180-b.alt, az=b.az-180)
                return b
            blocks = core.seq_map( fix_block, blocks)

        if len(self.remove_targets) > 0:
            blocks = core.seq_filter_out(
                lambda b: b.name in self.remove_targets,
                blocks
            )

        if self.corotator_override is not None:
            blocks = core.seq_map(
                lambda b: b.replace(
                    corotator_angle=self.corotator_override
                ), blocks
            )
        else: ## run with co-rotator locked to elevation
            blocks = core.seq_map(
                lambda b: b.replace(
                    corotator_angle=boresight_to_corotator(b.alt, 0 if b.alt<=90 else 180)
                ), blocks
            )

        blocks = core.seq_map(
            lambda b: b.replace(
                boresight_angle=corotator_to_boresight(b.alt, b.corotator_angle)
            ), blocks
        )
        blocks = core.seq_map(
            lambda b: b.replace(
                az_speed= round( self.az_speed/np.cos(np.radians(b.alt)),2),
                az_accel=self.az_accel,
            ), blocks
        )

        return super().apply_overrides(blocks)

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

    @classmethod
    def from_defaults(
        cls,
        master_file,
        state_file=None,
        az_speed=0.8,
        az_accel=1.5,
        iv_cadence=4*u.hour,
        bias_step_cadence=0.5*u.hour,
        max_cmb_scan_duration=1*u.hour,
        cal_targets=None,
        az_stow=None,
        el_stow=None,
        az_offset=0.,
        el_offset=0.,
        xi_offset=0.,
        eta_offset=0.,
        elevations_under_90=False,
        corotator_override=None,
        az_motion_override=False,
        az_branch_override=None,
        allow_partial_override=False,
        drift_override=True,
        remove_targets=(),
        **op_cfg
    ):
        if cal_targets is None:
            cal_targets = []

        x = cls(**make_config(
            master_file,
            state_file,
            az_speed,
            az_accel,
            iv_cadence,
            bias_step_cadence,
            max_cmb_scan_duration,
            cal_targets,
            elevations_under_90=elevations_under_90,
            az_stow=az_stow,
            el_stow=el_stow,
            az_offset=az_offset,
            el_offset=el_offset,
            xi_offset=xi_offset,
            eta_offset=eta_offset,
            corotator_override=corotator_override,
            az_motion_override=az_motion_override,
            az_branch_override=az_branch_override,
            allow_partial_override=allow_partial_override,
            drift_override=drift_override,
            remove_targets=remove_targets,
            **op_cfg
        ))

        return x

    def make_source_scans(self, target, blocks, sun_rule):
        # digest array_query: it could be a fnmatch pattern matching the path
        # in the geometry dict, or it could be looked up from a predefined
        # wafer_set dict. Here we account for the latter case:
        # look up predefined query in wafer_set
        if target.array_query in self.wafer_sets:
            array_query = self.wafer_sets[target.array_query]
        else:
            array_query = target.array_query

        # build array geometry information based on the query
        array_info = inst.array_info_from_query(self.geometries, array_query)
        logger.debug(f"-> array_info: {array_info}")

        # apply MakeCESourceScan rule to transform known observing windows into
        # actual scan blocks
        rule = ru.MakeCESourceScan(
            array_info=array_info,
            el_bore=target.el_bore,
            drift=target.drift,
            boresight_rot=target.boresight_rot,
            allow_partial=target.allow_partial,
            az_branch=target.az_branch,
            source_direction=target.source_direction,
        )
        source_scans = rule(blocks['calibration'][target.source])

        # sun check again: previous sun check ensure source is not too
        # close to the sun, but our scan may still get close enough to
        # the sun, in which case we will trim it or delete it depending
        # on whether allow_partial is True
        if target.allow_partial:
            logger.info("-> allow_partial = True: trimming scan options by sun rule")
            min_dur_rule = ru.make_rule('min-duration', **self.rules['min-duration'])
            source_scans = min_dur_rule(sun_rule(source_scans))
        else:
            logger.info("-> allow_partial = False: filtering scan options by sun rule")
            source_scans = core.seq_filter(lambda b: b == sun_rule(b), source_scans)

        # flatten and sort
        source_scans = core.seq_sort(source_scans, flatten=True)

        return source_scans

    def init_seqs(self, cfile: str, t0: dt.datetime, t1: dt.datetime) -> core.BlocksTree:
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

        # get cal targets
        if cfile is not None:
            cal_targets = inst.parse_cal_targets_from_toast_lat(cfile)
            cal_targets[:] = [cal_target for cal_target in cal_targets if cal_target.t0 >= t0 and cal_target.t0 < t1]

            for i, cal_target in enumerate(cal_targets):
                if cal_target.el_bore > 90:
                    if self.elevations_under_90:
                        cal_targets[i] = replace(cal_targets[i], el_bore=180-cal_targets[i].el_bore)

                if self.corotator_override is None:
                    if cal_target.el_bore <= 90:
                        boresight = 0
                    else:
                        boresight = 180
                    corotator = boresight_to_corotator(cal_target.el_bore, boresight)
                    boresight = corotator_to_boresight(cal_target.el_bore, corotator)
                else:
                    boresight = corotator_to_boresight(cal_target.el_bore, float(self.corotator_override))
                cal_targets[i] = replace(cal_targets[i], boresight_rot=boresight)

                if self.az_branch_override is not None:
                    cal_targets[i] = replace(cal_targets[i], az_branch=self.az_branch_override)
                cal_targets[i] = replace(cal_targets[i], drift=self.drift_override)

            self.cal_targets += cal_targets

        # by default add calibration blocks specified in cal_targets if not already specified
        for cal_target in self.cal_targets:
            if isinstance(cal_target, CalTarget):
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

    def apply(self, blocks: core.BlocksTree) -> core.BlocksTree:
        """
        Applies a set of observing rules to the a tree of blocks such as modifying
        it with sun avoidance constraints and planning source scans for calibration.

        Parameters
        ----------
        blocks : BlocksTree
            The original blocks tree structure defining observing sequences and constraints.

        Returns
        -------
        BlocksTree
            New blocks tree after applying the specified observing rules.

        """
        # -----------------------------------------------------------------
        # step 1: preliminary sun avoidance
        #   - get rid of source observing windows too close to the sun
        #   - likely won't affect scan blocks because master schedule already
        #     takes care of this
        # -----------------------------------------------------------------
        if 'sun-avoidance' in self.rules:
            logger.info(f"applying sun avoidance rule: {self.rules['sun-avoidance']}")
            sun_rule = SunAvoidance(**self.rules['sun-avoidance'])
            blocks = sun_rule(blocks)
        else:
            logger.error("no sun avoidance rule specified!")
            raise ValueError("Sun rule is required!")

        # -----------------------------------------------------------------
        # step 2: plan calibration scans
        #   - refer to each target specified in cal_targets
        #   - same source can be observed multiple times with different
        #     array configurations (i.e. using array_query)
        # -----------------------------------------------------------------
        logger.info("planning calibration scans...")
        cal_blocks = []

        for target in self.cal_targets:
            logger.info(f"-> planning calibration scans for {target}...")

            if isinstance(target, StimulatorTarget):
                continue

            assert target.source in blocks['calibration'], f"source {target.source} not found in sequence"

            source_scans = self.make_source_scans(target, blocks, sun_rule)

            if len(source_scans) == 0:
                # try allow_partial=True if overriding and cal target is from table
                if (not target.allow_partial) and target.from_table and self.allow_partial_override==True:
                    logger.warning(f"-> no scan options available for {target.source} ({target.array_query}). trying allow_partial=True")
                    target = replace(target, allow_partial=True)
                    source_scans = self.make_source_scans(target, blocks, sun_rule)

                if len(source_scans) == 0:
                    logger.warning(f"-> no scan options available for {target.source} ({target.array_query})")
                    continue

            # which one can be added without conflicting with already planned calibration blocks?
            source_scans = core.seq_sort(
                core.seq_filter(lambda b: not any([b.overlaps(b_) for b_ in cal_blocks]), source_scans),
                flatten=True
            )

            if len(source_scans) == 0:
                logger.warning(f"-> all scan options overlap with already planned source scans...")
                continue

            logger.info(f"-> found {len(source_scans)} scan options for {target.source} ({target.array_query}): {u.pformat(source_scans)}, adding the first one...")

            # add the first scan option
            cal_block = source_scans[0]

            # update tag, speed, accel, etc
            cal_block = cal_block.replace(
                az_speed = target.az_speed if target.az_speed is not None else self.az_speed,
                az_accel = target.az_accel if target.az_accel is not None else self.az_accel,
                tag=f"{cal_block.tag},{target.tag}"
            )

            # set corotator correctly in blocks
            cal_block = cal_block.replace(
                corotator_angle=boresight_to_corotator(
                    cal_block.alt, cal_block.boresight_angle
                )
            )

            cal_blocks.append(cal_block)

        blocks['calibration'] = cal_blocks

        logger.info(f"-> after calibration policy: {u.pformat(blocks['calibration'])}")

        # check sun avoidance again
        blocks['calibration'] = core.seq_flatten(sun_rule(blocks['calibration']))

        # min duration rule
        if 'min-duration' in self.rules:
            logger.info(f"applying min duration rule: {self.rules['min-duration']}")
            rule = ru.make_rule('min-duration', **self.rules['min-duration'])
            blocks['baseline'] = rule(blocks['baseline'])

        # az range rule
        if 'az-range' in self.rules:
            logger.info(f"applying az range rule: {self.rules['az-range']}")
            az_range = ru.AzRange(**self.rules['az-range'])
            blocks['calibration'] = az_range(blocks['calibration'])

        # -----------------------------------------------------------------
        # step 4: tags
        # -----------------------------------------------------------------

        # add proper subtypes
        blocks['calibration'] = core.seq_map(
            lambda block: block.replace(subtype="cal") if block.name != 'stimulator' else block,
            blocks['calibration']
        )

        blocks['baseline']['cmb'] = core.seq_map(
            lambda block: block.replace(
                subtype="cmb",
                tag=f"{block.tag},{block.az:.0f}-{block.az+block.throw:.0f}"
            ),
            blocks['baseline']['cmb']
        )

        # add scan tag if supplied
        if self.scan_tag is not None:
            blocks['baseline'] = core.seq_map(
                lambda block: block.replace(tag=f"{block.tag},{self.scan_tag}"),
                blocks['baseline']
            )

        blocks = core.seq_sort(blocks['baseline']['cmb'] + blocks['calibration'], flatten=True)

        # add az and el offsets (not used in calculations)
        blocks = core.seq_map(
            lambda block: block.replace(
                az_offset=self.az_offset,
                alt_offset=self.el_offset,
            ),
            blocks
        )

        # -----------------------------------------------------------------
        # step 5: verify
        # -----------------------------------------------------------------

        # check if blocks are above min elevation
        alt_limits = self.stages['build_op']['plan_moves']['el_limits']
        for block in core.seq_flatten(blocks):
            if hasattr(block, 'alt'):
                assert block.alt >= alt_limits[0], (
                f"Block {block} is below the minimum elevation "
                f"of {alt_limits[0]} degrees."
                )

                assert block.alt < alt_limits[1], (
                f"Block {block} is above the maximum elevation "
                f"of {alt_limits[1]} degrees."
                )

        return blocks

    def init_state(self, t0: dt.datetime) -> State:
        """customize typical initial state for lat, if needed"""
        if self.state_file is not None:
            logger.info(f"using state from {self.state_file}")
            state = State.load(self.state_file)
            if state.curr_time != t0:
                logger.info(
                    f"Loaded state is at {state.curr_time}. Updating time to"
                    f" {t0}"
                )
                state = state.replace(curr_time = t0)
            return state

        return State(
            curr_time=t0,
            az_now=180,
            el_now=40,
            corotator_now=0,
        )

    def seq2cmd(
        self,
        seq,
        t0: dt.datetime,
        t1: dt.datetime,
        state: Optional[State] = None,
        return_state: bool = False,
    ) -> List[Any]:
        """
        Converts a sequence of blocks into a list of commands to be executed
        between two given times.

        This method is responsible for generating commands based on a given
        sequence of observing blocks, considering specific hardware settings and
        constraints. It also includes timing considerations, such as time to
        relock a UFM or boresight angles, and ensures proper settings for
        azimuth speed and acceleration. It is assumed that the provided sequence
        is sorted in time.

        Parameters
        ----------
        seq : core.Blocks
            A tree-like sequence of Blocks representing the observation schedule
        t0 : datetime.datetime
            The starting datetime for the command sequence.
        t1 : datetime.datetime
            The ending datetime for the command sequence
        state : Optional[State], optional
            The initial state of the observatory, by default None

        Returns
        -------
        list of Operation

        """
        if state is None:
            state = self.init_state(t0)

        # load building stage
        build_op = get_build_stage('build_op', {'policy_config': self, **self.stages.get('build_op', {})})

        # first resolve overlapping between cal and cmb
        cal_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'cal', seq))
        cmb_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'cmb', seq))
        stimulator_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'stimulator', seq))
        cal_blocks += stimulator_blocks
        seq = core.seq_sort(core.seq_merge(cmb_blocks, cal_blocks, flatten=True))

        # divide cmb blocks
        if self.max_cmb_scan_duration is not None:
            seq = core.seq_flatten(core.seq_map(lambda b: self.divide_blocks(b, dt.timedelta(seconds=self.max_cmb_scan_duration)) if b.subtype=='cmb' else b, seq))

        # compile operations
        cal_pre = [op for op in self.operations if op['sched_mode'] == SchedMode.PreCal]
        cal_in = [op for op in self.operations if op['sched_mode'] == SchedMode.InCal]
        cal_post = [op for op in self.operations if op['sched_mode'] == SchedMode.PostCal]
        cmb_pre = [op for op in self.operations if op['sched_mode'] == SchedMode.PreObs]
        cmb_in = [op for op in self.operations if op['sched_mode'] == SchedMode.InObs]
        cmb_post = [op for op in self.operations if op['sched_mode'] == SchedMode.PostObs]
        pre_sess = [op for op in self.operations if op['sched_mode'] == SchedMode.PreSession]
        pos_sess = [op for op in self.operations if op['sched_mode'] == SchedMode.PostSession]
        stimulator_in = [op for op in self.operations if op['sched_mode'] == SchedMode.Stimulator]

        def map_block(block):
            if block.subtype == 'cal':
                return {
                    'name': block.name,
                    'block': block,
                    'pre': cal_pre,
                    'in': cal_in,
                    'post': cal_post,
                    'priority': 0
                }
            elif block.subtype == 'cmb':
                return {
                    'name': block.name,
                    'block': block,
                    'pre': cmb_pre,
                    'in': cmb_in,
                    'post': cmb_post,
                    'priority': 1
                }
            elif block.subtype == 'stimulator':
                return {
                    'name': block.name,
                    'block': block,
                    'pre': [],
                    'in': stimulator_in,
                    'post': [],
                    'priority': 0
                }
            else:
                raise ValueError(f"unexpected block subtype: {block.subtype}")

        seq = [map_block(b) for b in seq]

        start_block = {
            'name': 'pre-session',
            'block': inst.StareBlock(name="pre-session", az=state.az_now, alt=state.el_now, az_offset=self.az_offset, alt_offset=self.el_offset,
                                     t0=t0, t1=t0+dt.timedelta(seconds=1)),
            'pre': [],
            'in': [],
            'post': pre_sess,  # scheduled after t0
            'priority': 0,
            'pinned': True  # remain unchanged during multi-pass
        }
        # move to stow position if specified, otherwise keep final position
        if len(pos_sess) > 0:
            # find an alt, az that is sun-safe for the entire duration of the schedule.
            if not self.stages['build_op']['plan_moves']['stow_position']:
                az_start = 180
                alt_start = 60
                # add a buffer to start and end to be safe
                t_start = t0 - dt.timedelta(seconds=300)
                t_end = t1 + dt.timedelta(seconds=300)
                az_stow, alt_stow, _, _ = get_parking(t_start, t_end, alt_start, self.stages['build_op']['plan_moves']['sun_policy'])
                logger.info(f"found sun safe stow position at ({az_stow}, {alt_stow})")
            else:
                az_stow = self.stages['build_op']['plan_moves']['stow_position']['az_stow']
                alt_stow = self.stages['build_op']['plan_moves']['stow_position']['el_stow']
        elif len(seq) > 0:
            az_stow = seq[-1]['block'].az
            alt_stow = seq[-1]['block'].alt
        else:
            az_stow = state.az_now
            alt_stow = state.el_now
        end_block = {
            'name': 'post-session',
            'block': inst.StareBlock(name="post-session", az=az_stow, alt=alt_stow, az_offset=self.az_offset, alt_offset=self.el_offset,
                                    t0=t1-dt.timedelta(seconds=1), t1=t1),
            'pre': pos_sess, # scheduled before t1
            'in': [],
            'post': [],
            'priority': 0,
            'pinned': True # remain unchanged during multi-pass
        }
        seq = [start_block] + seq + [end_block]

        ops, state = build_op.apply(seq, t0, t1, state)
        if return_state:
            return ops, state
        return ops

    def add_cal_target(self, *args, **kwargs):
        self.cal_targets.append(make_cal_target(*args, **kwargs))

    def add_stimulator_target(self, el_target, hour_utc=12, az_target=180, duration=STIMULATOR_DURATION, **kwargs):
        self.cal_targets.append(StimulatorTarget(hour=hour_utc, az_target=az_target, el_target=el_target, duration=duration))
