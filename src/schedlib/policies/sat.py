"""A production-level implementation of the SAT policy

"""
import numpy as np
import yaml
import os.path as op
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

import datetime as dt
from typing import List, Union, Optional, Dict, Any, Tuple
import jax.tree_util as tu
from functools import reduce

from .. import config as cfg, core, source as src, rules as ru
from .. import commands as cmd, instrument as inst, utils as u
from ..thirdparty import SunAvoidance
from .stages import get_build_stage

logger = u.init_logger(__name__)

@dataclass_json
@dataclass(frozen=True)
class State(cmd.State):
    """
    State relevant to SAT operation scheduling. Inherits other fields:
    (`curr_time`, `az_now`, `el_now`, `az_speed_now`, `az_accel_now`)
    from the base State defined in `schedlib.commands`.

    Parameters
    ----------
    boresight_rot_now : int
        The current boresight rotation state.
    hwp_spinning : bool
        Whether the high-precision measurement wheel is spinning or not.
    last_ufm_relock : Optional[datetime.datetime]
        The last time the UFM was relocked, or None if it has not been relocked.
    last_bias_step : Optional[datetime.datetime]
        The last time a bias step was performed, or None if no bias step has been performed.
    is_det_setup : bool
        Whether the detectors have been set up or not.
    """
    boresight_rot_now: float = 0
    hwp_spinning: bool = False
    last_ufm_relock: Optional[dt.datetime] = None
    last_bias_step: Optional[dt.datetime] = None
    last_bias_step_boresight: Optional[float] = None
    last_bias_step_elevation: Optional[float] = None
    last_iv: Optional[dt.datetime] = None
    last_iv_boresight: Optional[float] = None
    last_iv_elevation: Optional[float] = None
    # relock sets to false, tracks if detectors are biased at all
    is_det_setup: bool = False


@dataclass(frozen=True)
class CalTarget:
    source: str
    array_query: str
    el_bore: float
    tag: str
    boresight_rot: float = 0
    allow_partial: bool = False
    drift: bool = True
    az_branch: Optional[float] = None
    az_speed: Optional[float]= None
    az_accel: Optional[float] = None

@dataclass(frozen=True)
class WiregridTarget:
    hour: int
    el_target: float
    az_target: float = 180
    duration: float = 15*u.minute

class SchedMode:
    """
    Enumerate different options for scheduling operations in SATPolicy.

    Attributes
    ----------
    PreCal : str
        'pre_cal'; Operations scheduled before block.t0 for calibration.
    PreObs : str
        'pre_obs'; Observations scheduled before block.t0 for observation.
    InCal : str
        'in_cal'; Calibration operations scheduled between block.t0 and block.t1.
    InObs : str
        'in_obs'; Observation operations scheduled between block.t0 and block.t1.
    PostCal : str
        'post_cal'; Calibration operations scheduled after block.t1.
    PostObs : str
        'post_obs'; Observations operations scheduled after block.t1.
    PreSession : str
        'pre_session'; Represents the start of a session, scheduled from the beginning of the requested t0.
    PostSession : str
        'post_session'; Indicates the end of a session, scheduled after the last operation.

    """
    PreCal = 'pre_cal'
    PreObs = 'pre_obs'
    InCal = 'in_cal'
    InObs = 'in_obs'
    PostCal = 'post_cal'
    PostObs = 'post_obs'
    PreSession = 'pre_session'
    PostSession = 'post_session'
    Wiregrid = 'wiregrid'

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
        0 : {
            'left' : 'ws3,ws2',
            'middle' : 'ws0,ws1,ws4',
            'right' : 'ws5,ws6',
            'bottom': 'ws1,ws2,ws6',
            'all' : 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
        },
        45 : {
            'left' : 'ws3,ws4',
            'middle' : 'ws2,ws0,ws5',
            'right' : 'ws1,ws6',
            'bottom': 'ws1,ws2,ws3',
            'all' : 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
        },
        -45 : {
            'left' : 'ws1,ws2',
            'middle' : 'ws6,ws0,ws3',
            'right' : 'ws4,ws5',
            'bottom': 'ws1,ws6,ws5',
            'all' : 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
        },
    }

    boresight = float(boresight)
    elevation = float(elevation)
    focus = focus.lower()

    focus_str = None
    if int(boresight) not in array_focus:
        logger.warning(
            f"boresight not in {array_focus.keys()}, assuming {focus} is a wafer string"
        )
        focus_str = focus ##
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

# ----------------------------------------------------
#                  Register operations
# ----------------------------------------------------
# Note: to avoid naming collisions. Use appropriate prefixes
# whenver necessary. For example, all satp1 specific
# operations should start with `satp1`.
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

@cmd.operation(name="sat.preamble", duration=0)
def preamble():
    return [
    "from nextline import disable_trace",
    "import time",
    "",
    "with disable_trace():",
    "    import numpy as np",
    "    import sorunlib as run",
    "    from ocs.ocs_client import OCSClient",
    "    run.initialize()",
    "",
    "acu = run.CLIENTS['acu']",
    "pysmurfs = run.CLIENTS['smurf']",
    "sup = OCSClient('hwp-supervisor')",
    "",
    ]

@cmd.operation(name='sat.ufm_relock', return_duration=True)
def ufm_relock(state, commands=None):
    if state.last_ufm_relock is None:
        doit = True
    elif (state.curr_time - state.last_ufm_relock).total_seconds() > 12*u.hour:
        doit = True
    else:
        doit = False

    if doit:
        if commands is None:
            commands = [
                "############# Daily Relock",
                "run.smurf.zero_biases()",
                "",
                "time.sleep(120)",
                "run.smurf.take_noise(concurrent=True, tag='res_check')",
                "run.smurf.uxm_relock(concurrent=True)",
                "",
            ]
        state = state.replace(
            last_ufm_relock=state.curr_time,
            is_det_setup=False,
        )
        return state, 15*u.minute, commands
    else:
        return state, 0, ["# no ufm relock needed at this time"]

@cmd.operation(name='sat.hwp_spin_up', return_duration=True)
def hwp_spin_up(state, disable_hwp=False, forward=True):
    if disable_hwp:
        return state, 0, ["# hwp disabled"]
    elif state.hwp_spinning:
        return state, 0, ["# hwp already spinning"]
    else:
        state = state.replace(hwp_spinning=True)
        if forward:
            freq = 2
        else:
            freq = -2
        return state, 20*u.minute, [
            "sup.enable_driver_board()",
            f"run.hwp.set_freq(freq={freq})",
        ]


@cmd.operation(name='sat.hwp_spin_down', return_duration=True)
def hwp_spin_down(state, disable_hwp=False):
    if disable_hwp:
        return state, 0, ["# hwp disabled"]
    elif not state.hwp_spinning:
        return state, 0, ["# hwp already stopped"]
    else:
        state = state.replace(hwp_spinning=False)
        return state, cmd.HWP_SPIN_DOWN, [
            "run.hwp.stop(active=True)",
            "sup.disable_driver_board()",
        ]

# per block operation: block will be passed in as parameter
@cmd.operation(name='sat.det_setup', return_duration=True)
def det_setup(state, block, commands=None, apply_boresight_rot=True, iv_cadence=None):
    # when should det setup be done?
    # -> should always be done if the block is a cal block
    # -> should always be done if elevation has changed
    # -> should always be done if det setup has not been done yet
    # -> should be done at a regular interval if iv_cadence is not None
    # -> should always be done if boresight rotation has changed
    doit = (block.subtype == 'cal')
    doit = doit or (not state.is_det_setup) or (state.last_iv is None)
    if not doit:
        if state.last_iv_elevation is not None:
            doit = doit or (
                not np.isclose(state.last_iv_elevation, block.alt, atol=1)
            )
        if apply_boresight_rot and state.last_iv_boresight is not None:
            doit = doit or (
                not np.isclose(
                    state.last_iv_boresight,
                    block.boresight_angle,
                    atol=1
                )
            )
        if iv_cadence is not None:
            time_since_last = (state.curr_time - state.last_iv).total_seconds()
            doit = doit or (time_since_last > iv_cadence)

    if doit:
        if commands is None:
            commands = [
                "",
                "################### Detector Setup######################",
                "with disable_trace():",
                "    run.initialize()",
                "run.smurf.take_bgmap(concurrent=True)",
                "run.smurf.take_noise(concurrent=True, tag='res_check')",
                "run.smurf.iv_curve(concurrent=True, ",
                "    iv_kwargs={'run_serially': False, 'cool_wait': 60*5})",
                "run.smurf.bias_dets(concurrent=True)",
                "time.sleep(180)",
                "run.smurf.bias_step(concurrent=True)",
                "run.smurf.take_noise(concurrent=True, tag='bias_check')",
                "#################### Detector Setup Over ####################",
                "",
            ]
        state = state.replace(
            is_det_setup=True,
            last_iv = state.curr_time,
            last_bias_step=state.curr_time,
            last_iv_elevation = block.alt,
            last_iv_boresight = block.boresight_angle,
            last_bias_step_elevation = block.alt,
            last_bias_step_boresight = block.boresight_angle,
        )
        return state, 12*u.minute, commands
    else:
        return state, 0, []

@cmd.operation(name='sat.cmb_scan', return_duration=True)
def cmb_scan(state, block):
    if (
        block.az_speed != state.az_speed_now or 
        block.az_accel != state.az_accel_now
    ):
        commands = [
            f"run.acu.set_scan_params({block.az_speed}, {block.az_accel})"
        ]
        state = state.replace(
            az_speed_now=block.az_speed, 
            az_accel_now=block.az_accel
        )
    else:
        commands = []

    commands.extend([
        "run.seq.scan(",
        f"    description='{block.name}',",
        f"    stop_time='{block.t1.isoformat()}',",
        f"    width={round(block.throw,3)}, az_drift=0,",
        f"    subtype='cmb', tag='{block.tag}',",
        f"    min_duration=600,",
        ")",
    ])
    return state, (block.t1 - state.curr_time).total_seconds(), commands

@cmd.operation(name='sat.source_scan', return_duration=True)
def source_scan(state, block):
    block = block.trim_left_to(state.curr_time)
    if block is None:
        return state, 0, ["# too late, don't scan"]
    if (
        block.az_speed != state.az_speed_now or 
        block.az_accel != state.az_accel_now
    ):
        commands = [
            f"run.acu.set_scan_params({block.az_speed}, {block.az_accel})"
        ]
        state = state.replace(
            az_speed_now=block.az_speed, 
            az_accel_now=block.az_accel
        )
    else:
        commands = []
    
    state = state.replace(az_now=block.az, el_now=block.alt)
    commands.extend([
        f"run.acu.move_to_target(az={round(block.az,3)}, el={round(block.alt,3)},",
        f"    start_time='{block.t0.isoformat()}',",
        f"    stop_time='{block.t1.isoformat()}',",
        f"    drift={round(block.az_drift,5)})",
        "",
        f"print('Waiting until {block.t0} to start scan')",
        f"run.wait_until('{block.t0.isoformat()}')",
        "",
        "run.seq.scan(",
        f"    description='{block.name}', ",
        f"    stop_time='{block.t1.isoformat()}', ",
        f"    width={round(block.throw,3)}, ",
        f"    az_drift={round(block.az_drift,5)}, ",
        f"    subtype='{block.subtype}',",
        f"    tag='{block.tag}',",
        ")",
    ])
    return state, block.duration.total_seconds(), commands

@cmd.operation(name='sat.setup_boresight', return_duration=True)
def setup_boresight(state, block, apply_boresight_rot=True):
    commands = []
    duration = 0

    if apply_boresight_rot and (
            state.boresight_rot_now is None or state.boresight_rot_now != block.boresight_angle
        ):
        if state.hwp_spinning:
            state = state.replace(hwp_spinning=False)
            duration += HWP_SPIN_DOWN
            commands += [
                "run.hwp.stop(active=True)",
                "sup.disable_driver_board()",
            ]

        assert not state.hwp_spinning
        commands += [f"run.acu.set_boresight({block.boresight_angle})"]
        state = state.replace(boresight_rot_now=block.boresight_angle)
        duration += 1*u.minute

    return state, duration, commands

# passthrough any arguments, to be used in any sched-mode
@cmd.operation(name='sat.bias_step', return_duration=True)
def bias_step(state, block, bias_step_cadence=None):
    # -> should be done at a regular interval if bias_step_cadence is not None
    doit = state.last_bias_step is None
    if not doit:
        if state.last_bias_step_elevation is not None:
            doit = doit or (
                not np.isclose(
                    state.last_bias_step_elevation,
                    block.alt,
                    atol=1
                )
            )
        if state.last_bias_step_boresight is not None:
            doit = doit or (
                not np.isclose(
                    state.last_bias_step_boresight,
                    block.boresight_angle,
                    atol=1
                )
            )
        if bias_step_cadence is not None:
            time_since = (state.curr_time - state.last_bias_step).total_seconds()
            doit = doit or (time_since >= bias_step_cadence)

    if doit :
        state = state.replace(
            last_bias_step=state.curr_time,
            last_bias_step_elevation = block.alt,
            last_bias_step_boresight = block.boresight_angle,
        )
        return state, 60, [ "run.smurf.bias_step(concurrent=True)", ]
    else:
        return state, 0, []

@cmd.operation(name='sat.wrap_up', duration=1)
def wrap_up(state, az_stow, el_stow):
    state = state.replace(az_now=az_stow, el_now=el_stow)
    return state, [
        # "# go home",
        # f"run.acu.move_to(az={az_stow}, el={el_stow})",
        "time.sleep(1)"
    ]

@cmd.operation(name='sat.wiregrid', duration=15*u.minute)
def wiregrid(state):
    return state, [
        "run.wiregrid.calibrate(continuous=False, elevation_check=True, boresight_check=False, temperature_check=False)"
    ]

@dataclass
class SATPolicy:
    """a more realistic SAT policy.

    Parameters
    ----------
    blocks : dict
        a dict of blocks, with keys 'baseline' and 'calibration'
    rules : dict
        a dict of rules, specifies rule cfgs for e.g., 'sun-avoidance', 'az-range', 'min-duration'
    geometries : dict
        a dict of geometries, with the leave node being dict with keys 'center' and 'radius'
    cal_targets : list[CalTarget]
        a list of calibration target each described by CalTarget object
    cal_policy : str
        calibration policy: default to round-robin
    scan_tag : str
        a tag to be added to all scans
    az_speed : float
        the az speed in deg / s
    az_accel : float
        the az acceleration in deg / s^2
    wafer_sets : dict[str, str]
        a dict of wafer sets definitions
    operations : List[Dict[str, Any]]
        an orderred list of operation configurations
    """
    blocks: Dict[str, Any] = field(default_factory=dict)
    rules: Dict[str, core.Rule] = field(default_factory=dict)
    geometries: List[Dict[str, Any]] = field(default_factory=list)
    cal_targets: List[CalTarget] = field(default_factory=list)
    scan_tag: Optional[str] = None
    boresight_override: Optional[float] = None
    az_speed: float = 1. # deg / s
    az_accel: float = 2. # deg / s^2
    iv_cadence : float = 4 * u.hour
    bias_step_cadence : float = 0.5 * u.hour
    min_hwp_el : float = 48 # deg
    max_cmb_scan_duration : float = 1 * u.hour
    allow_az_maneuver: bool = True
    wafer_sets: Dict[str, Any] = field(default_factory=dict)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    stages: Dict[str, Any] = field(default_factory=dict)

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
        def construct_seq(loader_cfg):
            if loader_cfg['type'] == 'source':
                return src.source_gen_seq(loader_cfg['name'], t0, t1)
            elif loader_cfg['type'] == 'toast':
                blocks = inst.parse_sequence_from_toast(loader_cfg['file'])
                if self.boresight_override is not None:
                    blocks = core.seq_map(
                        lambda b: b.replace(
                            boresight_angle=self.boresight_override
                        ), blocks
                    )
                return blocks
            else:
                raise ValueError(f"unknown sequence type: {loader_cfg['type']}")

        # construct seqs by traversing the blocks definition dict
        blocks = tu.tree_map(construct_seq, self.blocks,
                             is_leaf=lambda x: isinstance(x, dict) and 'type' in x)

        # by default add calibration blocks specified in cal_targets if not already specified
        for cal_target in self.cal_targets:
            if isinstance(cal_target, CalTarget):
                source = cal_target.source
                if source not in blocks['calibration']:
                    blocks['calibration'][source] = src.source_gen_seq(source, t0, t1)
            elif isinstance(cal_target, WiregridTarget):
                wiregrid_candidates = []
                current_date = t0.date()
                end_date = t1.date()

                while current_date <= end_date:
                    candidate_time = dt.datetime.combine(current_date, dt.time(cal_target.hour, 0), tzinfo=dt.timezone.utc)
                    if t0 <= candidate_time <= t1:
                        wiregrid_candidates.append(
                            inst.StareBlock(
                                name='wiregrid',
                                t0=candidate_time,
                                t1=candidate_time + dt.timedelta(seconds=cal_target.duration),
                                az=cal_target.az_target,
                                alt=cal_target.el_target,
                                subtype='wiregrid'
                            )
                        )
                    current_date += dt.timedelta(days=1)
                blocks['calibration']['wiregrid'] = wiregrid_candidates

        # update az speed in scan blocks
        blocks = core.seq_map_when(
            lambda b: isinstance(b, inst.ScanBlock),
            lambda b: b.replace(az_speed=self.az_speed,az_accel=self.az_accel),
            blocks
        )

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

            if isinstance(target, WiregridTarget):
                logger.info(f"-> planning wiregrid scans for {target}...")
                cal_blocks += core.seq_map(lambda b: b.replace(subtype='wiregrid'), 
                                           blocks['calibration']['wiregrid'])
                continue

            assert target.source in blocks['calibration'], f"source {target.source} not found in sequence"

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
            lambda block: block.replace(subtype="cal") if block.name != 'wiregrid' else block,
            blocks['calibration']
        )

        blocks['baseline']['cmb'] = core.seq_map(
            lambda block: block.replace(
                subtype="cmb",
                tag=f"{block.az:.0f}-{block.az+block.throw:.0f}"
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

        return blocks

    def init_state(self, t0: dt.datetime) -> State:
        """
        Initializes the observatory state with some reasonable guess.
        In practice it should ideally be replaced with actual data
        from the observatory controller.

        Parameters
        ----------
        t0 : float
            The initial time for the state, typically representing the current time in a specific format.

        Returns
        -------
        State
        """
        return State(
            curr_time=t0,
            az_now=180,
            el_now=48,
            boresight_rot_now=0,
            hwp_spinning=False,
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
        build_op = get_build_stage('build_op', **{'policy_config': self, **self.stages.get('build_op', {})})

        # first resolve overlapping between cal and cmb
        cal_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'cal', seq))
        cmb_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'cmb', seq))
        wiregrid_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'wiregrid', seq))
        cal_blocks += wiregrid_blocks
        seq = core.seq_sort(core.seq_merge(cmb_blocks, cal_blocks, flatten=True))

        # compile operations
        cal_pre = [op for op in self.operations if op['sched_mode'] == SchedMode.PreCal]
        cal_in = [op for op in self.operations if op['sched_mode'] == SchedMode.InCal]
        cal_post = [op for op in self.operations if op['sched_mode'] == SchedMode.PostCal]
        cmb_pre = [op for op in self.operations if op['sched_mode'] == SchedMode.PreObs]
        cmb_in = [op for op in self.operations if op['sched_mode'] == SchedMode.InObs]
        cmb_post = [op for op in self.operations if op['sched_mode'] == SchedMode.PostObs]
        pre_sess = [op for op in self.operations if op['sched_mode'] == SchedMode.PreSession]
        pos_sess = [op for op in self.operations if op['sched_mode'] == SchedMode.PostSession]
        wiregrid_in = [op for op in self.operations if op['sched_mode'] == SchedMode.Wiregrid]

        def map_block(block):
            if block.subtype == 'cal':
                return {
                    'name': block.name,
                    'block': block,
                    'pre': cal_pre,
                    'in': cal_in,
                    'post': cal_post,
                    'priority': 3
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
            elif block.subtype == 'wiregrid':
                return {
                    'name': block.name,
                    'block': block,
                    'pre': [],
                    'in': wiregrid_in,
                    'post': [],
                    'priority': 2
                }
            else:
                raise ValueError(f"unexpected block subtype: {block.subtype}")

        seq = [map_block(b) for b in seq]
        start_block = {
            'name': 'pre-session',
            'block': inst.StareBlock(name="pre-session", az=state.az_now, alt=state.el_now, t0=t0, t1=t0+dt.timedelta(seconds=1)),
            'pre': [],
            'in': [],
            'post': pre_sess,  # scheduled after t0
            'priority': 3,
            'pinned': True  # remain unchanged during multi-pass
        }
        end_block = {
            'name': 'post-session',
            'block': inst.StareBlock(name="post-session", az=180, alt=50, t0=t1-dt.timedelta(seconds=1), t1=t1),
            'pre': pos_sess, # scheduled before t1
            'in': [],
            'post': [],
            'priority': 3,
            'pinned': True # remain unchanged during multi-pass
        }
        seq = [start_block] + seq + [end_block]

        ops, state = build_op.apply(seq, t0, t1, state)
        if return_state:
            return ops, state
        return ops

    def cmd2txt(self, irs, t0, t1, state=None):
        """
        Convert a sequence of operation blocks into a text representation.

        Parameters
        ----------
        irs : list of IR
            A sequence of operation blocks.

        Returns
        -------
        str
            A text representation of the sequence of operation blocks.

        """
        if state is None:
            state = self.init_state(t0)
        build_sched = get_build_stage('build_sched', {'policy_config': self, **self.stages.get('build_sched', {})})
        commands = build_sched.apply(irs, t0, t1, state)
        return '\n'.join(commands)

    def build_schedule(self, t0: dt.datetime, t1: dt.datetime, state: State = None):
        """
        Run entire scheduling process to build a schedule for a given time range.

        Parameters
        ----------
        t0 : datetime.datetime
            The start time of the schedule.
        t1 : datetime.datetime
            The end time of the schedule.
        state : Optional[State]
            The initial state of the observatory. If not provided, a default
            state will be initialized.

        Returns
        -------
        schedule as a text

        """
        # initialize sequences
        seqs = self.init_seqs(t0, t1)

        # apply observing rules
        seqs = self.apply(seqs)

        # initialize state
        state = state or self.init_state(t0)

        # plan operation seq
        ir = self.seq2cmd(seqs, t0, t1, state)

        # construct schedule str
        schedule = self.cmd2txt(ir, t0, t1, state)

        return schedule

    def add_cal_target(self, *args, **kwargs):
        self.cal_targets.append(make_cal_target(*args, **kwargs))

    def add_wiregrid_target(self, el_target, hour_utc=12, az_target=180, duration=15*u.minute, **kwargs):
        self.cal_targets.append(WiregridTarget(hour=hour_utc, az_target=az_target, el_target=el_target, duration=duration))

# ------------------------
# utilities
# ------------------------

def round_robin(seqs_q, seqs_v=None, sun_avoidance=None, overlap_allowance=60*u.second):
    """
    Perform a round robin scheduling over sequences of time blocks, yielding non-overlapping blocks.

    This function goes through sequences of "query" time blocks (`seqs_q`) in a round robin fashion, checking for overlap
    between the blocks. An optional sequence of "value" time blocks (`seqs_v`) can be provided, which will be returned
    instead of the query blocks. The use case for having `seqs_v` different from `seqs_q` is that `seqs_q` can represent
    buffered time blocks used for determining overlap conditions, while `seqs_v`, representing the actual unbuffered time
    blocks, gets returned.

    Parameters
    ----------
    seqs_q : list of lists
        The query sequences. Each sub-list contains time blocks that are checked for overlap.
    seqs_v : list of lists, optional
        The value sequences. Each sub-list contains time blocks that are returned when their corresponding `seqs_q` block
        doesn't overlap with existing blocks.
    sun_avoidance : function / rule, optional
        If provided, a block is scheduled only if it satisfies this condition, this means the block is unchanged after
        the rule is applied.
    overlap_allowance: int
        minimum overlap to be considered in seconds, larger overlap will be rejected.

    Yields
    ------
    block
        Blocks from `seqs_v` that don't overlap with previously yielded blocks, as per the conditions defined.

    Notes
    -----
    This generator function exhaustively attempts to yield all non-overlapping time blocks from the provided sequences
    in a round robin order. The scheduling respects the order of sequences and the order of blocks within each sequence.
    It supports an optional sun avoidance condition to filter out undesirable time blocks based on external criteria
    (for example, blocks that are in direct sunlight).

    Examples
    --------
    >>> seqs_q = [[block1, block2], [block3]]
    >>> list(round_robin(seqs_q))
    [block1, block3, block2]

    """
    if seqs_v is None:
        seqs_v = seqs_q
    assert len(seqs_q) == len(seqs_v)

    n_seq = len(seqs_q)
    seq_i = 0
    block_i = [0] * n_seq

    merged = []
    while True:
        # return if we have exhausted all scans in all seqs
        if all([block_i[i] >= len(seqs_q[i]) for i in range(n_seq)]):
            return

        # cycle through seq -> add the latest non-overlaping block -> continue to next seq
        # skip if we have exhaused all scans in a sequence
        if block_i[seq_i] >= len(seqs_q[seq_i]):
            seq_i = (seq_i + 1) % n_seq
            continue

        seq_q = seqs_q[seq_i]
        seq_v = seqs_v[seq_i]
        block_q = seq_q[block_i[seq_i]]
        block_v = seq_v[block_i[seq_i]]

        # can we schedule this block?
        #  yes if:
        #  - it doesn't overlap with existing blocks
        #  - it satisfies sun avoidance condition if specified
        overlap_ok = not core.seq_has_overlap_with_block(merged, block_q, allowance=overlap_allowance)
        if not overlap_ok:
            logger.info(f"-> Block {block_v} overlaps with existing block, skipping")

        if sun_avoidance is not None:
            sun_ok = block_q == sun_avoidance(block_q)
            if not sun_ok:
                logger.info(f"-> Block {block_v} fails sun check, skipping")

        ok = overlap_ok * sun_ok
        if ok:
            # schedule and move on to next seq
            yield block_v
            merged += [block_q]
            seq_i = (seq_i + 1) % n_seq

        block_i[seq_i] += 1

def simplify_hwp(op_seq):
    # if hwp is spinning up and down right next to each other, we can just remove them
    core.seq_assert_sorted(op_seq)
    def rewriter(seq_prev, b_next):
        if len(seq_prev) == 0:
            return [b_next]
        b_prev = seq_prev[-1]
        if (b_prev.name == 'sat.hwp_spin_up' and b_next.name == 'sat.hwp_spin_down') or \
           (b_prev.name == 'sat.hwp_spin_down' and b_next.name == 'sat.hwp_spin_up'):
            return seq_prev[:-1] + [cmd.OperationBlock(
                name='wait-until', 
                t0=b_prev.t0, 
                t1=b_next.t1, 
            )]
        else:
            return seq_prev+[b_next]
    return reduce(rewriter, op_seq, [])
