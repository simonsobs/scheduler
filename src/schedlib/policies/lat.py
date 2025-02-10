"""A production-level implementation of the LAT policy

"""
import numpy as np
import yaml
import os.path as op
from dataclasses import dataclass, field
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
from .tel import State, CalTarget, make_blocks

logger = u.init_logger(__name__)

BORESIGHT_DURATION = 1*u.minute
STIMULATOR_DURATION = 15*u.minute

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
def preamble():
    return tel.preamble()

@cmd.operation(name='lat.ufm_relock', return_duration=True)
def ufm_relock(state, commands=None):
    return tel.ufm_relock(state, commands)

# per block operation: block will be passed in as parameter
@cmd.operation(name='lat.det_setup', return_duration=True)
def det_setup(state, block, commands=None, apply_boresight_rot=True, iv_cadence=None):
    return tel.det_setup(state, block, commands, apply_boresight_rot, iv_cadence)

@cmd.operation(name='lat.cmb_scan', return_duration=True)
def cmb_scan(state, block):
    return tel.cmb_scan(state, block)

@cmd.operation(name='lat.source_scan', return_duration=True)
def source_scan(state, block):
    return tel.source_scan(state, block)

@cmd.operation(name='lat.setup_boresight', return_duration=True)
def setup_boresight(state, block, apply_boresight_rot=True):
    commands = []
    duration = 0

    if apply_boresight_rot and (
            state.boresight_rot_now is None or state.boresight_rot_now != block.boresight_angle
        ):

        commands += [f"run.acu.set_boresight({block.boresight_angle})"]
        state = state.replace(boresight_rot_now=block.boresight_angle)
        duration += BORESIGHT_DURATION

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
def move_to(state, az, el, min_el=48, force=False):
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
    # These are just the median of the wafers and an ~estimated radius
    # To be updated later
    return {
        "i1_ws0": {
            "center": [1.3516076803207397, 0.5679303407669067],
            "radius": 0.3,
        },
        "i1_ws1": {
            "center": [1.363024353981018, 1.2206860780715942],
            "radius": 0.3,
        },
        "i1_ws2": {
            "center": [1.9164373874664307, 0.9008757472038269],
            "radius": 0.3,
        },
        "i6_ws0": {
            "center": [1.3571038246154785, -1.2071731090545654],
            "radius": 0.3,
        },
        "i6_ws1": {
            "center": [1.3628365993499756, -0.5654135942459106],
            "radius": 0.3,
        },
        "i6_ws2": {
            "center": [1.9065929651260376, -0.8826764822006226],
            "radius": 0.3,
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
    az_motion_override=False,
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
        'min_angle': 30,
        'min_sun_time': 1980,
        'min_el': 30,
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
                'min_duration': 60
            },
            'sun-avoidance': sun_policy,
            'az-range': az_range,
        },
        'operations': operations,
        'cal_targets': cal_targets,
        'scan_tag': None,
        'boresight_override': boresight_override,
        'az_motion_override': az_motion_override,
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
    """

    state_file: Optional[str] = None

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
    def from_defaults(cls, master_file, az_speed=0.8, az_accel=1.5,
        iv_cadence=4*u.hour, bias_step_cadence=0.5*u.hour,
        max_cmb_scan_duration=1*u.hour, cal_targets=None,
        az_stow=None, el_stow=None, boresight_override=None,
        az_motion_override=False, state_file=None, **op_cfg
    ):
        if cal_targets is None:
            cal_targets = []

        x = cls(**make_config(
            master_file, az_speed, az_accel, iv_cadence,
            bias_step_cadence, max_cmb_scan_duration,
            cal_targets, az_stow, el_stow, boresight_override,
            az_motion_override, **op_cfg
        ))
        x.state_file=state_file
        return x

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
        columns = ["start_utc", "stop_utc", "hwp_dir", "rotation", "az_min", "az_max",
                   "el", "speed", "accel", "pass", "sub", "uid", "patch"]
        # construct seqs by traversing the blocks definition dict
        blocks = tu.tree_map(
            partial(self.construct_seq, t0=t0, t1=t1, columns=columns),
            self.blocks,
            is_leaf=lambda x: isinstance(x, dict) and 'type' in x
        )

        # by default add calibration blocks specified in cal_targets if not already specified
        for cal_target in self.cal_targets:
            if isinstance(cal_target, CalTarget):
                source = cal_target.source
                if source not in blocks['calibration']:
                    blocks['calibration'][source] = src.source_gen_seq(source, t0, t1)
            elif isinstance(cal_target, StimulatorTarget):
                stimulator_candidates = []
                current_date = t0.date()
                end_date = t1.date()

                while current_date <= end_date:
                    candidate_time = dt.datetime.combine(current_date, dt.time(cal_target.hour, 0), tzinfo=dt.timezone.utc)
                    if t0 <= candidate_time <= t1:
                        stimulator_candidates.append(
                            inst.StareBlock(
                                name='stimulator',
                                t0=candidate_time,
                                t1=candidate_time + dt.timedelta(seconds=cal_target.duration),
                                az=cal_target.az_target,
                                alt=cal_target.el_target,
                                subtype='stimulator'
                            )
                        )
                    current_date += dt.timedelta(days=1)
                blocks['calibration']['stimulator'] = stimulator_candidates

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
                logger.info(f"-> planning stimulator scans for {target}...")
                cal_blocks += core.seq_map(lambda b: b.replace(subtype='stimulator'), 
                                           blocks['calibration']['stimulator'])
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

        return blocks

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
            elif block.subtype == 'stimulator':
                return {
                    'name': block.name,
                    'block': block,
                    'pre': [],
                    'in': stimulator_in,
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
        else:
            az_stow = seq[-1]['block'].az
            alt_stow = seq[-1]['block'].alt
        end_block = {
            'name': 'post-session',
            'block': inst.StareBlock(name="post-session", az=az_stow, alt=alt_stow, t0=t1-dt.timedelta(seconds=1), t1=t1),
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

    def add_cal_target(self, *args, **kwargs):
        self.cal_targets.append(make_cal_target(*args, **kwargs))

    def add_stimulator_target(self, el_target, hour_utc=12, az_target=180, duration=STIMULATOR_DURATION, **kwargs):
        self.cal_targets.append(StimulatorTarget(hour=hour_utc, az_target=az_target, el_target=el_target, duration=duration))