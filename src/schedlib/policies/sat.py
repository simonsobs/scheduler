"""A production-level implementation of the SAT policy

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
from ..instrument import CalTarget, WiregridTarget


logger = u.init_logger(__name__)

HWP_SPIN_UP = 7*u.minute
HWP_SPIN_DOWN = 15*u.minute
BORESIGHT_DURATION = 1*u.minute

COMMANDS_HWP_BRAKE = [
    "run.smurf.stream('on', subtype='cal', tag='hwp_spin_down')",
    "run.hwp.stop(active=True)",
    "sup.disable_driver_board()",
    "run.smurf.stream('off')",
    "",
]
COMMANDS_HWP_STOP = [
    "run.smurf.stream('on', subtype='cal', tag='hwp_spin_down')",
    "run.hwp.stop(active=False)",
    "sup.disable_driver_board()",
    "run.smurf.stream('off')",
    "",
]

@dataclass_json
@dataclass(frozen=True)
class State(tel.State):
    """
    State relevant to SAT operation scheduling. Inherits other fields:
    (`curr_time`, `az_now`, `el_now`, `az_speed_now`, `az_accel_now`)
    from the base State defined in `schedlib.commands`.And others from
    `tel.State`

    Parameters
    ----------
    boresight_rot_now : int
        The current boresight rotation state.
    hwp_spinning : bool
        Whether the high-precision measurement wheel is spinning or not.
    hwp_dir : bool
        Current direction of HWP. True is Counter-clockwise seen from sky
        (positive frequency), False is clock wise seen from sky (negative
        frequency).
    """
    boresight_rot_now: float = 0
    hwp_spinning: bool = False
    hwp_dir: bool = None

    def get_boresight(self):
        return self.boresight_rot_now

class SchedMode(tel.SchedMode):
    """
    Enumerate different options for scheduling operations in SATPolicy.

    Attributes
    ----------
    Wiregrid : str
        'wiregrid'; Wiregrid observations scheduled between block.t0 and block.t1
    """
    PreWiregrid = 'pre_wiregrid'
    Wiregrid = 'wiregrid'


def make_geometry(xi_offset=0., eta_offset=0.):
    logger.info(f"making geometry with xi offset={xi_offset}, eta offset={eta_offset}")
    ## default SAT optics offsets
    d_xi = 10.9624
    d_eta_side = 6.46363
    d_eta_mid = 12.634

    return {
        'ws3': {
            'center': [-d_xi + xi_offset, d_eta_side + eta_offset],
            'radius': 6,
        },
        'ws2': {
            'center': [-d_xi + xi_offset, -d_eta_side + eta_offset],
            'radius': 6,
        },
        'ws4': {
            'center': [0 + xi_offset, d_eta_mid + eta_offset],
            'radius': 6,
        },
        'ws0': {
            'center': [0 + xi_offset, 0 + eta_offset],
            'radius': 6,
        },
        'ws1': {
            'center': [0 + xi_offset, -d_eta_mid + eta_offset],
            'radius': 6,
        },
        'ws5': {
            'center': [d_xi + xi_offset, d_eta_side + eta_offset],
            'radius': 6,
        },
        'ws6': {
            'center': [d_xi + xi_offset, -d_eta_side + eta_offset],
            'radius': 6,
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
    source_direction=None,
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
        focus_str = focus
    else:
        focus_str = array_focus[int(boresight)].get(focus, focus)

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
    base = tel.preamble()
    append = ["sup = OCSClient('hwp-supervisor')", "",]
    return base + append

@cmd.operation(name='sat.wrap_up', duration=0)
def wrap_up(state, block):
    return tel.wrap_up(state, block)

@cmd.operation(name='sat.ufm_relock', return_duration=True)
def ufm_relock(state, commands=None, relock_cadence=24*u.hour):
    return tel.ufm_relock(state, commands, relock_cadence)

@cmd.operation(name='sat.hwp_spin_up', return_duration=True)
def hwp_spin_up(state, block, disable_hwp=False, brake_hwp=True):
    cmds = []
    duration = 0

    if disable_hwp:
        return state, 0, ["# hwp disabled"]

    elif state.hwp_spinning:
        # if spinning in opposite direction, spin down first
        if block.hwp_dir is not None and state.hwp_dir != block.hwp_dir:
            duration += HWP_SPIN_DOWN
            cmds += COMMANDS_HWP_BRAKE if brake_hwp else COMMANDS_HWP_STOP
        else:
            direction = "ccw (positive frequency)" if state.hwp_dir \
                else "cw (negative frequency)"
            return state, 0, [f"# hwp already spinning " + direction]

    hwp_dir = block.hwp_dir if block.hwp_dir is not None else state.hwp_dir
    state = state.replace(hwp_dir=hwp_dir)
    state = state.replace(hwp_spinning=True)

    freq = 2 if hwp_dir else -2
    return state, duration + HWP_SPIN_UP, cmds + [
        "run.smurf.stream('on', subtype='cal', tag='hwp_spin_up')",
        "sup.enable_driver_board()",
        f"run.hwp.set_freq(freq={freq})",
        "run.smurf.stream('off')",
        "",
    ]

@cmd.operation(name='sat.hwp_spin_down', return_duration=True)
def hwp_spin_down(state, disable_hwp=False, brake_hwp=True):
    if disable_hwp:
        return state, 0, ["# hwp disabled"]
    elif not state.hwp_spinning:
        return state, 0, ["# hwp already stopped"]
    else:
        state = state.replace(hwp_spinning=False)
        cmd = COMMANDS_HWP_BRAKE if brake_hwp else COMMANDS_HWP_STOP
        return state, HWP_SPIN_DOWN, cmd

# per block operation: block will be passed in as parameter
@cmd.operation(name='sat.det_setup', return_duration=True)
def det_setup(state, block, commands=None, apply_boresight_rot=True, iv_cadence=None, det_setup_duration=20*u.minute):
    return tel.det_setup(state, block, commands, apply_boresight_rot, iv_cadence, det_setup_duration)

@cmd.operation(name='sat.cmb_scan', return_duration=True)
def cmb_scan(state, block):
    return tel.cmb_scan(state, block)

@cmd.operation(name='sat.source_scan', return_duration=True)
def source_scan(state, block):
    return tel.source_scan(state, block)

@cmd.operation(name='sat.setup_boresight', return_duration=True)
def setup_boresight(state, block, apply_boresight_rot=True, brake_hwp=True, cryo_stabilization_time=0*u.second):
    commands = []
    duration = 0

    if apply_boresight_rot and (
            state.boresight_rot_now is None or state.boresight_rot_now != block.boresight_angle
        ):
        if state.hwp_spinning:
            state = state.replace(hwp_spinning=False)
            duration += HWP_SPIN_DOWN
            commands += COMMANDS_HWP_BRAKE if brake_hwp else COMMANDS_HWP_STOP
        assert not state.hwp_spinning
        commands += [f"run.acu.set_boresight({block.boresight_angle})"]
        state = state.replace(boresight_rot_now=block.boresight_angle)
        duration += BORESIGHT_DURATION

        if cryo_stabilization_time > 0:
            commands += [f"time.sleep({cryo_stabilization_time})"]
            duration += cryo_stabilization_time

    return state, duration, commands

# passthrough any arguments, to be used in any sched-mode
@cmd.operation(name='sat.bias_step', return_duration=True)
def bias_step(state, block, bias_step_cadence=None):
    return tel.bias_step(state, block, bias_step_cadence)

@cmd.operation(name='sat.wiregrid', return_duration=True)
def wiregrid(state, block, min_wiregrid_el=47.5):
    assert state.hwp_spinning == True, "hwp is not spinning"
    assert block.alt >= min_wiregrid_el, f"Block {block} is below the minimum wiregrid elevation of {min_wiregrid_el} degrees."

    if block.name == 'wiregrid_gain':
        return state, (block.t1 - state.curr_time).total_seconds(), [
            "run.wiregrid.calibrate(continuous=False, elevation_check=True, boresight_check=False, temperature_check=False)"
        ]
    elif block.name == 'wiregrid_time_const':
        # wiregrid time const reverses the hwp direction
        state = state.replace(hwp_dir=not state.hwp_dir)
        direction = "ccw (positive frequency)" if state.hwp_dir \
                else "cw (negative frequency)"
        return state, (block.t1 - state.curr_time).total_seconds(), [
            "run.wiregrid.time_constant(num_repeats=1)",
            f"# hwp direction reversed, now spinning " + direction,
            ]

@cmd.operation(name="move_to", return_duration=True)
def move_to(state, az, el, az_offset=0, el_offset=0, min_el=48, brake_hwp=True, force=False):
    if not force and (state.az_now == az and state.el_now == el):
        return state, 0, []

    duration = 0
    cmd = []

    if state.hwp_spinning and el < min_el:
        state = state.replace(hwp_spinning=False)
        duration += HWP_SPIN_DOWN
        cmd += COMMANDS_HWP_BRAKE if brake_hwp else COMMANDS_HWP_STOP
    cmd += [
        f"run.acu.move_to(az={round(az + az_offset, 3)}, el={round(el + el_offset, 3)})",
    ]
    state = state.replace(az_now=az, el_now=el)

    return state, duration, cmd

@dataclass
class SATPolicy(tel.TelPolicy):
    """a more realistic SAT policy.

    Parameters
    ----------
    hwp_override : bool
        a bool that specifies the hwp direction if overriding the master schedule.  True is forward
        and False is reverse.
    brake_hwp : bool
        a bool that specifies whether or not active braking should be used for the hwp.
    disable_hwp : bool
        a bool that specifies whether or not to disable the hwp entirely.
    min_hwp_el : float
        the minimum elevation a move command to go to without stopping the hwp first
    boresight_override : float
        the angle of boresight to use if not None
    wiregrid_az : float
        azimuth to use for wiregrid measurements
    wiregrid_el : float
        elevation to use for wiregrid measurements
    """
    hwp_override: Optional[bool] = None
    brake_hwp: Optional[bool] = True
    disable_hwp: bool = False
    min_hwp_el: float = 48 # deg
    boresight_override: Optional[float] = None
    wiregrid_az: float = 180
    wiregrid_el: float = 48

    def apply_overrides(self, blocks):
        if self.boresight_override is not None:
            blocks = core.seq_map(
                lambda b: b.replace(
                    boresight_angle=self.boresight_override
                ), blocks
            )
        # override hwp direction
        if self.hwp_override is not None:
            blocks = core.seq_map(
                lambda b: b.replace(
                    hwp_dir=self.hwp_override
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

    def init_cmb_seqs(self, t0: dt.datetime, t1: dt.datetime) -> core.BlocksTree:
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
            partial(self.construct_seq, t0=t0, t1=t1,),
            self.blocks,
            is_leaf=lambda x: isinstance(x, dict) and 'type' in x
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
        blocks : BlocksTree
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

        # min duration rule
        if 'min-duration' in self.rules:
            # initial min duration rule to remove edge cases of very short scans
            min_dur_rule = ru.make_rule('min-duration', **self.rules['min-duration'])
            blocks['calibration'] = min_dur_rule(blocks['calibration'])

        # -----------------------------------------------------------------
        # step 2: plan calibration scans
        #   - refer to each target specified in cal_targets
        #   - same source can be observed multiple times with different
        #     array configurations (i.e. using array_query)
        # -----------------------------------------------------------------
        logger.info("planning calibration scans...")
        cal_blocks = []

        saved_cal_targets = []
        for target in self.cal_targets:
            logger.info(f"-> planning calibration scans for {target}...")

            if isinstance(target, WiregridTarget):
                continue

            assert target.source in blocks['calibration'], f"source {target.source} not found in sequence"

            # get list of possible array queries
            if isinstance(target.array_query, list):
                array_queries = target.array_query.copy()
            else:
                array_queries = [target.array_query]

            # get list of allow_partials
            if isinstance(target.allow_partial, list):
                allow_partial = target.allow_partial.copy()
            else:
                allow_partial = [target.allow_partial]

            # loop over array queries and try to find a source
            for array_query in array_queries:
                logger.info(f"trying array_query={array_query}")
                target = replace(target, array_query=array_query)
                target = replace(target, allow_partial=allow_partial)

                source_scans = self.make_source_scans(target, blocks, sun_rule)

                if len(source_scans) == 0:
                    # try allow_partial=True if overriding
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

                if array_query not in target.tag:
                    tag = f"{cal_block.tag},{array_query},{target.tag}"
                else:
                    tag = f"{cal_block.tag},{target.tag}"

                # update tag, speed, accel, etc
                cal_block = cal_block.replace(
                    az_speed = target.az_speed if target.az_speed is not None else self.az_speed,
                    az_accel = target.az_accel if target.az_accel is not None else self.az_accel,
                    tag=tag
                )

                # override hwp direction
                if self.hwp_override is not None:
                    cal_block = cal_block.replace(
                        hwp_dir=self.hwp_override
                    )

                cal_blocks.append(cal_block)
                saved_cal_targets.append(target)

                # don't test other array queries if we have one that works
                break

        unique_cal_blocks = []
        for i, cal_block in enumerate(cal_blocks):
            if not saved_cal_targets[i].from_table:
                unique_cal_blocks.append(cal_block)
            else:
                # whether to keep rising or setting blocks for current week
                rising = cal_block.t0.isocalendar()[1] % 2 == 0
                other_cal_blocks = [other_cal_block for j, other_cal_block in enumerate(cal_blocks) if j!=i]
                other_saved_cal_targets = [other_saved_cal_target for j, other_saved_cal_target in enumerate(saved_cal_targets) if j!=i]

                # if any blocks has same source and array query
                if any(other_cal_block.name==cal_block.name for other_cal_block in other_cal_blocks) and \
                    any(other_saved_cal_target.array_query==saved_cal_targets[i].array_query for other_saved_cal_target in other_saved_cal_targets):
                    # add if source direction matches week's direction (if not it will be skipped)
                    if (saved_cal_targets[i].source_direction == "rising" and rising) or \
                    (saved_cal_targets[i].source_direction == "setting" and not rising):
                        unique_cal_blocks.append(cal_block)
                # if no other similar blocks schedule it
                else:
                    unique_cal_blocks.append(cal_block)

        blocks['calibration'] = unique_cal_blocks + blocks['calibration']['wiregrid']

        logger.info(f"-> after calibration policy: {u.pformat(blocks['calibration'])}")

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
            lambda block: block.replace(subtype="cal") if block.subtype != 'wiregrid' else block,
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

        # add hwp direction to cal blocks
        if self.hwp_override is None:
            for i, block in enumerate(blocks):
                if (block.subtype=='cal' or block.subtype=='wiregrid') and block.hwp_dir is None:
                    candidates = [cmb_block for cmb_block in blocks if cmb_block.subtype == "cmb" and cmb_block.t0 < block.t0]
                    if candidates:
                        cmb_block = max(candidates, key=lambda x: x.t0)
                    else:
                        candidates = [cmb_block for cmb_block in blocks if cmb_block.subtype == "cmb" and cmb_block.t0 > block.t0]
                        if candidates:
                            cmb_block = min(candidates, key=lambda x: x.t0)
                        else:
                            raise ValueError(f"Cannot assign HWP direction to cal block {block}")
                    blocks[i] = block.replace(hwp_dir=cmb_block.hwp_dir)

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
            el_now=48,
            boresight_rot_now=None,
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
        build_op = get_build_stage('build_op', {'policy_config': self, **self.stages.get('build_op', {})})

        # first resolve overlapping between cal and cmb
        cal_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'cal', seq))
        cmb_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'cmb', seq))

        wiregrid_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'wiregrid', seq))
        cal_blocks += wiregrid_blocks
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
        wiregrid_pre = [op for op in self.operations if op['sched_mode'] == SchedMode.PreWiregrid]
        wiregrid_in = [op for op in self.operations if op['sched_mode'] == SchedMode.Wiregrid]

        def map_block(block):
            if block.subtype == 'cal':
                return {
                    'name': block.name,
                    'block': block,
                    'pre': cal_pre,
                    'in': cal_in,
                    'post': cal_post,
                    'priority': -1
                }
            elif block.subtype == 'cmb':
                return {
                    'name': block.name,
                    'block': block,
                    'pre': cmb_pre,
                    'in': cmb_in,
                    'post': cmb_post,
                    'priority': block.priority
                }
            elif block.subtype == 'wiregrid':
                return {
                    'name': block.name,
                    'block': block,
                    'pre': wiregrid_pre,
                    'in': wiregrid_in,
                    'post': [],
                    'priority': -1
                }
            else:
                raise ValueError(f"unexpected block subtype: {block.subtype}")

        seq = [map_block(b) for b in seq]

        # check if any observations were added
        #assert len(seq) != 0, "No observations fall within time-range"

        start_block = {
            'name': 'pre-session',
            'block': inst.StareBlock(name="pre-session", az=state.az_now, alt=state.el_now, az_offset=self.az_offset, alt_offset=self.el_offset,
                                     t0=t0, t1=t0+dt.timedelta(seconds=1)),
            'pre': [],
            'in': [],
            'post': pre_sess,  # scheduled after t0
            'priority': -1,
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
            'priority': -1,
            'pinned': True # remain unchanged during multi-pass
        }
        seq = [start_block] + seq + [end_block]

        ops, state = build_op.apply(seq, t0, t1, state)
        if return_state:
            return ops, state
        return ops

    def add_cal_target(self, *args, **kwargs):
        self.cal_targets.append(make_cal_target(*args, **kwargs))

    def add_wiregrid_target(self, el_target, hour_utc=12, az_target=180, **kwargs):
        self.cal_targets.append(WiregridTarget(hour=hour_utc, az_target=az_target, el_target=el_target))

# ------------------------
# utilities
# ------------------------

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
