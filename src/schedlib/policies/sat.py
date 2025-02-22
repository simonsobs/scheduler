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
from functools import reduce, partial

from .. import config as cfg, core, source as src, rules as ru
from .. import commands as cmd, instrument as inst, utils as u
from ..thirdparty import SunAvoidance
from .stages import get_build_stage
from .stages.build_op import get_parking
from . import tel
from .tel import CalTarget

logger = u.init_logger(__name__)

HWP_SPIN_UP = 7*u.minute
HWP_SPIN_DOWN = 15*u.minute
BORESIGHT_DURATION = 1*u.minute
WIREGRID_DURATION = 15*u.minute

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
        Current direction of HWP.  True is forward, False is backwards.
    """
    boresight_rot_now: float = 0
    hwp_spinning: bool = False
    hwp_dir: bool = None

    def get_boresight(self):
        return self.boresight_rot_now

@dataclass(frozen=True)
class WiregridTarget:
    hour: int
    el_target: float
    az_target: float = 180
    duration: float = WIREGRID_DURATION

class SchedMode(tel.SchedMode):
    """
    Enumerate different options for scheduling operations in SATPolicy.

    Attributes
    ----------
    Wiregrid : str
        'wiregrid'; Wiregrid observations scheduled between block.t0 and block.t1
    """
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
    )

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
    base = tel.preamble()
    append = ["sup = OCSClient('hwp-supervisor')", "",]
    return base + append

@cmd.operation(name='sat.ufm_relock', return_duration=True)
def ufm_relock(state, commands=None):
    return tel.ufm_relock(state, commands)

@cmd.operation(name='sat.hwp_spin_up', return_duration=True)
def hwp_spin_up(state, block, disable_hwp=False):
    cmds = []
    duration = 0

    if disable_hwp:
        return state, 0, ["# hwp disabled"]

    elif state.hwp_spinning:
        # if spinning in opposite direction, spin down first
        if block.hwp_dir is not None and state.hwp_dir != block.hwp_dir:
            duration += HWP_SPIN_DOWN
            cmds += [
            "run.hwp.stop(active=True)",
            "sup.disable_driver_board()",
            ]
        else:
            return state, 0, [f"# hwp already spinning with forward={state.hwp_dir}"]

    hwp_dir = block.hwp_dir if block.hwp_dir is not None else state.hwp_dir
    state = state.replace(hwp_dir=hwp_dir)
    state = state.replace(hwp_spinning=True)

    freq = 2 if hwp_dir else -2
    return state, duration + HWP_SPIN_UP, cmds + [
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
        return state, HWP_SPIN_DOWN, [
            "run.hwp.stop(active=True)",
            "sup.disable_driver_board()",
        ]

# per block operation: block will be passed in as parameter
@cmd.operation(name='sat.det_setup', return_duration=True)
def det_setup(state, block, commands=None, apply_boresight_rot=True, iv_cadence=None):
    return tel.det_setup(state, block, commands, apply_boresight_rot, iv_cadence)

@cmd.operation(name='sat.cmb_scan', return_duration=True)
def cmb_scan(state, block):
    return tel.cmb_scan(state, block)

@cmd.operation(name='sat.source_scan', return_duration=True)
def source_scan(state, block):
    return tel.source_scan(state, block)

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
        duration += BORESIGHT_DURATION

    return state, duration, commands

# passthrough any arguments, to be used in any sched-mode
@cmd.operation(name='sat.bias_step', return_duration=True)
def bias_step(state, block, bias_step_cadence=None):
    return tel.bias_step(state, block, bias_step_cadence)

@cmd.operation(name='sat.wiregrid', duration=WIREGRID_DURATION)
def wiregrid(state):
    return state, [
        "run.wiregrid.calibrate(continuous=False, elevation_check=True, boresight_check=False, temperature_check=False)"
    ]

@cmd.operation(name="move_to", return_duration=True)
def move_to(state, az, el, min_el=48, force=False):
    if not force and (state.az_now == az and state.el_now == el):
        return state, 0, []

    duration = 0
    cmd = []

    if state.hwp_spinning and el < min_el:
        state = state.replace(hwp_spinning=False)
        duration += HWP_SPIN_DOWN
        cmd += [
            "run.hwp.stop(active=True)",
            "sup.disable_driver_board()",
        ]

    cmd += [
        f"run.acu.move_to(az={round(az, 3)}, el={round(el, 3)})",
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
    min_hwp_el : float
        the minimum elevation a move command to go to without stopping the hwp first
    """
    hwp_override: Optional[bool] = None
    min_hwp_el: float = 48 # deg
    boresight_override: Optional[float] = None
 
    def apply_overrides(self, blocks):
        if self.boresight_override is not None:
            blocks = core.seq_map(
                lambda b: b.replace(
                    boresight_angle=self.boresight_override
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

    def divide_blocks(self, block, max_dt=dt.timedelta(minutes=60), min_dt=dt.timedelta(minutes=15)):
        duration = block.duration

        # if the block is small enough, return it as is
        if duration <= (max_dt + min_dt):
            return [block]

        n_blocks = duration // max_dt
        remainder = duration % max_dt

        # split if 1 block with remainder > min duration
        if n_blocks == 1:
            return core.block_split(block, block.t0 + max_dt)

        blocks = []
        # calculate the offset for splitting
        offset = (remainder + max_dt) / 2 if remainder.total_seconds() > 0 else max_dt

        split_blocks = core.block_split(block, block.t0 + offset)
        blocks.append(split_blocks[0])

        # split the remaining block into chunks of max duration
        for i in range(n_blocks - 1):
            split_blocks = core.block_split(split_blocks[-1], split_blocks[-1].t0 + max_dt)
            blocks.append(split_blocks[0])

        # add the remaining part
        if remainder.total_seconds() > 0:
            split_blocks = core.block_split(split_blocks[-1], split_blocks[-1].t0 + offset)
            blocks.append(split_blocks[0])

        return blocks

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
            partial(self.construct_seq, t0=t0, t1=t1,),
            self.blocks,
            is_leaf=lambda x: isinstance(x, dict) and 'type' in x
        )

        # override hwp direction
        if self.hwp_override is not None:
            blocks['baseline'] = core.seq_map(
                lambda b: b.replace(
                    hwp_dir=self.hwp_override
                ), blocks['baseline']
            )

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
        wiregrid_in = [op for op in self.operations if op['sched_mode'] == SchedMode.Wiregrid]

        def map_block(block):
            if block.subtype == 'cal':
                return {
                    'name': block.name,
                    'block': block,
                    'pre': cal_pre,
                    'in': cal_in,
                    'post': cal_post,
                    'priority': block.priority
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
                    'pre': [],
                    'in': wiregrid_in,
                    'post': [],
                    'priority': block.priority
                }
            else:
                raise ValueError(f"unexpected block subtype: {block.subtype}")

        seq = [map_block(b) for b in seq]

        # check if any observations were added
        assert len(seq) != 0, "No observations fall within time-range"

        start_block = {
            'name': 'pre-session',
            'block': inst.StareBlock(name="pre-session", az=state.az_now, alt=state.el_now, t0=t0, t1=t0+dt.timedelta(seconds=1)),
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
        else:
            az_stow = seq[-1]['block'].az
            alt_stow = seq[-1]['block'].alt
        end_block = {
            'name': 'post-session',
            'block': inst.StareBlock(name="post-session", az=az_stow, alt=alt_stow, t0=t1-dt.timedelta(seconds=1), t1=t1),
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

    def add_wiregrid_target(self, el_target, hour_utc=12, az_target=180, duration=WIREGRID_DURATION, **kwargs):
        self.cal_targets.append(WiregridTarget(hour=hour_utc, az_target=az_target, el_target=el_target, duration=duration))

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
