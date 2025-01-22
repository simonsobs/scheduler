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
    last_ufm_relock : Optional[datetime.datetime]
        The last time the UFM was relocked, or None if it has not been relocked.
    last_bias_step : Optional[datetime.datetime]
        The last time a bias step was performed, or None if no bias step has been performed.
    is_det_setup : bool
        Whether the detectors have been set up or not.
    """
    boresight_rot_now: float = 0
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


# ----------------------------------------------------
#                Operation Helpers
# ----------------------------------------------------
# Note that we are not registering here
# These are helper functions for the LAT and SAT to use in there operations

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
    "",
    ]

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
        f"    subtype='{block.subtype}', tag='{block.tag}',",
        f"    min_duration=600,",
        ")",
    ])
    return state, (block.t1 - state.curr_time).total_seconds(), commands

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

@dataclass
class TelPolicy:
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
    max_cmb_scan_duration : float = 1 * u.hour
    allow_az_maneuver: bool = True
    wafer_sets: Dict[str, Any] = field(default_factory=dict)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    stages: Dict[str, Any] = field(default_factory=dict)

    def construct_seq(self, loader_cfg, t0, t1):
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
            lambda block: block.replace(subtype="cal"),
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
        ops, state = build_op.apply(seq, t0, t1, state, self.operations)
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

    def build_schedule(self, t0: dt.datetime, t1: dt.datetime, state: Optional[State] = None):
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