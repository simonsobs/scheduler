import numpy as np
import datetime as dt
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Union, Optional, Dict, Any, Tuple
import jax.tree_util as tu
from functools import partial
from importlib.metadata import version

from .. import core, source as src, utils as u, rules as ru
from .. import commands as cmd, instrument as inst
from .stages import get_build_stage
from ..instrument import CalTarget


logger = u.init_logger(__name__)

# ----------------------------------------------------
#                  Shared Command Durations
# ----------------------------------------------------

RELOCK_DURATION = 15*u.minute


# ----------------------------------------------------
#                  Base Telescope State
# ----------------------------------------------------

@dataclass_json
@dataclass(frozen=True)
class State(cmd.State):
    """
    State relevant to SAT operation scheduling. Inherits other fields:
    (`curr_time`, `az_now`, `el_now`, `az_speed_now`, `az_accel_now`)
    from the base State defined in `schedlib.commands`.

    Parameters
    ----------
    last_ufm_relock : Optional[datetime.datetime]
        The last time the UFM was relocked, or None if it has not been relocked.
    last_bias_step : Optional[datetime.datetime]
        The last time a bias step was performed, or None if no bias step has been performed.
    last_bias_step_boresight: Optional[float]
        The boresight (deg) at which the last bias step was taken, or None if no bias step has been performed.
    last_bias_step_elevation: Optional[float]
        The elevation (deg) at which the last bias step was taken, or None if no bias step has been performed.
    last_iv : Optional[datetime.datetime]
        The last time an iv curve was taken, or None if no iv curve has been taken.
    last_iv_boresight: Optional[float]
        The boresight (deg) at which the iv curve step was taken, or None if no iv curve has been taken.
    last_iv_elevation: Optional[float]
        The elevation (deg) at which the last iv curve was taken, or None if no iv curve has been taken.
    is_det_setup : bool
        Whether the detectors have been set up or not.
    has_active_channels : Optional[bool]
        Whether there are any active channels such that take_noise can be run befor relock
    """

    last_ufm_relock: Optional[dt.datetime] = None
    last_bias_step: Optional[dt.datetime] = None
    last_bias_step_boresight: Optional[float] = None
    last_bias_step_elevation: Optional[float] = None
    last_iv: Optional[dt.datetime] = None
    last_iv_boresight: Optional[float] = None
    last_iv_elevation: Optional[float] = None
    # relock sets to false, tracks if detectors are biased at all
    is_det_setup: bool = False
    has_active_channels: Optional[bool] = True

    def get_boresight(self):
        raise NotImplementedError(
            "get_boresight must be defined by child classes"
        )


# ----------------------------------------------------
#                  Base SchedMode
# ----------------------------------------------------

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


# ----------------------------------------------------
#                Operation Helpers
# ----------------------------------------------------
# Note that we are not registering here
# These are helper functions for the LAT and SAT to use in there operations

def preamble():
    return [
        f"# schedlib version: {version('schedlib')}",
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
        "",
        ]

def wrap_up(state, block):
    return state, [
         f"run.wait_until('{block.t1.isoformat(timespec='seconds')}')",
        "acu.stop_and_clear()"
    ]

def ufm_relock(state, commands=None, relock_cadence=24*u.hour):

    doit = False
    if state.last_ufm_relock is None:
        doit = True
    if not doit and relock_cadence is not None:
        if (state.curr_time - state.last_ufm_relock).total_seconds() > relock_cadence:
            doit = True
    if not doit and not state.has_active_channels:
        doit = True

    if doit:
        if commands is None:
            commands = [
                "",
                "####################### Relock #######################",
                "run.smurf.zero_biases()",
                "time.sleep(120)",
                "run.smurf.uxm_relock(concurrent=True)",
                "################## Relock Over #######################",
                ""
            ]
        elif not state.has_active_channels:
            if "run.smurf.take_noise(concurrent=True, tag='res_check')" in commands:
                commands.remove("run.smurf.take_noise(concurrent=True, tag='res_check')")

        state = state.replace(
            last_ufm_relock=state.curr_time,
            is_det_setup=False,
            has_active_channels=True
        )
        return state, RELOCK_DURATION, commands
    else:
        return state, 0, []

def det_setup(
        state,
        block,
        commands=None,
        apply_rot=True,
        iv_cadence=None,
        det_setup_duration=20*u.minute
    ):
    # when should det setup be done?
    # -> should always be done if the block is a cal block
    # -> should always be done if elevation has changed
    # -> should always be done if det setup has not been done yet
    # -> should be done at a regular interval if iv_cadence is not None
    # -> should always be done if boresight or corotator angle has changed
    doit = (block.subtype == 'cal')
    doit = doit or (not state.is_det_setup) or (state.last_iv is None)
    if not doit:
        if state.last_iv_elevation is not None:
            doit = doit or (
                not np.isclose(state.last_iv_elevation, block.alt, atol=1)
            )
        if apply_rot and state.last_iv_boresight is not None:
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
        return state, det_setup_duration, commands
    else:
        return state, 0, []

def cmb_scan(state, block):
    if (
        block.az_speed != state.az_speed_now or
        block.az_accel != state.az_accel_now or
        block.el_freq != state.el_freq_now or
        block.turnaround_method != state.turnaround_method_now
    ):
        args = [
            f"az_speed={block.az_speed}",
            f"az_accel={block.az_accel}, el_freq={block.el_freq}",
            f"turnaround_method='{block.turnaround_method}'",
        ]

        state = state.replace(
            az_speed_now=block.az_speed,
            az_accel_now=block.az_accel,
            el_freq_now=block.el_freq,
            turnaround_method_now=block.turnaround_method
        )
    else:
        args = []

    if block.el_mode != state.el_mode_now:
        args.append(f"el_mode='{block.el_mode}'")

        state = state.replace(
            el_mode_now=block.el_mode
        )

    if args:
        commands = [
            f"run.acu.set_scan_params({', '.join(args)})"
        ]
    else:
        commands = []

    commands.extend([
        f"# scan duration = {dt.timedelta(seconds=round((block.t1 - state.curr_time).total_seconds()))}",
        f"run.seq.scan(",
        f"    description='{block.name}',",
        f"    stop_time='{block.t1.isoformat(timespec='seconds')}',",
        f"    width={round(block.throw, 3)}" + (", az_drift=0" if block.scan_type == 1 else ", az_drift=None") + ",",
        f"    el_amp={block.el_amp},",
        f"    scan_type={block.scan_type},",
        f"    subtype='{block.subtype}', tag='{block.tag}',",
        f"    min_duration=600,",
    ])
    # add extra type3 params
    if block.scan_type == 3:
        commands += [
            f"    az_start='mid_dec',",
            f"    step_time=.05,",
        ]

    commands += [")"]

    return state, (block.t1 - state.curr_time).total_seconds(), commands

def source_scan(state, block):
    block = block.trim_left_to(state.curr_time)
    if block is None:
        return state, 0, ["# too late, don't scan"]
    if (
        block.az_speed != state.az_speed_now or
        block.az_accel != state.az_accel_now or
        block.el_freq != state.el_freq_now or
        block.turnaround_method != state.turnaround_method_now
    ):
        args = [
            f"az_speed={block.az_speed}",
            f"az_accel={block.az_accel}, el_freq={block.el_freq}",
            f"turnaround_method='{block.turnaround_method}'",
        ]

        state = state.replace(
            az_speed_now=block.az_speed,
            az_accel_now=block.az_accel,
            el_freq_now=block.el_freq,
            turnaround_method_now=block.turnaround_method
        )
    else:
        args = []

    if block.el_mode != state.el_mode_now:
        args.append(f"el_mode='{block.el_mode}'")

        state = state.replace(
            el_mode_now=block.el_mode
        )

    if args:
        commands = [
            f"run.acu.set_scan_params({', '.join(args)})"
        ]
    else:
        commands = []

    state = state.replace(az_now=block.az, el_now=block.alt)
    commands.extend([
        f"run.acu.move_to_target(az={round(block.az + block.az_offset,3)}, el={round(block.alt + block.alt_offset,3)},",
        f"    start_time='{block.t0.isoformat(timespec='seconds')}',",
        f"    stop_time='{block.t1.isoformat(timespec='seconds')}',",
        f"    drift={round(block.az_drift,5)})",
        "",
        f"print('Waiting until {block.t0.isoformat(timespec='seconds')} to start scan')",
        f"run.wait_until('{block.t0.isoformat(timespec='seconds')}')",
        "",
        f"# scan duration = {dt.timedelta(seconds=round((block.t1 - state.curr_time).total_seconds()))}",
        "run.seq.scan(",
        f"    description='{block.name}', ",
        f"    stop_time='{block.t1.isoformat(timespec='seconds')}', ",
        f"    width={round(block.throw,3)}, ",
        f"    az_drift={round(block.az_drift,5)}, ",
        f"    el_amp={block.el_amp},",
        f"    scan_type={block.scan_type},",
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
        return state, 60, ["run.smurf.bias_step(concurrent=True)",
                        f"run.wait_until('{(state.curr_time + dt.timedelta(seconds=60)).isoformat(timespec='seconds')}')"]
    else:
        return state, 0, []


# ----------------------------------------------------
#                  Base Telescope State
# ----------------------------------------------------

@dataclass
class TelPolicy:
    cmb_plan: str
    cal_plan: str
    stages: Dict[str, Any]
    rules: Dict[str, core.Rule]
    state_file: str = None
    blocks: Dict[str, Any] = field(default_factory=dict)
    geometries: List[Dict[str, Any]] = field(default_factory=list)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    cal_targets: List[CalTarget] = field(default_factory=list)
    wafer_sets: Dict[str, Any] = field(default_factory=dict)
    scan_tag: str = None
    az_speed: float = 0.5 # deg / s
    az_accel: float = 1.0 # deg / s^2
    az_offset: float = 0.0 # deg
    el_offset: float = 0.0 # deg
    xi_offset: float = 0.0 # deg
    eta_offset: float = 0.0 #deg
    az_motion_override: bool = False
    elevation_override: bool = False
    el_mode_override: str = ''
    az_branch_override: float = None
    allow_partial_override: float = None
    drift_override: bool = True
    allow_az_maneuver: bool = True
    iv_cadence: float = 4 * u.hour
    bias_step_cadence: float = 0.5 * u.hour
    relock_cadence: float = 24 * u.hour
    max_cmb_scan_duration: float = 1 * u.hour
    cryo_stabilization_time: float = 0 * u.second
    home_at_end: bool = False
    stow_position: Dict[str, Any] = field(default_factory=dict)
    rng: np.random.Generator = field(init=False, default=None)

    def construct_seq(self, loader_cfg, t0, t1):
        if loader_cfg['type'] == 'source':
            return src.source_gen_seq(loader_cfg['name'], t0, t1)
        elif loader_cfg['type'] == 'sat-cmb':
            blocks = inst.parse_sequence_from_toast_sat(
                loader_cfg['file'],
            )
            blocks = self.apply_overrides(blocks)
            return blocks
        elif loader_cfg['type'] == 'lat-cmb':
            blocks = inst.parse_sequence_from_toast_lat(
                loader_cfg['file'],
            )
            blocks = self.apply_overrides(blocks)
            return blocks
        else:
            raise ValueError(f"unknown sequence type: {loader_cfg['type']}")

    def apply_overrides(self, blocks):
        # these overrides get applied AFTER the telescope specific overrides
        if self.az_motion_override:
            blocks = core.seq_map(
                lambda b: b.replace(
                    az_speed=self.az_speed
                ), blocks
            )
            blocks = core.seq_map(
                lambda b: b.replace(
                    az_accel=self.az_accel
                ), blocks
            )
            blocks = core.seq_map(
                lambda b: b.replace(
                    el_freq=self.el_freq if b.scan_type==3 else b.el_freq
                ), blocks
            )
        if self.elevation_override is not None:
            blocks = core.seq_map(
                lambda b: b.replace(
                    alt=self.elevation_override
                ), blocks
            )

        if self.el_mode_override is not None:
            blocks = core.seq_map(
                lambda b: b.replace(
                    el_mode=self.el_mode_override
                ), blocks
            )
        return blocks

    def make_blocks(self, cmb_plan_type):
        assert cmb_plan_type in ['sat-cmb', 'lat-cmb'], (
            "Unrecognized cmb_plan_type.  Should be either sat-cmb or lat-cmb")

        # CMB
        blocks = {
            'baseline': {
                'cmb': {
                    'type': cmb_plan_type,
                    'file': self.cmb_plan,
                }
            },
        }

        # Cal
        blocks['calibration'] = {}
        for source in [
            "saturn", "jupiter", "moon", "uranus",
            "neptune", "mercury", "venus", "mars"
        ]:
            blocks['calibration'][source] = {
                'type': 'source',
                'name': source,
            }

        return blocks

    def divide_blocks(self,
                      block, scan_dt=dt.timedelta(minutes=60),
                      min_dt=dt.timedelta(minutes=15),
                      max_dt=dt.timedelta(minutes=75)
                    ):
        """
        Divide CMB blocks into smaller sub-blocks.  If a block has a duration less than max_dt,
        it is returned unchanged.  This function will randomize the duration of the first block
        between min_dt and scan_dt.  The remainder will be divided into as many scan_dt blocks as
        possible and the remainder will either be another shorter final block or added to one of the
        scan_dt blocks if the combined duration is less than max_dt.

        Parameters
        ----------
        block : core.ScanBlock
            The scan block to subidivide.
        scan_dt : dt.timedelta
            Nominal length of a scan.
        min_dt : dt.timedelta
            Minimum allowed length of a scan.
        max_dt : dt.timedelta
            Maximum allowed length of a scan.

        Returns
        -------
        blocks : list[core.ScanBlock]
            List of subdivided blocks
        """
        def check_blocks(block, blocks):
            assert blocks[0].t0 == block.t0, f"{block} division failed. t0 does not match."
            assert blocks[-1].t1 == block.t1, f"{block} division failed. t1 does not match."
            assert np.round(sum(b.duration.total_seconds() for b in blocks),0) == np.round(block.duration.total_seconds(),0), \
                f"{block} division failed. duration does not match."

        # add iteration number for divided block to uid
        def update_uid(blocks):
            for i, b in enumerate(blocks):
                tags = b.tag.split(',')
                for j, item in enumerate(tags):
                    if item.startswith('uid'):
                        tags[j] = item + '-pass-' + str(i)
                        break
                blocks[i] = blocks[i].replace(tag=",".join(tags))

        duration = block.duration

        # if the block is small enough, return it as is
        if duration <= max_dt:
            return [block]

        first_dur = self.rng.uniform(scan_dt.total_seconds() / 2, scan_dt.total_seconds())
        # how much time is left over after subtracting out first block
        remaining = (duration - dt.timedelta(seconds=first_dur)).total_seconds()

        # number of blocks after subtracting out random first block
        n_blocks = int(remaining // scan_dt.total_seconds())
        # leftover for final block
        remainder = remaining % scan_dt.total_seconds()

        # when the remainder is less than scan_dt
        if n_blocks == 0:
            if dt.timedelta(seconds=remainder) > min_dt:
                blocks = core.block_split(block, block.t0 + dt.timedelta(seconds=first_dur))
                update_uid(blocks)
                check_blocks(block, blocks)
                return blocks
            else:
                return [block]
            return blocks

        # split out first block with random duration
        blocks = []
        t_now = block.t0 + dt.timedelta(seconds=first_dur)
        split_blocks = core.block_split(block, t_now)
        blocks.append(split_blocks[0])

        # split out other blocks
        while t_now < block.t1:
            t_now = t_now + scan_dt
            split_blocks = core.block_split(split_blocks[-1], t_now)
            blocks.append(split_blocks[0])

        # add final block to last max_dt block if total duration <= max_dt + min_dt
        if len(blocks) > 2 and blocks[-1].duration <= min_dt:
            blocks[-2] = blocks[-2].extend_right(blocks[-1].duration)
            # throw away last block
            blocks = blocks[:-1]

        update_uid(blocks)
        check_blocks(block, blocks)
        return blocks

    def init_cmb_seqs(self, t0: dt.datetime, t1: dt.datetime) -> core.BlocksTree:
        """
        Initialize the CMB sequences for the scheduler to process.

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

        # set the seed for shuffling blocks
        self.rng = np.random.default_rng(int(t0.timestamp()))

        return blocks

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

    def build_schedule(self, t0: dt.datetime, t1: dt.datetime, state: State=None):
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
