import numpy as np
import datetime as dt
from dataclasses import dataclass, field, replace
from dataclasses_json import dataclass_json
from typing import List, Union, Optional, Dict, Any, Tuple

from .. import core, utils as u, source as src, rules as ru
from .. import commands as cmd, instrument as inst
from ..thirdparty import SunAvoidance
from .stages import get_build_stage
from .stages.build_op import get_parking
from . import tel
from .tel import SchedMode
from ..instrument import CalTarget


logger = u.init_logger(__name__)


# ----------------------------------------------------
#                  LAT Command Durations
# ----------------------------------------------------

COROTATOR_DURATION = 1*u.minute
STIMULATOR_DURATION = 4*u.minute

# ----------------------------------------------------
#                   Utilities
# ----------------------------------------------------

def el_to_locked_corotator(el: float) -> float:
    """
    Calculates the locked corotator angle for a given elevation.  The
    corotator angle is set to 0 deg at elevations of 60 deg and 120 deg.

    Parameters
    ----------
    el : float
        The elevation of the boresight in degrees.
    """
    if el <= 90:
        return el - 60
    else:
        return el - 120


def corotator_to_boresight(el: float, corotator: float) -> float:
    """
    Calculates the boresight angle (-roll) from the corotator angle.
    When the corotator is locked, The corotator angle is determined from
    the function ``el_to_locked_corotator``, otherwise it can be any value
    between bounds (hardware min and max are -45 and 45 degrees).

    Parameters
    ----------
    el : float
        The elevation of the boresight in degrees.
    corotator : float
        The rotation angle of the corotator in degrees.
    """
    return -(el - 60 - corotator)


def boresight_to_corotator(el: float, boresight: float) -> float:
    """
    Calculates the corotator angle from the boresight (-roll) angle.
    The boresight angle is determined from the function ``corotator_to_boresight``
    using a fixed corotator angle or the value from ``el_to_locked_corotator``.

    Parameters
    ----------
    el : float
        The elevation of the boresight in degrees.
    boresight : float
        The rotation angle of the boresight (-roll) angle in degrees.
    """
    return boresight + el - 60


# ----------------------------------------------------
#                  LAT State
# ----------------------------------------------------

@dataclass_json
@dataclass(frozen=True)
class State(tel.State):
    """
    State relevant to LAT operation scheduling. Inherits other fields:
    (`curr_time`, `az_now`, `el_now`, `az_speed_now`, `az_accel_now`, 'el_freq_now')
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


# ----------------------------------------------------
#                  LAT Operations
# ----------------------------------------------------

@cmd.operation(name="lat.preamble", return_duration=True)
def preamble(state, open_shutter=False):
    cmd = tel.preamble()
    cmd += ["acu.clear_faults()"]
    cmd += [
        "################### Basic Checks ###################",
        "acu_data = acu.monitor.status().session['data']",
        "",
        f"assert np.round(acu_data['StatusDetailed']['Elevation current position'], 1) == {state.el_now}, 'Elevation check failed'",
        f"assert np.round(acu_data['Status3rdAxis']['Co-Rotator current position'], 1) == {state.corotator_now}, 'Corotator angle check failed'",
        "################### Checks  Over ###################",
        "",
        ]
    if open_shutter:
        cmd += ["acu.stop_and_clear()",
                "run.acu.set_shutter(action='open')"
            ]
    return state, 0, cmd

@cmd.operation(name='lat.wrap_up', duration=0)
def wrap_up(state, block, close_shutter=False):
    state, cmd = tel.wrap_up(state, block)
    if close_shutter:
        cmd += ["run.acu.set_shutter(action='close')"]
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

@cmd.operation(name='lat.stimulator', return_duration=True)
def stimulator(state):
    cmd = [
        "# run stimulator",
        f"run.stimulator.calibrate_gain_tau()"
        ""
    ]
    return state, STIMULATOR_DURATION, cmd

@cmd.operation(name='lat.setup_corotator', return_duration=True)
def setup_corotator(state, block, apply_corotator_rot=True, cryo_stabilization_time=180*u.second,
                    corotator_offset=0., corotator_bounds=[-45, 45]):
    commands = []
    duration = 0

    if apply_corotator_rot and (
            state.corotator_now is None or state.corotator_now != block.corotator_angle
        ):

        assert np.abs(np.max(corotator_bounds)) <= 45, f"corotator bounds {corotator_bounds} is above hardware limit"

        assert (block.corotator_angle >= corotator_bounds[0] and block.corotator_angle <= corotator_bounds[1]), (
            f"corotator angle {block.corotator_angle} not within bounds of [{corotator_bounds[0]}, {corotator_bounds[1]}])")

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
#                  Base LAT Policy
# ----------------------------------------------------

@dataclass
class LATPolicy(tel.TelPolicy):
    platform: str = "lat"
    corotator_override: float = None
    elevations_under_90: bool = False
    apply_corotator_rot: bool = True
    radius: float = 0.3
    corotator_offset: float = 0.0
    corotator_bounds: list = field(default_factory=lambda: [-45.0, 45.0])
    el_freq: float = 0.0
    run_stimulator: bool = False
    open_shutter: bool = False
    close_shutter: bool = False
    det_setup_duration: float = 20.0*u.minute
    remove_cmb_targets: Optional[Tuple] = ()
    remove_cal_targets: Optional[Tuple] = ()

    def __post_init__(self):
        self.blocks = self.make_blocks('lat-cmb')
        self.geometries = self.make_geometry()
        self.operations = self.make_operations()

    def apply_overrides(self, blocks):

        if self.elevations_under_90:
            def fix_block(b):
                if b.alt > 90:
                    return b.replace(alt=180-b.alt, az=b.az-180)
                return b
            blocks = core.seq_map( fix_block, blocks)

        if len(self.remove_cmb_targets) > 0:
            blocks = core.seq_filter_out(
                lambda b: b.name in self.remove_cmb_targets,
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
                    corotator_angle=el_to_locked_corotator(b.alt)
                ), blocks
            )

        blocks = core.seq_map(
            lambda b: b.replace(
                boresight_angle=corotator_to_boresight(b.alt, b.corotator_angle)
            ), blocks
        )
        blocks = core.seq_map(
            lambda b: b.replace(
                az_speed=round( b.az_speed/np.cos(np.radians(b.alt if b.alt <= 90 else 180 - b.alt)),2),
            ), blocks
        )

        return super().apply_overrides(blocks)

    def make_geometry(self):
        logger.info(f"making geometry with xi offset={self.xi_offset}, eta offset={self.eta_offset}, radius={self.radius}")
        return {
            "c1_ws0": {"center": [-0.3710+self.xi_offset, 0.0000+self.eta_offset], "radius": self.radius,}, # uhf
            "c1_ws1": {"center": [ 0.1815+self.xi_offset, 0.3211+self.eta_offset], "radius": self.radius,}, # uhf
            "c1_ws2": {"center": [ 0.1815+self.xi_offset, -0.3211+self.eta_offset], "radius": self.radius,}, # uhf
            "i1_ws0": {"center": [-1.9112+self.xi_offset, -0.9052+self.eta_offset], "radius": self.radius,}, # mf
            "i1_ws1": {"center": [-1.3584+self.xi_offset, -0.5704+self.eta_offset], "radius": self.radius,}, # mf
            "i1_ws2": {"center": [-1.3587+self.xi_offset, -1.2133+self.eta_offset], "radius": self.radius,}, # mf
            "i2_ws0": {"center": [-0.36415+self.xi_offset, -1.78324+self.eta_offset], "radius": self.radius,}, # uhf
            "i2_ws1": {"center": [0.18876+self.xi_offset, -1.46305+self.eta_offset], "radius": self.radius,}, # uhf
            "i2_ws2": {"center": [0.19272+self.xi_offset, -2.10348+self.eta_offset], "radius": self.radius,}, # uhf
            "i3_ws0": {"center": [ 1.1865+self.xi_offset, -0.8919+self.eta_offset], "radius": self.radius,}, # mf
            "i3_ws1": {"center": [ 1.7326+self.xi_offset, -0.5705+self.eta_offset], "radius": self.radius,}, # mf
            "i3_ws2": {"center": [ 1.7333+self.xi_offset, -1.2135+self.eta_offset], "radius": self.radius,}, # mf
            "i4_ws0": {"center": [ 1.1732+self.xi_offset, 0.9052+self.eta_offset], "radius": self.radius,}, # mf
            "i4_ws1": {"center": [ 1.7332+self.xi_offset, 1.2135+self.eta_offset], "radius": self.radius,}, # mf
            "i4_ws2": {"center": [ 1.7326+self.xi_offset, 0.5705+self.eta_offset], "radius": self.radius,}, # mf
            "i5_ws0": {"center": [-0.3655+self.xi_offset, 1.7833+self.eta_offset], "radius": self.radius,}, # uhf
            "i5_ws1": {"center": [ 0.1879+self.xi_offset, 2.1045+self.eta_offset], "radius": self.radius,}, # uhf
            "i5_ws2": {"center": [ 0.1867+self.xi_offset, 1.4620+self.eta_offset], "radius": self.radius,}, # uhf
            "i6_ws0": {"center": [-1.9082+self.xi_offset, 0.8920+self.eta_offset], "radius": self.radius,}, # mf
            "i6_ws1": {"center": [-1.3577+self.xi_offset, 1.2133+self.eta_offset], "radius": self.radius,}, # mf
            "i6_ws2": {"center": [-1.3584+self.xi_offset, 0.5854+self.eta_offset], "radius": self.radius,}, # mf
            "o1_ws0": {"center": [-1.89594+self.xi_offset, -2.67462+self.eta_offset], "radius": self.radius,}, # uhf
            "o1_ws1": {"center": [-1.34547+self.xi_offset, -2.35298+self.eta_offset], "radius": self.radius,}, # uhf
            "o1_ws2": {"center": [-1.33923+self.xi_offset, -2.99545+self.eta_offset], "radius": self.radius,}, # uhf
            "o2_ws0": {"center": [1.18755+self.xi_offset, -2.67467+self.eta_offset], "radius": self.radius,}, # mf
            "o2_ws1": {"center": [1.74466+self.xi_offset, -2.35369+self.eta_offset], "radius": self.radius,}, # mf
            "o2_ws2": {"center": [1.75046+self.xi_offset, -2.99649+self.eta_offset], "radius": self.radius,}, # mf
            "o3_ws0": {"center": [2.73022+self.xi_offset, 2e-05+self.eta_offset], "radius": self.radius,}, # mf
            "o3_ws1": {"center": [3.2929+self.xi_offset, 0.32195+self.eta_offset], "radius": self.radius,}, # mf
            "o3_ws2": {"center": [3.2929+self.xi_offset, -0.32193+self.eta_offset], "radius": self.radius,}, # mf
            "o4_ws0": {"center": [1.18755+self.xi_offset, 2.6747+self.eta_offset], "radius": self.radius,}, # mf
            "o4_ws1": {"center": [1.75045+self.xi_offset, 2.99652+self.eta_offset], "radius": self.radius,}, # mf
            "o4_ws2": {"center": [1.74467+self.xi_offset, 2.35372+self.eta_offset], "radius": self.radius,}, # mf
            "o5_ws0": {"center": [-1.89594+self.xi_offset, 2.67466+self.eta_offset], "radius": self.radius,}, # mf
            "o5_ws1": {"center": [-1.33923+self.xi_offset, 2.99547+self.eta_offset], "radius": self.radius,}, # mf
            "o5_ws2": {"center": [-1.34546+self.xi_offset, 2.353+self.eta_offset], "radius": self.radius,}, # mf
            "o6_ws0": {"center": [-3.43694+self.xi_offset, 2e-05+self.eta_offset], "radius": self.radius,}, # lf
            "o6_ws1": {"center": [-2.88688+self.xi_offset, 0.32179+self.eta_offset], "radius": self.radius,}, # lf
            "o6_ws2": {"center": [-2.88688+self.xi_offset, -0.32176+self.eta_offset], "radius": self.radius,}, # lf
        }

    def make_operations(self):
        cal_ops = []
        cmb_ops = []
        post_session_ops = []
        pre_session_ops = [
            {
                'name': 'lat.preamble',
                'sched_mode': SchedMode.PreSession,
                'open_shutter': self.open_shutter
            },
            {
                'name': 'start_time',
                'sched_mode': SchedMode.PreSession
            },
            {
                'name': 'set_scan_params',
                'sched_mode': SchedMode.PreSession,
                'az_speed': self.az_speed,
                'az_accel': self.az_accel,
                'el_freq': self.el_freq,
                'az_motion_override': self.az_motion_override,
                'el_mode_override': self.el_mode_override,
            },
        ]

        ops = [pre_session_ops, cmb_ops, cal_ops]
        sched_modes = [SchedMode.PreSession, SchedMode.PreObs, SchedMode.PreCal]

        if self.relock_cadence is not None:
            for op, sched_mode in zip(ops, sched_modes):
                op += [
                    {
                        'name': 'lat.ufm_relock',
                        'sched_mode': sched_mode,
                        'relock_cadence': self.relock_cadence
                    }
                ]

        ops = [cmb_ops, cal_ops]
        sched_modes = [SchedMode.PreObs, SchedMode.PreCal]

        for op, sched_mode in zip(ops, sched_modes):
            op += [
                {
                    'name': 'lat.setup_corotator',
                    'sched_mode': sched_mode,
                    'apply_corotator_rot': self.apply_corotator_rot,
                    'cryo_stabilization_time': self.cryo_stabilization_time,
                    'corotator_offset': self.corotator_offset,
                    'corotator_bounds': self.corotator_bounds,
                },
                {
                    'name': 'lat.det_setup',
                    'sched_mode': sched_mode,
                    'apply_corotator_rot': self.apply_corotator_rot,
                    'iv_cadence': self.iv_cadence,
                    'det_setup_duration': self.det_setup_duration,
                }
            ]

        cmb_ops += [
            {
                'name': 'lat.bias_step',
                'sched_mode': SchedMode.PreObs,
                'bias_step_cadence': self.bias_step_cadence
            },
        ]

        if self.run_stimulator:
            cmb_ops += [
                {
                    'name': 'lat.stimulator',
                    'sched_mode': SchedMode.PreObs,
                },
            ]
        cmb_ops += [
            {
                'name': 'lat.cmb_scan',
                'sched_mode': SchedMode.InObs
            },
        ]

        if self.run_stimulator:
            cal_ops += [
                {
                    'name': 'lat.stimulator',
                    'sched_mode': SchedMode.PreCal,
                },
            ]
        cal_ops += [
            {
                'name': 'lat.source_scan',
                'sched_mode': SchedMode.InCal
            },
        ]
        cal_ops += [
            {
                'name': 'lat.bias_step',
                'sched_mode': SchedMode.PostCal,
                'bias_step_cadence': self.bias_step_cadence
            },
        ]

        if self.run_stimulator:
             cal_ops += [
                {
                    'name': 'lat.stimulator',
                    'sched_mode': SchedMode.PostCal,
                },
             ]

        post_session_ops = [
            {
                'name': 'lat.wrap_up',
                'sched_mode': SchedMode.PostSession,
                'close_shutter': self.close_shutter,
            },
        ]

        return pre_session_ops + cal_ops + cmb_ops + post_session_ops

    def init_state(self, t0: dt.datetime) -> State:
        """
        Customize typical initial state, if needed

        Parameters
        ----------
        t0 : datetime.datetime
            The start time of the sequences.

        Returns
        -------
        lat.State :
            The initial LAT State object
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
            el_now=40,
            corotator_now=0,
        )

    def init_cal_seqs(self, blocks, t0, t1):
        """
        Initialize the cal sequences for the scheduler to process.

        Parameters
        ----------
        blocks : core.BlocksTree:
            The CMB block sequence from init_cmb_seq.
        t0 : datetime.datetime
            The start time of the sequences.
        t1 : datetime.datetime
            The end time of the sequences.

        Returns
        -------
        BlocksTree (nested dict / list of blocks)
            The initialized CMB and cal sequences
        """

        # get cal targets
        if self.cal_plan is not None:
            cal_targets = inst.parse_cal_targets_from_toast_lat(self.cal_plan)
            # keep all cal targets within range (don't restrict cal_target.t1 to t1 so we can keep partial scans)
            cal_targets[:] = [cal_target for cal_target in cal_targets if cal_target.t0 >= t0 and cal_target.t0 < t1]
        else:
            cal_targets = []

        for i, cal_target in enumerate(cal_targets):
            if cal_target.el_bore > 90:
                if self.elevations_under_90:
                    cal_targets[i] = replace(cal_targets[i], el_bore=180-cal_targets[i].el_bore)

            if self.corotator_override is None:
                corotator = el_to_locked_corotator(cal_targets[i].el_bore)
                boresight = corotator_to_boresight(cal_targets[i].el_bore, corotator)
            else:
                boresight = corotator_to_boresight(cal_target.el_bore, float(self.corotator_override))
            cal_targets[i] = replace(cal_targets[i], boresight_rot=boresight)

            if self.az_branch_override is not None:
                cal_targets[i] = replace(cal_targets[i], az_branch=self.az_branch_override)
                cal_targets[i] = replace(cal_targets[i], drift=self.drift_override)

        self.cal_targets += [target for target in cal_targets if target is not None]
        self.cal_targets = [cal_target for cal_target in self.cal_targets if cal_target.source not in self.remove_cal_targets]

        # by default add calibration blocks specified in cal_targets if not already specified
        for cal_target in self.cal_targets:
            if isinstance(cal_target, CalTarget):
                source = cal_target.source
                if source not in blocks['calibration']:
                    blocks['calibration'][source] = src.source_gen_seq(source, t0, t1)

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
        #   - likely won't affect scan blocks because ref plan already
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

            # override el mode
            if self.el_mode_override is not None:
                cal_block = cal_block.replace(
                    el_mode=self.el_mode_override
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
            blocks = az_range(blocks)

        # -----------------------------------------------------------------
        # step 4: tags
        # -----------------------------------------------------------------

        # add proper subtypes
        blocks['calibration'] = core.seq_map(
            lambda block: block.replace(subtype="cal"), blocks['calibration']
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

        # add scan type
        blocks = core.seq_map(
            lambda block: block.replace(tag=f"{block.tag},type{block.scan_type}"),
            blocks
        )

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

                assert block.alt <= alt_limits[1], (
                f"Block {block} is above the maximum elevation "
                f"of {alt_limits[1]} degrees."
                )

        return blocks

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
            A tree-like sequence of Blocks representing the observation schedule.
        t0 : datetime.datetime
            The starting datetime for the command sequence.
        t1 : datetime.datetime
            The ending datetime for the command sequence.
        state : Optional[State], optional
            The initial state of the observatory, by default None.

        Returns
        -------
        ops : list
            The list of operations.
        """
        if state is None:
            state = self.init_state(t0)

        # load building stage
        build_op = get_build_stage('build_op', {'policy_config': self, **self.stages.get('build_op', {})})

        # first resolve overlapping between cal and cmb
        cal_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'cal', seq))
        cmb_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'cmb', seq))
        seq = core.seq_sort(core.seq_merge(cmb_blocks, cal_blocks, flatten=True))

        # divide cmb blocks
        if self.max_cmb_scan_duration is not None:
            seq = core.seq_flatten(
                core.seq_map(
                    lambda b: self.divide_blocks(b, dt.timedelta(seconds=self.max_cmb_scan_duration))
                    if b.subtype == 'cmb'
                    else b,
                    seq,
                )
            )

        # compile operations
        cal_pre = [op for op in self.operations if op['sched_mode'] == SchedMode.PreCal]
        cal_in = [op for op in self.operations if op['sched_mode'] == SchedMode.InCal]
        cal_post = [op for op in self.operations if op['sched_mode'] == SchedMode.PostCal]
        cmb_pre = [op for op in self.operations if op['sched_mode'] == SchedMode.PreObs]
        cmb_in = [op for op in self.operations if op['sched_mode'] == SchedMode.InObs]
        cmb_post = [op for op in self.operations if op['sched_mode'] == SchedMode.PostObs]
        pre_sess = [op for op in self.operations if op['sched_mode'] == SchedMode.PreSession]
        pos_sess = [op for op in self.operations if op['sched_mode'] == SchedMode.PostSession]

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
            else:
                raise ValueError(f"unexpected block subtype: {block.subtype}")

        seq = [map_block(b) for b in seq]

        start_block = {
            'name': 'pre-session',
            'block': inst.StareBlock(name="pre-session", az=state.az_now, alt=state.el_now, az_offset=self.az_offset,
                                    alt_offset=self.el_offset, t0=t0, t1=t0+dt.timedelta(seconds=1)),
            'pre': [],
            'in': [],
            'post': pre_sess, # scheduled after t0
            'priority': 0, # for now cal, wiregrid, presession and postsession must have the same priority
            'pinned': True # remain unchanged during multi-pass
        }
        # move to a stow position if specified, otherwise find a stow position or stay in final position
        if len(pos_sess) > 0:
            # find an alt, az that is sun-safe for the entire duration of the schedule.
            if all(self.stow_position.get(k) is not None for k in ("az_stow", "el_stow")):
                az_stow = self.stow_position['az_stow']
                alt_stow = self.stow_position['el_stow']
            else:
                az_start = 180
                alt_start = self.elevation_override if self.elevation_override is not None else min(60.0, self.stages['build_op']['plan_moves']['el_limits'][1])
                # add a buffer to start and end to be safe
                if len(seq) > 0:
                    t_start = seq[-1]['block'].t1 - dt.timedelta(seconds=300)
                else:
                    t_start = t0 - dt.timedelta(seconds=3600)
                t_end = t1 + dt.timedelta(seconds=3600)
                az_stow, alt_stow, _, _ = get_parking(t_start, t_end, alt_start, self.stages['build_op']['plan_moves']['sun_policy'])
                logger.info(f"found sun safe stow position at ({az_stow}, {alt_stow})")
        elif len(seq) > 0:
            az_stow = seq[-1]['block'].az
            alt_stow = seq[-1]['block'].alt
        else:
            az_stow = state.az_now
            alt_stow = state.el_now
        end_block = {
            'name': 'post-session',
            'block': inst.StareBlock(name="post-session", az=az_stow, alt=alt_stow, az_offset=self.az_offset,
                                    alt_offset=self.el_offset, t0=t1-dt.timedelta(seconds=1), t1=t1),
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

    def add_cal_target(
        self,
        source: str,
        corotator: int,
        elevation: int,
        focus: str,
        allow_partial=False,
        drift=True,
        az_branch=None,
        az_speed=None,
        az_accel=None,
        source_direction=None
    ):
        array_focus = {
            'c1': 'c1_ws0,c1_ws1,c1_ws2', # uhf
            'i1': 'i1_ws0,i1_ws1,i1_ws2', # mf
            'i2': 'i2_ws0,i2_ws1,i2_ws2', # uhf
            'i3': 'i3_ws0,i3_ws1,i3_ws2', # mf
            'i4': 'i4_ws0,i4_ws1,i4_ws2', # mf
            'i5': 'i5_ws0,i5_ws1,i5_ws2', # uhf
            'i6': 'i6_ws0,i6_ws1,i6_ws2', # mf
            'o1': 'o1_ws0,o1_ws1,o1_ws2', # uhf
            'o2': 'o2_ws0,o2_ws1,o2_ws2', # mf
            'o3': 'o3_ws0,o3_ws1,o3_ws2', # mf
            'o4': 'o4_ws0,o4_ws1,o4_ws2', # mf
            'o5': 'o5_ws0,o5_ws1,o5_ws2', # mf
            'o6': 'o6_ws0,o6_ws1,o6_ws2', # lf
        }

        array_focus['all'] = ','.join([v for k, v in array_focus.items()])

        elevation = float(elevation)
        if corotator is None:
            corotator = el_to_locked_corotator(elevation)
        boresight = corotator_to_boresight(elevation, float(corotator))

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

        self.cal_targets.append(
            CalTarget(
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
        )
