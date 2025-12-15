import numpy as np
import datetime as dt
from dataclasses import dataclass, field, replace
from dataclasses_json import dataclass_json
from typing import List, Union, Optional, Dict, Any, Tuple
from functools import reduce

from .. import core, utils as u, source as src, rules as ru
from .. import commands as cmd, instrument as inst
from ..thirdparty import SunAvoidance
from .stages import get_build_stage
from .stages.build_op import get_parking
from . import tel
from .tel import SchedMode
from ..instrument import CalTarget, StareBlock, WiregridTarget


logger = u.init_logger(__name__)


# ----------------------------------------------------
#                  SAT Command Durations
# ----------------------------------------------------

HWP_SPIN_UP = 7*u.minute
HWP_SPIN_DOWN = 15*u.minute
BORESIGHT_DURATION = 1*u.minute

COMMANDS_HWP_BRAKE = [
    "run.hwp.spin_down(active=True)",
    "",
]
COMMANDS_HWP_STOP = [
    "run.hwp.spin_down(active=False)",
    "",
]


# ----------------------------------------------------
#                   Utilities
# ----------------------------------------------------

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



# ----------------------------------------------------
#                  SAT State
# ----------------------------------------------------

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


# ----------------------------------------------------
#                  SAT SchedMode
# ----------------------------------------------------

class SchedMode(tel.SchedMode):
    """
    Enumeration of scheduling modes for SATPolicy operations.

    Attributes
    ----------
    PreWiregrid : str
        'pre_wiregrid'; scheduling mode for wiregrid operations
        that occur before the main wiregrid block.
    Wiregrid : str
        'wiregrid'; scheduling mode for wiregrid observations
        between block.t0 and block.t1.
    """
    PreWiregrid = 'pre_wiregrid'
    Wiregrid = 'wiregrid'


# ----------------------------------------------------
#                  SAT Operations
# ----------------------------------------------------

@cmd.operation(name="sat.preamble", return_duration=True)
def preamble(state):
    base = tel.preamble()
    append = [
        "################### Basic Checks ###################",
        "acu_data = acu.monitor.status().session['data']",
        "hwp_state = run.CLIENTS['hwp'].monitor.status().session['data']['hwp_state']",
        "",
        f"assert np.round(acu_data['StatusDetailed']['Elevation current position'], 1) == {state.el_now}, 'Elevation check failed'",
        f"assert np.round(acu_data['StatusDetailed']['Boresight current position'], 2) == {state.boresight_rot_now}, 'Boresight angle check failed'",
        f"assert hwp_state['is_spinning'] == {state.hwp_spinning}, 'HWP spinning check failed'",
    ]
    if state.hwp_spinning:
        append += [
            f"assert (hwp_state['direction'] == 'ccw') == {state.hwp_dir}, 'HWP direction check failed'",
        ]
    else:
        append += [
            f"assert hwp_state['gripper']['grip_state'] == 'ungripped', 'HWP gripper check failed'",
        ]
    append += [
        "################### Checks  Over ###################",
        "",
        ]
    return state, 0, base + append

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
        f"run.hwp.spin_up(freq={freq})"
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

@cmd.operation(name='sat.bias_step', return_duration=True)
def bias_step(state, block, bias_step_cadence=None):
    return tel.bias_step(state, block, bias_step_cadence)

@cmd.operation(name='sat.wiregrid', return_duration=True)
def wiregrid(state, block, min_wiregrid_el=49.9):
    assert state.hwp_spinning == True, "hwp is not spinning"
    assert block.alt >= min_wiregrid_el, f"Block {block} is below the minimum wiregrid elevation of {min_wiregrid_el} degrees."

    if block.name == 'wiregrid_gain':
        return state, block.duration.total_seconds(), [
            "run.wiregrid.calibrate(continuous=False, elevation_check=True, boresight_check=False, temperature_check=False)"
        ]
    elif block.name == 'wiregrid_time_const':
        # wiregrid time constant measurement reverses the hwp direction
        state = state.replace(hwp_dir=not state.hwp_dir)
        direction = "ccw (positive frequency)" if state.hwp_dir \
                else "cw (negative frequency)"
        return state, block.duration.total_seconds(), [
            "run.wiregrid.time_constant(num_repeats=1)",
            f"# hwp direction reversed, now spinning " + direction,
            ]

@cmd.operation(name="move_to", return_duration=True)
def move_to(state, az, el, az_offset=0, el_offset=0, min_el=48, max_el=60, brake_hwp=True, force=False):
    if not force and (state.az_now == az and state.el_now == el):
        return state, 0, []

    duration = 0
    cmd = []

    if state.hwp_spinning and (el < min_el or el > max_el):
        state = state.replace(hwp_spinning=False)
        duration += HWP_SPIN_DOWN
        cmd += COMMANDS_HWP_BRAKE if brake_hwp else COMMANDS_HWP_STOP
    cmd += [
        f"run.acu.move_to(az={round(az + az_offset, 3)}, el={round(el + el_offset, 3)})",
    ]
    state = state.replace(az_now=az, el_now=el)

    return state, duration, cmd


# ----------------------------------------------------
#                  Base SAT Policy
# ----------------------------------------------------

@dataclass
class SATPolicy(tel.TelPolicy):
    wiregrid_plan: str = None
    hwp_override: bool = None
    brake_hwp: bool = True
    disable_hwp: bool = False
    min_hwp_el: float = 40.0
    max_hwp_el: float = 60.0
    force_max_hwp_el: bool = True
    boresight_override: float = None
    apply_boresight_rot: bool = True
    det_setup_duration: float = 20.0*u.minute
    wiregrid_az: float = 180.0 # deg
    wiregrid_el: float = 50.0 # deg
    ignore_wafers: list[str] = None

    def __post_init__(self):
        self.blocks = self.make_blocks('sat-cmb')
        self.geometries = self.make_geometry()
        self.operations = self.make_operations()

        if self.elevation_override is not None:
            self.stages["build_op"]["plan_moves"]["el_limits"] = 2*[self.elevation_override]
        elif self.force_max_hwp_el and self.max_hwp_el is not None:
            self.stages["build_op"]["plan_moves"]["el_limits"][1] = self.max_hwp_el

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

    def make_geometry(self):
        logger.info(f"making geometry with xi offset={self.xi_offset}, eta offset={self.eta_offset}")
        # default SAT optics offsets
        d_xi = 10.9624
        d_eta_side = 6.46363
        d_eta_mid = 12.634
        radius = 6.0

        return {
            'ws0': {
                'center': [0 + self.xi_offset, 0 + self.eta_offset],
                'radius': radius,
            },
            'ws1': {
                'center': [0 + self.xi_offset, -d_eta_mid + self.eta_offset],
                'radius': radius,
            },
            'ws2': {
                'center': [-d_xi + self.xi_offset, -d_eta_side + self.eta_offset],
                'radius': radius,
            },
            'ws3': {
                'center': [-d_xi + self.xi_offset, d_eta_side + self.eta_offset],
                'radius': radius,
            },
            'ws4': {
                'center': [0 + self.xi_offset, d_eta_mid + self.eta_offset],
                'radius': radius,
            },
            'ws5': {
                'center': [d_xi + self.xi_offset, d_eta_side + self.eta_offset],
                'radius': radius,
            },
            'ws6': {
                'center': [d_xi + self.xi_offset, -d_eta_side + self.eta_offset],
                'radius': radius,
            },
        }

    def make_operations(self, hwp_cfg=None, cmds_uxm_relock=None, cmds_det_setup=None):
        cmb_ops = []
        cal_ops = []
        wiregrid_ops = []
        post_session_ops = []

        if hwp_cfg is None:
            hwp_cfg = {
                'gripper': 'hwp-gripper',
                'hwp-pmx': 'pmx',
                'iboot2': 'power-iboot-hwp-2',
                'pid': 'hwp-pid',
                'pmx': 'hwp-pmx',
            }
        pre_session_ops = [
            {
                'name': 'sat.preamble',
                'sched_mode': SchedMode.PreSession,
            },
            {
                'name': 'start_time',
                'sched_mode': SchedMode.PreSession,
            },
            {
                'name': 'set_scan_params',
                'sched_mode': SchedMode.PreSession,
                'az_speed': self.az_speed,
                'az_accel': self.az_accel,
                'az_motion_override': self.az_motion_override,
            },
        ]

        ops = [pre_session_ops, cmb_ops, cal_ops]
        sched_modes = [SchedMode.PreSession, SchedMode.PreObs, SchedMode.PreCal]

        if self.relock_cadence is not None:
            for op, sched_mode in zip(ops, sched_modes):
                op += [
                    {
                        'name': 'sat.ufm_relock',
                        'sched_mode': sched_mode,
                        'relock_cadence': self.relock_cadence,
                        'commands': cmds_uxm_relock,
                    }
                ]

        ops = [cmb_ops, cal_ops]
        sched_modes = [SchedMode.PreObs, SchedMode.PreCal]
        if not self.disable_hwp:
            ops += [wiregrid_ops]
            sched_modes += [SchedMode.PreWiregrid]

        for op, sched_mode in zip(ops, sched_modes):
            op += [
                {
                    'name': 'sat.setup_boresight',
                    'sched_mode': sched_mode,
                    'apply_boresight_rot': self.apply_boresight_rot,
                    'brake_hwp': self.brake_hwp,
                    'cryo_stabilization_time': self.cryo_stabilization_time,
                },
                {
                    'name': 'sat.det_setup',
                    'sched_mode': sched_mode,
                    'apply_boresight_rot': self.apply_boresight_rot,
                    'iv_cadence': self.iv_cadence,
                    'det_setup_duration': self.det_setup_duration,
                    'commands': cmds_det_setup,
                },
                {
                    'name': 'sat.hwp_spin_up',
                    'sched_mode': sched_mode,
                    'disable_hwp': self.disable_hwp,
                    'brake_hwp': self.brake_hwp,
                },
            ]

        cmb_ops += [
            {
                'name': 'sat.bias_step',
                'sched_mode': SchedMode.PreObs,
                'bias_step_cadence': self.bias_step_cadence
            },
            {
                'name': 'sat.cmb_scan',
                'sched_mode': SchedMode.InObs
            },
        ]

        cal_ops += [
            {
                'name': 'sat.source_scan',
                'sched_mode': SchedMode.InCal
            },
            {
                'name': 'sat.bias_step',
                'sched_mode': SchedMode.PostCal,
                'bias_step_cadence': self.bias_step_cadence
            },
        ]

        if not self.disable_hwp:
            wiregrid_ops += [
                {
                    'name': 'sat.wiregrid',
                    'sched_mode': SchedMode.Wiregrid
                },
            ]

        if self.home_at_end:
            post_session_ops += [
                {'name': 'sat.hwp_spin_down',
                'sched_mode': SchedMode.PostSession,
                'disable_hwp': self.disable_hwp,
                'brake_hwp': self.brake_hwp
                },
        ]
        post_session_ops += [
            {
                'name': 'sat.wrap_up',
                'sched_mode': SchedMode.PostSession
            },
        ]

        return pre_session_ops + cal_ops + cmb_ops + post_session_ops + wiregrid_ops

    def init_state(self, t0: dt.datetime) -> State:
        """
        Customize typical initial state, if needed

        Parameters
        ----------
        t0 : datetime.datetime
            The start time of the sequences.

        Returns
        -------
        sat.State :
            The initial SAT State object
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
            az_now=180.0,
            el_now=40.0,
            boresight_rot_now=0.0,
            hwp_spinning=False,
        )

    def init_cal_seqs(self, blocks, t0, t1):
        """
        Initialize the cal and wiregrid sequences for the scheduler to process.

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
            The initialized CMB, cal, and wiregrid sequences
        """
        # get cal targets
        if self.cal_plan is not None:
            cal_targets = inst.parse_cal_targets_from_toast_sat(self.cal_plan)
            # keep all cal targets within range (don't restrict cal_target.t1 to t1 so we can keep partial scans)
            cal_targets[:] = [cal_target for cal_target in cal_targets if cal_target.t0 >= t0 and cal_target.t0 < t1]
        else:
            cal_targets = []

        for i, cal_target in enumerate(cal_targets):
            # remove ignored wafers
            if self.ignore_wafers is not None:
                wafers = cal_target.array_query.split(',')
                filtered = [w for w in wafers if w not in self.ignore_wafers]
                if filtered:
                    cal_targets[i] = replace(cal_targets[i], array_query=",".join(filtered))
                else:
                    cal_targets[i] = None
                    continue

            # find nearest cmb block either before or after the cal target
            candidates = [block for block in blocks['baseline']['cmb'] if block.t0 < cal_target.t0]

            if candidates:
                block = max(candidates, key=lambda x: x.t0)
            else:
                candidates = [block for block in blocks['baseline']['cmb'] if block.t0 > cal_target.t0]
                if candidates:
                    block = min(candidates, key=lambda x: x.t0)
                else:
                    raise ValueError("Cannot find nearby CMB block")

            if cal_target.boresight_rot is None:
                candidates = [block for block in blocks['baseline']['cmb'] if block.t0 < cal_target.t0]

            # overrides
            if self.az_branch_override is not None:
                cal_targets[i] = replace(cal_targets[i], az_branch=self.az_branch_override)

            if self.elevation_override is not None:
                cal_targets[i] = replace(cal_targets[i], el_bore=self.elevation_override)

            if self.drift_override is not None:
                cal_targets[i] = replace(cal_targets[i], drift=self.drift_override)

        self.cal_targets += [target for target in cal_targets if target is not None]

        # get wiregrid plan
        if self.wiregrid_plan is not None and not self.disable_hwp:
            wiregrid_candidates = inst.parse_wiregrid_targets_from_file(self.wiregrid_plan)
            wiregrid_candidates[:] = [
                wg for wg in wiregrid_candidates
                if wg.t0 >= t0 and wg.t1 <= t1
            ]
            self.cal_targets += wiregrid_candidates

        wiregrid_candidates = []

        # by default add calibration blocks specified in cal_targets if not already specified
        for cal_target in self.cal_targets:
            if isinstance(cal_target, CalTarget):
                source = cal_target.source
                if source not in blocks['calibration']:
                    blocks['calibration'][source] = src.source_gen_seq(source, t0, t1)
            elif isinstance(cal_target, WiregridTarget):
                wiregrid_candidates.append(
                    StareBlock(
                        name=cal_target.name,
                        t0=cal_target.t0,
                        t1=cal_target.t1,
                        az=self.wiregrid_az,
                        alt=self.wiregrid_el,
                        tag='',
                        subtype='wiregrid',
                        hwp_dir=self.hwp_override if self.hwp_override is not None else None
                    )
                )
        blocks['calibration']['wiregrid'] = wiregrid_candidates

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

        for target in self.cal_targets:
            logger.info(f"-> planning calibration scans for {target}...")

            if isinstance(target, WiregridTarget):
                continue

            assert target.source in blocks['calibration'], f"source {target.source} not found in sequence"
            logger.info(f"trying array_query={target.array_query}")
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

            if target.array_query not in target.tag:
                tag = f"{cal_block.tag},{target.array_query},{target.tag}"
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

        blocks['calibration'] = cal_blocks + blocks['calibration']['wiregrid']

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
            blocks = az_range(blocks)

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
        wiregrid_blocks = core.seq_flatten(core.seq_filter(lambda b: b.subtype == 'wiregrid', seq))

        for i, wiregrid_block in enumerate(wiregrid_blocks):
            if core.seq_has_overlap_with_block(cal_blocks, wiregrid_block):
                logger.warn(f"wiregrid block {wiregrid_block} has overlap with cal scans. removing.")
                wiregrid_blocks[i] = None

        cal_blocks += wiregrid_blocks

        seq = core.seq_sort(core.seq_merge(cmb_blocks, cal_blocks, flatten=True))

        # divide cmb blocks
        if self.max_cmb_scan_duration is not None:
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

        start_block = {
            'name': 'pre-session',
            'block': inst.StareBlock(name="pre-session", az=state.az_now, alt=state.el_now, az_offset=self.az_offset,
                                    alt_offset=self.el_offset, t0=t0, t1=t0+dt.timedelta(seconds=1)),
            'pre': [],
            'in': [],
            'post': pre_sess,  # scheduled after t0
            'priority': -1, # for now cal, wiregrid, presession and postsession must have the same priority
            'pinned': True  # remain unchanged during multi-pass
        }

        # move to a stow position if specified, otherwise find a stow position or stay in final position
        if len(pos_sess) > 0:
            # find an alt, az that is sun-safe for the entire duration of the schedule.
            if all(self.stow_position.get(k) is not None for k in ("az_stow", "el_stow")):
                az_stow = self.stow_position['az_stow']
                alt_stow = self.stow_position['el_stow']
            else:
                az_start = 180.0
                alt_start = self.elevation_override if self.elevation_override is not None else 60.0
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
            'priority': -1,
            'pinned': True # remain unchanged during multi-pass
        }
        seq = [start_block] + seq + [end_block]

        ops, state = build_op.apply(seq, t0, t1, state)
        if return_state:
            return ops, state
        return ops

    def make_cal_target(
        self,
        array_focus: Dict[str, str],
        source: str,
        boresight: int,
        elevation: int,
        focus: str,
        allow_partial=False,
        drift=True,
        az_branch=None,
        az_speed=None,
        az_accel=None,
        source_direction=None
    ):

        if self.ignore_wafers is not None:
            keys_to_remove = []
            for key, val in array_focus.items():
                wafers = val.split(',')
                cleaned = [w for w in wafers if w not in self.ignore_wafers]
                if cleaned:
                    array_focus[key] = ','.join(cleaned)
                else:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                array_focus.pop(key)

        boresight = int(boresight)
        elevation = int(elevation)
        focus = focus.lower()

        focus_str = None
        focus_str = array_focus.get(focus, focus)

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
            from_table=False
        )
