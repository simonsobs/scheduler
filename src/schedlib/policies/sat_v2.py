import numpy as np
import datetime as dt
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Union, Optional, Dict, Any, Tuple

from .. import utils as u


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
    Enumerate different options for scheduling operations in SATPolicy.

    Attributes
    ----------
    Wiregrid : str
        'wiregrid'; Wiregrid observations scheduled between block.t0 and block.t1
    """
    PreWiregrid = 'pre_wiregrid'
    Wiregrid = 'wiregrid'


# ----------------------------------------------------
#                  SAT Operations
# ----------------------------------------------------

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
def wiregrid(state, block, min_wiregrid_el=49.9):
    assert state.hwp_spinning == True, "hwp is not spinning"
    assert block.alt >= min_wiregrid_el, f"Block {block} is below the minimum wiregrid elevation of {min_wiregrid_el} degrees."

    if block.name == 'wiregrid_gain':
        return state, block.duration.total_seconds(), [
            "run.wiregrid.calibrate(continuous=False, elevation_check=True, boresight_check=False, temperature_check=False)"
        ]
    elif block.name == 'wiregrid_time_const':
        # wiregrid time const reverses the hwp direction
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
class SATPolicy(TelPolicy):
    hwp_override: bool = None
    brake_hwp: bool = True
    disable_hwp: bool = False
    boresight_override: float = None
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

    def make_geometry(self):
        logger.info(f"making geometry with xi offset={self.xi_offset}, eta offset={self.eta_offset}")
        # default SAT optics offsets
        d_xi = 10.9624
        d_eta_side = 6.46363
        d_eta_mid = 12.634
        radius = 6

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

    def make_operatons(self):
        cmb_ops = []
        cal_ops = []
        wiregrid_ops = []
        post_session_ops = []

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

        ops = [cmb_ops, cal_ops]
        sched_modes = [SchedMode.PreObs, SchedMode.PreCal]

        if self.relock_cadence is not None:
            for op, sched_mode in zip(ops, sched_mode):
                op += [
                    {
                        'name': 'sat.ufm_relock',
                        'sched_mode': sched_mode,
                        'relock_cadence': self.relock_cadence
                    }
                ]

        if not self.disable_wiregrid:
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
                    'det_setup_duration': det_setup_duration,
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
                'bias_step_cadence': bias_step_cadence
            },
            {
                'name': 'sat.cmb_scan',
                'sched_mode': SchedMode.InObs
            },
        ]

        cal_ops += [
            {
                'name': 'sat.source_scan',
                'sched_mode': SchedMode.InObs
            },
            {
                'name': 'sat.bias_step',
                'sched_mode': SchedMode.PostCal,
                'bias_step_cadence': bias_step_cadence
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
                'brake_hwp': brake_hwp
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
            az_now=180,
            el_now=40,
            boresight_rot_now=0,
            hwp_spinning=False,
        )

    def init_cal_seq(self, t0, t1, blocks):
        """
        Initialize the cal and wiregrid sequences for the scheduler to process.

        Parameters
        ----------
        t0 : datetime.datetime
            The start time of the sequences.
        t1 : datetime.datetime
            The end time of the sequences.
        blocks : core.BlocksTree:
            The CMB block sequence from init_cmb_seq

        Returns
        -------
        BlocksTree (nested dict / list of blocks)
            The initialized CMB, cal, and wiregrid sequences
        """
        # get cal targets
        if self.cal_plan is not None:
            cal_targets = inst.parse_cal_targets_from_toast_sat(cfile)
            # keep all cal targets within range (don't restrict cal_target.t1 to t1 so we can keep partial scans)
            cal_targets[:] = [cal_target for cal_target in cal_targets if cal_target.t0 >= t0 and cal_target.t0 < t1]

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
            wiregrid_candidates = parse_wiregrid_targets_from_file(wgfile)
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
                        tag=cal_target.tag,
                        subtype='wiregrid',
                        hwp_dir=self.hwp_override if self.hwp_override is not None else None
                    )
                )
        blocks['calibration']['wiregrid'] = wiregrid_candidates

        return blocks

    def make_cal_target(
        self,
        array_focus: Dict[str],
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
