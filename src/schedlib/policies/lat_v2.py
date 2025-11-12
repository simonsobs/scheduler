import numpy as np
import datetime as dt
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Union, Optional, Dict, Any, Tuple

from .. import utils as u


logger = u.init_logger(__name__)

# ----------------------------------------------------
#                  LAT Command Durations
# ----------------------------------------------------

COROTATOR_DURATION = 1*u.minute
STIMULATOR_DURATION = 4*u.minute

# ----------------------------------------------------
#                   Utilities
# ----------------------------------------------------

def el_to_locked_corotator(el):
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


def corotator_to_boresight(el, corotator):
    """
    Calculates the boresight angle (-roll) from the corotator angle.
    When the corotator is locked, The corotator angle is determined from
    the function ``el_to_locked_corotator``, otherwise it can be any value
    between -45 and 45 degrees.

    Parameters
    ----------
    el : float
        The elevation of the boresight in degrees.
    corotator : float
        The rotation angle of the corotator in degrees.
    """
    return -(el - 60 - corotator)


def boresight_to_corotator(el, boresight):
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
#                  SAT Operations
# ----------------------------------------------------

@cmd.operation(name="lat.preamble", duration=0)
def preamble(open_shutter=False):
    cmd = tel.preamble()
    cmd += ["acu.clear_faults()"]
    if open_shutter:
        cmd += ["acu.stop_and_clear()",
                "run.acu.set_shutter(action='open')"
            ]
    return cmd

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

class LATPolicy(tel.TelPolicy):
    corotator_override: float = None
    elevations_under_90: bool = False
    apply_corotator_rot: bool = True
    corotator_offset: float = 0.0
    open_shutter: bool = False
    close_shutter: bool = False
    remove_cmb_targets: Optional[Tuple] = ()
    remove_cal_targets: Optional[Tuple] = ()

    def __post_init__(self):
        self.blocks = self.make_blocks('lat-cmb')
        self.geometries = self.make_geometry()
        self.operations = self.make_operations()

    def make_geometry(self):
        logger.info(f"making geometry with xi offset={self.xi_offset}, eta offset={self.eta_offset}")
        radius = 0.3
        return {
            "c1_ws0": {"center": [-0.3710+self.xi_offset, 0+self.eta_offset], "radius": radius,},
            "c1_ws1": {"center": [ 0.1815+self.xi_offset, 0.3211+self.eta_offset], "radius": radius,},
            "c1_ws2": {"center": [ 0.1815+self.xi_offset,-0.3211+self.eta_offset], "radius": radius,},
            "i1_ws0": {"center": [-1.9112+self.xi_offset,-0.9052+self.eta_offset], "radius": radius,},
            "i1_ws1": {"center": [-1.3584+self.xi_offset,-0.5704+self.eta_offset], "radius": radius,},
            "i1_ws2": {"center": [-1.3587+self.xi_offset,-1.2133+self.eta_offset], "radius": radius,},
            "i3_ws0": {"center": [ 1.1865+self.xi_offset,-0.8919+self.eta_offset], "radius": radius,},
            "i3_ws1": {"center": [ 1.7326+self.xi_offset,-0.5705+self.eta_offset], "radius": radius,},
            "i3_ws2": {"center": [ 1.7333+self.xi_offset,-1.2135+self.eta_offset], "radius": radius,},
            "i4_ws0": {"center": [ 1.1732+self.xi_offset, 0.9052+self.eta_offset], "radius": radius,},
            "i4_ws1": {"center": [ 1.7332+self.xi_offset, 1.2135+self.eta_offset], "radius": radius,},
            "i4_ws2": {"center": [ 1.7326+self.xi_offset, 0.5705+self.eta_offset], "radius": radius,},
            "i5_ws0": {"center": [-0.3655+self.xi_offset, 1.7833+self.eta_offset], "radius": radius,},
            "i5_ws1": {"center": [ 0.1879+self.xi_offset, 2.1045+self.eta_offset], "radius": radius,},
            "i5_ws2": {"center": [ 0.1867+self.xi_offset, 1.4620+self.eta_offset], "radius": radius,},
            "i6_ws0": {"center": [-1.9082+self.xi_offset, 0.8920+self.eta_offset], "radius": radius,},
            "i6_ws1": {"center": [-1.3577+self.xi_offset, 1.2133+self.eta_offset], "radius": radius,},
            "i6_ws2": {"center": [-1.3584+self.xi_offset, 0.5854+self.eta_offset], "radius": radius,},
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
                'az_motion_override': self.az_motion_override
            },
        ]

        ops = [cmb_ops, cal_ops]
        sched_modes = [SchedMode.PreObs, SchedMode.PreCal]

        if self.relock_cadence is not None:
            for op, sched_mode in zip(ops, sched_mode):
                op += [
                    {
                        'name': 'lat.ufm_relock',
                        'sched_mode': sched_mode,
                        'relock_cadence': self.relock_cadence
                    }
                ]

        for op, sched_mode in zip(ops, sched_modes):
            op += [
                {
                    'name': 'lat.setup_corotator',
                    'sched_mode': sched_mode,
                    'apply_boresight_rot': self.apply_corotator_rot,
                    'cryo_stabilization_time': self.cryo_stabilization_time,
                    'corotator_offset': self.corotator_offset,
                },
                {
                    'name': 'lat.det_setup',
                    'sched_mode': sched_mode,
                    'apply_corotator_rot': self.apply_corotator_rot,
                    'iv_cadence': self.iv_cadence,
                    'det_setup_duration': det_setup_duration,
                }
            ]

            if self.run_stimulator:
                op += [
                    {
                        'name': 'lat.stimulator',
                        'sched_mode': sched_mode,
                    },
                ]

        cmb_ops += [
            {
                'name': 'lat.bias_step',
                'sched_mode': SchedMode.PreObs,
                'bias_step_cadence': bias_step_cadence
            },
            {
                'name': 'lat.cmb_scan',
                'sched_mode': SchedMode.InObs
            },
        ]

        cal_ops += [
            {
                'name': 'lat.source_scan',
                'sched_mode': SchedMode.InObs
            },
            {
                'name': 'lat.bias_step',
                'sched_mode': SchedMode.PostCal,
                'bias_step_cadence': bias_step_cadence
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

        # cal
        cal_ops += [
            { 'name': 'lat.setup_corotator' , 'sched_mode': SchedMode.PreCal, 'apply_corotator_rot': apply_corotator_rot,
            'cryo_stabilization_time': cryo_stabilization_time, 'corotator_offset': corotator_offset},
            { 'name': 'lat.det_setup'       , 'sched_mode': SchedMode.PreCal, 'apply_corotator_rot': apply_corotator_rot, 'iv_cadence':iv_cadence }
        ]
        if run_stimulator:
            cal_ops += [
                { 'name': 'lat.stimulator'      , 'sched_mode': SchedMode.PreCal, }
            ]
        cal_ops += [
            { 'name': 'lat.source_scan'     , 'sched_mode': SchedMode.InCal, },
            { 'name': 'lat.bias_step'       , 'sched_mode': SchedMode.PostCal, 'bias_step_cadence': bias_step_cadence},
        ]
        if run_stimulator:
            cal_ops += [
                { 'name': 'lat.stimulator'      , 'sched_mode': SchedMode.PostCal, }
            ]
        # cmb
        cmb_ops += [
            { 'name': 'lat.setup_corotator' , 'sched_mode': SchedMode.PreObs, 'apply_corotator_rot': apply_corotator_rot,
            'cryo_stabilization_time': cryo_stabilization_time, 'corotator_offset': corotator_offset},
            { 'name': 'lat.det_setup'       , 'sched_mode': SchedMode.PreObs, 'apply_corotator_rot': apply_corotator_rot, 'iv_cadence':iv_cadence},
            { 'name': 'lat.bias_step'       , 'sched_mode': SchedMode.PreObs, 'bias_step_cadence': bias_step_cadence},
        ]
        if run_stimulator:
            cmb_ops += [
                { 'name': 'lat.stimulator'      , 'sched_mode': SchedMode.PreObs, }
            ]
        cmb_ops += [
            { 'name': 'lat.cmb_scan'        , 'sched_mode': SchedMode.InObs, },
        ]

        post_session_ops = [
            { 'name': 'lat.wrap_up'   , 'sched_mode': SchedMode.PostSession, 'close_shutter': close_shutter},
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
            cal_targets = parse_cal_targets_from_toast_lat(self.cal_plan)
            # keep all cal targets within range (don't restrict cal_target.t1 to t1 so we can keep partial scans)
            cal_targets[:] = [cal_target for cal_target in cal_targets if cal_target.t0 >= t0 and cal_target.t0 < t1]

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

            self.cal_targets += cal_targets

        self.cal_targets = [cal_target for cal_target in self.cal_targets if cal_target.source not in self.remove_cal_targets]

        # by default add calibration blocks specified in cal_targets if not already specified
        for cal_target in self.cal_targets:
            if isinstance(cal_target, CalTarget):
                source = cal_target.source
                if source not in blocks['calibration']:
                    blocks['calibration'][source] = src.source_gen_seq(source, t0, t1)

        return blocks


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
            'c1' : 'c1_ws0,c1_ws1,c1_ws2',
            'i1' : 'i1_ws0,i1_ws1,i1_ws2',
            'i3' : 'i3_ws0,i3_ws1,i3_ws2',
            'i4' : 'i4_ws0,i4_ws1,i4_ws2',
            'i5' : 'i5_ws0,i5_ws1,i5_ws2',
            'i6' : 'i6_ws0,i6_ws1,i6_ws2',
        }

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