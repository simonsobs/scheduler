import numpy as np
from dataclasses import dataclass
import datetime as dt

from typing import Optional

from .. import source as src, utils as u
from .sat import SATPolicy, State, SchedMode
from .tel import make_blocks, CalTarget

logger = u.init_logger(__name__)


# ----------------------------------------------------
#         setup satp2 specific configs
# ----------------------------------------------------

def make_geometry():
    ws0_shift = np.degrees([0, 0])
    ws1_shift = np.degrees([0, 0])
    ws2_shift = np.degrees([0, 0])
    ws3_shift = np.degrees([0, 0])
    ws4_shift = np.degrees([0, 0])
    ws5_shift = np.degrees([0, 0])
    ws6_shift = np.degrees([0, 0])

    ## default SAT optics offests
    d_xi = 10.9624
    d_eta_side = 6.46363
    d_eta_mid = 12.634

    return {
        'ws3': {
            'center': [-d_xi+ws3_shift[0], d_eta_side+ws3_shift[1]],
            'radius': 6,
        },
        'ws2': {
            'center': [-d_xi+ws2_shift[0], -d_eta_side+ws2_shift[1]],
            'radius': 6,
        },
        'ws4': {
            'center': [0+ws4_shift[0], d_eta_mid+ws4_shift[1]],
            'radius': 6,
        },
        'ws0': {
            'center': [0+ws0_shift[0], 0+ws0_shift[1]],
            'radius': 6,
        },
        'ws1': {
            'center': [0+ws1_shift[0], -d_eta_mid+ws1_shift[1]],
            'radius': 6,
        },
        'ws5': {
            'center': [d_xi+ws5_shift[0], d_eta_side+ws5_shift[1]],
            'radius': 6,
        },
        'ws6': {
            'center': [d_xi+ws6_shift[0], -d_eta_side+ws6_shift[1]],
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
        focus_str = focus ##
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

def make_operations(
    az_speed,
    az_accel,
    iv_cadence=4*u.hour,
    bias_step_cadence=0.5*u.hour,
    det_setup_duration=20*u.minute,
    brake_hwp=True,
    disable_hwp=False,
    apply_boresight_rot=True,
    hwp_cfg=None,
    home_at_end=False,
    relock_cadence=24*u.hour
):
    if hwp_cfg is None:
        hwp_cfg = { 'iboot2': 'power-iboot-hwp-2', 'pid': 'hwp-pid', 'pmx': 'hwp-pmx', 'hwp-pmx': 'pmx', 'gripper': 'hwp-gripper',}
    pre_session_ops = [
        { 'name': 'sat.preamble'    , 'sched_mode': SchedMode.PreSession},
        { 'name': 'start_time'      , 'sched_mode': SchedMode.PreSession},
        { 'name': 'set_scan_params' , 'sched_mode': SchedMode.PreSession, 'az_speed': az_speed, 'az_accel': az_accel, },
    ]

    cal_ops = []
    cmb_ops = []

    if relock_cadence is not None:
        pre_session_ops += [
            { 'name': 'sat.ufm_relock'      , 'sched_mode': SchedMode.PreSession, 'relock_cadence': relock_cadence}
        ]
        cal_ops += [
            { 'name': 'sat.ufm_relock'      , 'sched_mode': SchedMode.PreCal, 'relock_cadence': relock_cadence}
        ]
        cmb_ops += [
            { 'name': 'sat.ufm_relock'      , 'sched_mode': SchedMode.PreObs, 'relock_cadence': relock_cadence}
        ]

    cal_ops += [
        { 'name': 'sat.setup_boresight' , 'sched_mode': SchedMode.PreCal, 'apply_boresight_rot': apply_boresight_rot, 'brake_hwp': brake_hwp},
        { 'name': 'sat.det_setup'       , 'sched_mode': SchedMode.PreCal, 'apply_boresight_rot': apply_boresight_rot, 'iv_cadence':iv_cadence,
        'det_setup_duration': det_setup_duration},
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreCal, 'disable_hwp': disable_hwp, 'brake_hwp': brake_hwp},
        { 'name': 'sat.source_scan'     , 'sched_mode': SchedMode.InCal, },
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PostCal, 'bias_step_cadence': bias_step_cadence},
    ]
    cmb_ops += [
        { 'name': 'sat.setup_boresight' , 'sched_mode': SchedMode.PreObs, 'apply_boresight_rot': apply_boresight_rot, 'brake_hwp': brake_hwp},
        { 'name': 'sat.det_setup'       , 'sched_mode': SchedMode.PreObs, 'apply_boresight_rot': apply_boresight_rot, 'iv_cadence':iv_cadence,
        'det_setup_duration': det_setup_duration},
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreObs, 'disable_hwp': disable_hwp, 'brake_hwp': brake_hwp},
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PreObs,  'bias_step_cadence': bias_step_cadence},
        { 'name': 'sat.cmb_scan'        , 'sched_mode': SchedMode.InObs, },
    ]
    post_session_ops = []
    if home_at_end:
        post_session_ops += [
            { 'name': 'sat.hwp_spin_down'   , 'sched_mode': SchedMode.PostSession, 'disable_hwp': disable_hwp, 'brake_hwp': brake_hwp},
        ]
    post_session_ops += [
        { 'name': 'sat.wrap_up'   , 'sched_mode': SchedMode.PostSession},
    ]

    return pre_session_ops + cal_ops + cmb_ops + post_session_ops

def make_config(
    master_file,
    state_file,
    az_speed,
    az_accel,
    iv_cadence,
    bias_step_cadence,
    max_cmb_scan_duration,
    cal_targets,
    min_hwp_el=None,
    az_stow=None,
    el_stow=None,
    boresight_override=None,
    hwp_override=None,
    brake_hwp=True,
    az_motion_override=False,
    **op_cfg
):
    blocks = make_blocks(master_file, 'sat-cmb')
    geometries = make_geometry()

    det_setup_duration = 20*u.minute

    operations = make_operations(
        az_speed, az_accel,
        iv_cadence, bias_step_cadence,
        det_setup_duration,
        brake_hwp,
        **op_cfg
    )

    sun_policy = {
        'min_angle': 49,
        'min_sun_time': 1980,
        'min_el': 40,
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

    el_range = {
        'el_range': [40, 90]
    }

    config = {
        'state_file': state_file,
        'blocks': blocks,
        'geometries': geometries,
        'rules': {
            'min-duration': {
                'min_duration': 600
            },
            'sun-avoidance': sun_policy,
            'az-range': az_range
        },
        'operations': operations,
        'cal_targets': cal_targets,
        'scan_tag': None,
        'boresight_override': boresight_override,
        'hwp_override':  hwp_override,
        'brake_hwp': brake_hwp,
        'az_motion_override': az_motion_override,
        'az_speed' : az_speed,
        'az_accel' : az_accel,
        'iv_cadence' : iv_cadence,
        'bias_step_cadence' : bias_step_cadence,
        'min_hwp_el' : min_hwp_el,
        'max_cmb_scan_duration' : max_cmb_scan_duration,
        'stages': {
            'build_op': {
                'plan_moves': {
                    'stow_position': stow_position,
                    'sun_policy': sun_policy,
                    'az_step': 0.5,
                    'az_limits': az_range['az_range'],
                    'el_limits': el_range['el_range'],
                }
            }
        }
    }
    return config


# ----------------------------------------------------
#
#         Policy customizations, if any
#
# ----------------------------------------------------
# here we add some convenience wrappers

@dataclass
class SATP2Policy(SATPolicy):
    @classmethod
    def from_defaults(cls,
        master_file,
        state_file=None,
        az_speed=0.8,
        az_accel=1.5,
        iv_cadence=4*u.hour,
        bias_step_cadence=0.5*u.hour,
        max_cmb_scan_duration=1*u.hour,
        cal_targets=None,
        min_hwp_el=48,
        az_stow=None,
        el_stow=None,
        boresight_override=None,
        hwp_override=None,
        brake_hwp=True,
        az_motion_override=False,
        **op_cfg
    ):
        if cal_targets is None:
            cal_targets = []

        x = cls(**make_config(
            master_file,
            state_file,
            az_speed,
            az_accel,
            iv_cadence,
            bias_step_cadence,
            max_cmb_scan_duration,
            cal_targets,
            min_hwp_el,
            az_stow,
            el_stow,
            boresight_override,
            hwp_override,
            brake_hwp,
            az_motion_override,
            **op_cfg
        ))
        return x

    def add_cal_target(self, *args, **kwargs):
        self.cal_targets.append(make_cal_target(*args, **kwargs))

    def init_state(self, t0: dt.datetime) -> State:
        """customize typical initial state for satp2, if needed"""
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
