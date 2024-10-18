import numpy as np
from dataclasses import dataclass
import datetime as dt

from typing import Optional

from .. import source as src, utils as u
from .sat import SATPolicy, State, CalTarget
from ..commands import SchedMode

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
        az_accel=az_accel
    )

def make_blocks(master_file):
    return {
        'baseline': {
            'cmb': {
                'type': 'toast',
                'file': master_file
            }
        },
        'calibration': {
            'saturn': {
                'type' : 'source',
                'name' : 'saturn',
            },
            'jupiter': {
                'type' : 'source',
                'name' : 'jupiter',
            },
            'moon': {
                'type' : 'source',
                'name' : 'moon',
            },
            'uranus': {
                'type' : 'source',
                'name' : 'uranus',
            },
            'neptune': {
                'type' : 'source',
                'name' : 'neptune',
            },
            'mercury': {
                'type' : 'source',
                'name' : 'mercury',
            },
            'venus': {
                'type' : 'source',
                'name' : 'venus',
            },
            'mars': {
                'type' : 'source',
                'name' : 'mars',
            }
        },
    }

## need custom command list while dealing with needing to disable slots
commands_det_setup = [
    "",
    "################### Detector Setup######################",
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

def make_operations(
    az_speed, az_accel, disable_hwp=False, 
    apply_boresight_rot=True, hwp_cfg=None, hwp_dir=True,
    iv_cadence=4*u.hour, home_at_end=False, run_relock=False,
    bad_smurf_slots=None,
):
    if hwp_cfg is None:
        hwp_cfg = { 'iboot2': 'power-iboot-hwp-2', 'pid': 'hwp-pid', 'pmx': 'hwp-pmx', 'hwp-pmx': 'pmx', 'gripper': 'hwp-gripper', 'forward':hwp_dir }
    pre_session_ops = [
        { 'name': 'sat.preamble'        , 'sched_mode': SchedMode.PreSession},
        { 'name': 'disable_smurf_slots' , 'sched_mode': SchedMode.PreSession, 'disable_list' : bad_smurf_slots},
        { 'name': 'start_time'          ,'sched_mode': SchedMode.PreSession},
        { 'name': 'set_scan_params' , 'sched_mode': SchedMode.PreSession, 'az_speed': az_speed, 'az_accel': az_accel, },
    ]
    if run_relock:
        pre_session_ops += [
            { 'name': 'sat.ufm_relock'      , 'sched_mode': SchedMode.PreSession, }
        ]
    cal_ops = [
        { 'name': 'sat.setup_boresight' , 'sched_mode': SchedMode.PreCal, 'apply_boresight_rot': apply_boresight_rot, },
        { 'name': 'sat.det_setup'       , 'sched_mode': SchedMode.PreCal, 'apply_boresight_rot': apply_boresight_rot, 'iv_cadence':iv_cadence,'commands': commands_det_setup, },
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreCal, 'disable_hwp': disable_hwp, 'forward':hwp_dir},
        { 'name': 'sat.source_scan'     , 'sched_mode': SchedMode.InCal, },
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PostCal, 'indent': 4},
    ]
    cmb_ops = [
        { 'name': 'sat.setup_boresight' , 'sched_mode': SchedMode.PreObs, 'apply_boresight_rot': apply_boresight_rot, },
        { 'name': 'sat.det_setup'       , 'sched_mode': SchedMode.PreObs, 'apply_boresight_rot': apply_boresight_rot, 'iv_cadence':iv_cadence,'commands': commands_det_setup,},
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreObs, 'disable_hwp': disable_hwp, 'forward':hwp_dir},
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PreObs, },
        { 'name': 'sat.cmb_scan'        , 'sched_mode': SchedMode.InObs, },
    ]
    if home_at_end:
        post_session_ops = [
            { 'name': 'sat.hwp_spin_down'   , 'sched_mode': SchedMode.PostSession, 'disable_hwp': disable_hwp, },
            { 'name': 'sat.wrap_up'         , 'sched_mode': SchedMode.PostSession, 'az_stow': 180, 'el_stow': 48},
        ]
    else:
        post_session_ops = []
    return pre_session_ops + cal_ops + cmb_ops + post_session_ops

def make_config(
    master_file,
    az_speed,
    az_accel,
    cal_targets,
    boresight_override=None,
    **op_cfg
):
    blocks = make_blocks(master_file)
    geometries = make_geometry()
    operations = make_operations(
        az_speed, az_accel,
        **op_cfg
    )

    sun_policy = {
        'min_angle': 49,
        'min_sun_time': 1980,
        'min_el': 48,
    }

    config = {
        'blocks': blocks,
        'geometries': geometries,
        'rules': {
            'min-duration': {
                'min_duration': 600
            },
            'sun-avoidance': sun_policy,
        },
        'operations': operations,
        'cal_targets': cal_targets,
        'scan_tag': None,
        'boresight_override': boresight_override,
        'az_speed' : az_speed,
        'az_accel' : az_accel,
        'stages': {
            'build_op': {
                'plan_moves': {
                    'sun_policy': sun_policy,
                    'az_step': 0.5,
                    'az_limits': [-45, 405],
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
    state_file: Optional[str] = None

    @classmethod
    def from_defaults(cls, master_file, az_speed=0.8, az_accel=1.5, 
        cal_targets=[], boresight_override=None, 
        state_file=None, **op_cfg
    ):
        x = cls(**make_config(
            master_file, az_speed, az_accel, 
            cal_targets, boresight_override, **op_cfg
        ))
        x.state_file=state_file
        return x

    def add_cal_target(self, *args, **kwargs):
        self.cal_targets.append(make_cal_target(*args, **kwargs))

    def init_state(self, t0: dt.datetime) -> State:
        """customize typical initial state for satp1, if needed"""
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
