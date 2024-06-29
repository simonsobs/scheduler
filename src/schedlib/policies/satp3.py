import numpy as np
from dataclasses import dataclass
import datetime as dt

from .. import source as src, utils as u, commands as cmd
from .sat import SATPolicy, State, CalTarget
from ..commands import SchedMode

logger = u.init_logger(__name__)


# ----------------------------------------------------
#         setup satp3 specific configs
# ----------------------------------------------------

def make_geometry():
    ufm_mv12_shift = np.degrees([0, 0])
    ufm_mv35_shift = np.degrees([0, 0])
    ufm_mv23_shift = np.degrees([0, 0])
    ufm_mv5_shift  = np.degrees([0, 0])
    ufm_mv27_shift = np.degrees([0, 0])
    ufm_mv33_shift = np.degrees([0, 0])
    ufm_mv17_shift = np.degrees([0, 0])

    d_xi = 10.9624
    d_eta_side = 6.46363
    d_eta_mid = 12.634

    return {
      'ws3': {
        'center': [-d_xi+ufm_mv12_shift[0], d_eta_side+ufm_mv12_shift[1]],
        'radius': 6,
      },
      'ws2': {
        'center': [-d_xi+ufm_mv35_shift[0], -d_eta_side+ufm_mv35_shift[1]],
        'radius': 6,
      },
      'ws4': {
        'center': [0+ufm_mv23_shift[0], d_eta_mid+ufm_mv23_shift[1]],
        'radius': 6,
      },
      'ws0': {
        'center': [0+ufm_mv5_shift[0], 0+ufm_mv5_shift[1]],
        'radius': 6,
      },
      'ws1': {
        'center': [0+ufm_mv27_shift[0], -d_eta_mid+ufm_mv27_shift[1]],
        'radius': 6,
      },
      'ws5': {
        'center': [d_xi+ufm_mv33_shift[0], d_eta_side+ufm_mv33_shift[1]],
        'radius': 6,
      },
      'ws6': {
        'center': [d_xi+ufm_mv17_shift[0], -d_eta_side+ufm_mv17_shift[1]],
        'radius': 6,
      },
    }

def make_cal_target(
    source: str,
    boresight: int,
    elevation: int,
    focus: str,
    allow_partial=False,
    drift=True,
) -> CalTarget:
    array_focus = {
        'left' : 'ws3,ws2',
        'middle' : 'ws0,ws1,ws4',
        'right' : 'ws5,ws6',
        'top': 'ws3,ws4,ws5',
        'toptop': 'ws4',
        'center': 'ws0',
        'bottom': 'ws1,ws2,ws6',
        'bottombottom': 'ws1',
        'all' : 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
    }

    boresight = int(boresight)
    elevation = int(elevation)
    focus = focus.lower()

    focus_str = None
    focus_str = array_focus.get(focus, focus)

    assert source in src.SOURCES, f"source should be one of {src.SOURCES.keys()}"

    return CalTarget(
        source=source,
        array_query=focus_str,
        el_bore=elevation,
        boresight_rot=boresight,
        tag=focus_str,
        allow_partial=allow_partial,
        drift=drift
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
            },
            'taua': {
                'type' : 'source',
                'name' : 'taua',
            },
            'galcenter': {
                'type' : 'source',
                'name' : 'galcenter',
            },
        },
    }

@cmd.operation(name='satp3.det_setup', return_duration=True)
def det_setup(state, block, apply_boresight_rot=False, iv_cadence=None):
    # when should det setup be done?
    # -> should always be done if the block is a cal block
    # -> should always be done if elevation has changed
    # -> should always be done if det setup has not been done yet
    # -> should be done at a regular interval if iv_cadence is not None
    # -> should always be done if boresight rotation has changed
    doit = (block.subtype == 'cal') or (block.alt != state.el_now)
    doit = doit or (not state.is_det_setup)
    doit = doit or (iv_cadence is not None and ((state.last_iv is None) or ((state.curr_time - state.last_iv).total_seconds() > iv_cadence)))
    if apply_boresight_rot and (block.boresight_angle != state.boresight_rot_now):
        doit = True

    if doit:
        commands = [
            "",
            "################### Detector Setup######################",
            "run.smurf.take_bgmap(concurrent=True)",
            "run.smurf.iv_curve(concurrent=True)",
            "for smurf in pysmurfs:",
            "    smurf.bias_dets.start(rfrac=0.5, kwargs=dict(bias_groups=[0,1,2,3,4,5,6,7,8,9,10,11]))",
            "time.sleep(300)",
            "run.smurf.bias_step(concurrent=True)",
            "#################### Detector Setup Over ####################",
            "",
        ]
        state = state.replace(
            last_bias_step=state.curr_time,
            is_det_setup=True,
            last_iv = state.curr_time,
        )
        return state, 12*u.minute, commands
    else:
        return state, 0, []

def make_operations(
    az_speed, az_accel, disable_hwp=False,
    apply_boresight_rot=False, hwp_cfg=None, hwp_dir=True,
    iv_cadence=4*u.hour,
):
    if hwp_cfg is None:
        hwp_cfg = { 'iboot2': 'power-iboot-hwp-2', 'pid': 'hwp-pid', 'pmx': 'hwp-pmx', 'hwp-pmx': 'pmx', 'gripper': 'hwp-gripper', 'forward':hwp_dir }
    pre_session_ops = [
        { 'name': 'sat.preamble'        , 'sched_mode': SchedMode.PreSession, },
        { 'name': 'sat.ufm_relock'      , 'sched_mode': SchedMode.PreSession, },
        { 'name': 'set_scan_params' , 'sched_mode': SchedMode.PreSession, 'az_speed': az_speed, 'az_accel': az_accel, },
    ]
    cal_ops = [
        { 'name': 'satp3.det_setup'     , 'sched_mode': SchedMode.PreCal, 'apply_boresight_rot': apply_boresight_rot, },
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreCal, 'disable_hwp': disable_hwp, 'forward':hwp_dir},
        { 'name': 'sat.source_scan'     , 'sched_mode': SchedMode.InCal, },
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PostCal, 'indent': 4},
    ]
    cmb_ops = [
        { 'name': 'satp3.det_setup'     , 'sched_mode': SchedMode.PreObs, 'apply_boresight_rot': apply_boresight_rot, 'iv_cadence':iv_cadence},
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreObs, 'disable_hwp': disable_hwp, 'forward':hwp_dir},
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PreObs, },
        { 'name': 'sat.cmb_scan'        , 'sched_mode': SchedMode.InObs, },
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PostObs, 'indent': 4, 'divider': ['']},
    ]
    post_session_ops = [
        { 'name': 'sat.hwp_spin_down'   , 'sched_mode': SchedMode.PostSession, 'disable_hwp': disable_hwp, },
        { 'name': 'sat.wrap_up'         , 'sched_mode': SchedMode.PostSession, 'az_stow': 180, 'el_stow': 60},
    ]
    return pre_session_ops + cal_ops + cmb_ops + post_session_ops

def make_config(
    master_file,
    az_speed,
    az_accel,
    cal_targets,
    **op_cfg
):
    blocks = make_blocks(master_file)
    geometries = make_geometry()
    operations = make_operations(
        az_speed, az_accel,
        **op_cfg
    )

    sun_policy = { 'min_angle': 49, 'min_sun_time': 1980 }

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
        'az_speed' : az_speed,
        'az_accel' : az_accel,
        'stages': {
            'build_op': {
                'plan_moves': {
                    'sun_policy': sun_policy,
                    'az_step': 0.5,
                    'az_limits': [-90, 450],
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
class SATP3Policy(SATPolicy):
    @classmethod
    def from_defaults(cls, master_file, az_speed=0.5, az_accel=0.25, cal_targets=[], **op_cfg):
        return cls(**make_config(master_file, az_speed, az_accel, cal_targets, **op_cfg))

    def add_cal_target(self, *args, **kwargs):
        self.cal_targets.append(make_cal_target(*args, **kwargs))

    def init_state(self, t0: dt.datetime) -> State:
        """customize typical initial state for satp1, if needed"""
        return State(
            curr_time=t0,
            az_now=180,
            el_now=60,
            boresight_rot_now=0,
            hwp_spinning=False,
        )
