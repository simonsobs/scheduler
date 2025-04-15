import numpy as np
from dataclasses import dataclass, replace
import datetime as dt

from typing import Optional

from .. import source as src, utils as u
from .sat import SATPolicy, State, SchedMode, WiregridTarget
from .tel import make_blocks, CalTarget
from ..instrument import parse_cal_targets_from_toast_sat

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
    az_branch=None,
    az_speed=None,
    az_accel=None,
    source_direction=None,
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


commands_uxm_relock = [
    "",
    "####################### Relock #######################",
    "run.smurf.zero_biases()",
    "time.sleep(120)",
    "run.smurf.take_noise(concurrent=True, tag='res_check')",
    "run.smurf.uxm_relock(concurrent=True)",
    "run.smurf.take_bgmap(concurrent=True)",
    "################## Relock Over #######################",
    "",
]


commands_det_setup = [
    "",
    "################### Detector Setup######################",
    "with disable_trace():",
    "    run.initialize()",
    "run.smurf.take_bgmap(concurrent=True)",
    "run.smurf.iv_curve(concurrent=True)",
    "run.smurf.bias_dets(rfrac=0.5, concurrent=True)",
    "time.sleep(300)",
    "run.smurf.bias_step(concurrent=True)",
    "#################### Detector Setup Over ####################",
    "",
]

def make_operations(
    az_speed,
    az_accel,
    iv_cadence=4*u.hour,
    bias_step_cadence=0.5*u.hour,
    det_setup_duration=20*u.minute,
    brake_hwp=True,
    disable_hwp=False,
    apply_boresight_rot=False,
    hwp_cfg=None,
    home_at_end=False,
    relock_cadence=24*u.hour
):
    if hwp_cfg is None:
        hwp_cfg = { 'iboot2': 'power-iboot-hwp-2', 'pid': 'hwp-pid', 'pmx': 'hwp-pmx', 'hwp-pmx': 'pmx', 'gripper': 'hwp-gripper',}

    pre_session_ops = [
        { 'name': 'sat.preamble'        , 'sched_mode': SchedMode.PreSession, },
        { 'name': 'start_time'          ,'sched_mode': SchedMode.PreSession},
        { 'name': 'set_scan_params' , 'sched_mode': SchedMode.PreSession, 'az_speed': az_speed, 'az_accel': az_accel, },
    ]

    cal_ops = []
    cmb_ops = []

    if relock_cadence is not None:
        pre_session_ops += [
            { 'name': 'sat.ufm_relock'      , 'sched_mode': SchedMode.PreSession, 'relock_cadence': relock_cadence, 'commands': commands_uxm_relock,}
        ]
        cal_ops += [
            { 'name': 'sat.ufm_relock'      , 'sched_mode': SchedMode.PreCal, 'relock_cadence': relock_cadence, 'commands': commands_uxm_relock,}
        ]
        cmb_ops += [
            { 'name': 'sat.ufm_relock'      , 'sched_mode': SchedMode.PreObs, 'relock_cadence': relock_cadence, 'commands': commands_uxm_relock,}
        ]

    cal_ops += [
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreCal, 'disable_hwp': disable_hwp, 'brake_hwp': brake_hwp},
        { 'name': 'sat.det_setup'       , 'sched_mode': SchedMode.PreCal, 'commands': commands_det_setup, 'apply_boresight_rot': apply_boresight_rot,
        'det_setup_duration': det_setup_duration},
        { 'name': 'sat.source_scan'     , 'sched_mode': SchedMode.InCal, },
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PostCal, 'bias_step_cadence': bias_step_cadence},
    ]
    cmb_ops += [
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreObs, 'disable_hwp': disable_hwp, 'brake_hwp': brake_hwp},
        { 'name': 'sat.det_setup'       , 'sched_mode': SchedMode.PreObs, 'commands': commands_det_setup, 'apply_boresight_rot': apply_boresight_rot, 'iv_cadence':iv_cadence,
        'det_setup_duration': det_setup_duration},
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PreObs, 'bias_step_cadence': bias_step_cadence},
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

    wiregrid_ops = [
        { 'name': 'sat.wiregrid', 'sched_mode': SchedMode.Wiregrid }
    ]
    return pre_session_ops + cal_ops + cmb_ops + post_session_ops + wiregrid_ops

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

    if boresight_override is not None:
        logger.warning("Boresight Override does nothing for SATp3")
        boresight_override = None

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
            'az-range': az_range,
        },
        'operations': operations,
        'cal_targets': cal_targets,
        'scan_tag': None,
        'boresight_override': boresight_override,
        'hwp_override': hwp_override,
        'brake_hwp': brake_hwp,
        'az_motion_override': az_motion_override,
        'az_speed': az_speed,
        'az_accel': az_accel,
        'iv_cadence': iv_cadence,
        'bias_step_cadence': bias_step_cadence,
        'min_hwp_el': min_hwp_el,
        'max_cmb_scan_duration': max_cmb_scan_duration,
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
class SATP3Policy(SATPolicy):
    @classmethod
    def from_defaults(cls,
        master_file,
        state_file=None,
        az_speed=0.5,
        az_accel=0.25,
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
        az_branch_override=None,
        allow_partial_override=None,
        drift_override=True,
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
            cal_targets, min_hwp_el,
            az_stow,
            el_stow,
            boresight_override,
            hwp_override,
            brake_hwp,
            az_motion_override,
            **op_cfg)
        )

        return x

    def add_cal_target(self, *args, **kwargs):
        self.cal_targets.append(make_cal_target(*args, **kwargs))

    def init_cal_seq(self, cfile, cal_targets, blocks, t0: dt.datetime, t1: dt.datetime):
        array_focus = {
            'left' : 'ws3,ws2',
            'middle' : 'ws0,ws1,ws4',
            'right' : 'ws5,ws6',
            'top': 'ws3,ws4,ws5',
            'toptop': 'ws4',
            'center': 'ws0',
            'bottom': 'ws1,ws2,ws6',
            'bottombottom': 'ws1',
            #'all' : 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
        }
       # get cal targets
        if cfile is not None:
            cal_targets = parse_cal_targets_from_toast_sat(cfile)
            # keep all cal targets within range
            cal_targets[:] = [cal_target for cal_target in cal_targets if cal_target.t0 >= t0 and cal_target.t0 < t1]

            for i, cal_target in enumerate(cal_targets):
                candidates = [block for block in blocks['baseline']['cmb'] if block.t0 < cal_target.t0]
                if candidates:
                    block = max(candidates, key=lambda x: x.t0)
                else:
                    candidates = [block for block in blocks['baseline']['cmb'] if block.t0 > cal_target.t0]
                    if candidates:
                        block = min(candidates, key=lambda x: x.t0)
                    else:
                        raise ValueError("Cannot find nearby block")

                cal_targets[i] = replace(cal_targets[i], boresight_rot=block.boresight_angle)
                focus_str = array_focus[cal_targets[i].boresight_rot]
                array_query = u.get_cycle_option(t0, list(focus_str.keys()))
                cal_targets[i] = replace(cal_targets[i], array_query=focus_str[array_query])
                cal_targets[i] = replace(cal_targets[i], tag=f"{focus_str[array_query]},{cal_targets[i].tag}")

                if self.az_branch_override is not None:
                    cal_targets[i] = replace(cal_targets[i], az_branch=self.az_branch_override)

                cal_targets[i] = replace(cal_targets[i], allow_partial=self.allow_partial_override)
                cal_targets[i] = replace(cal_targets[i], drift=self.drift_override)

            self.cal_targets += cal_targets

            for target in self.cal_targets:
                if target.source not in src.get_source_list():
                   if target.ra is not None and target.dec is not None:
                        src.add_fixed_source(
                            name=target.source,
                            ra=target.ra, dec=target.dec,
                            ra_units='deg'
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

        return blocks

    def init_state(self, t0: dt.datetime) -> State:
        """customize typical initial state for satp3, if needed"""
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
