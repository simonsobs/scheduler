import numpy as np
from dataclasses import dataclass, replace
import datetime as dt

from typing import Optional

from ..thirdparty import SunAvoidance
from .. import config as cfg, core, source as src, rules as ru
from .. import source as src, utils as u
from .sat import SATPolicy, State, SchedMode, make_geometry
from .tel import make_blocks, CalTarget
from ..instrument import WiregridTarget, StareBlock, parse_cal_targets_from_toast_sat, parse_wiregrid_targets_from_file

logger = u.init_logger(__name__)


# ----------------------------------------------------
#         setup satp2 specific configs
# ----------------------------------------------------


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
    ignore_wafers=None,
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

    if ignore_wafers is not None:
        for angle_key, focus_dict in array_focus.items():
            keys_to_remove = []
            for key, val in focus_dict.items():
                wafers = val.split(',')
                cleaned = [w for w in wafers if w not in ignore_wafers]
                if cleaned:
                    focus_dict[key] = ','.join(cleaned)
                else:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                focus_dict.pop(key)

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
        from_table=False
    )

def make_operations(
    az_speed,
    az_accel,
    az_motion_override=False,
    iv_cadence=4*u.hour,
    bias_step_cadence=0.5*u.hour,
    det_setup_duration=20*u.minute,
    brake_hwp=True,
    disable_hwp=False,
    apply_boresight_rot=True,
    cryo_stabilization_time=0*u.second,
    hwp_cfg=None,
    home_at_end=False,
    relock_cadence=24*u.hour
):
    if hwp_cfg is None:
        hwp_cfg = { 'iboot2': 'power-iboot-hwp-2', 'pid': 'hwp-pid', 'pmx': 'hwp-pmx', 'hwp-pmx': 'pmx', 'gripper': 'hwp-gripper',}
    pre_session_ops = [
        { 'name': 'sat.preamble'    , 'sched_mode': SchedMode.PreSession},
        { 'name': 'start_time'      , 'sched_mode': SchedMode.PreSession},
        { 'name': 'set_scan_params' , 'sched_mode': SchedMode.PreSession, 'az_speed': az_speed, 'az_accel': az_accel, 'az_motion_override': az_motion_override},
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
        { 'name': 'sat.setup_boresight' , 'sched_mode': SchedMode.PreCal, 'apply_boresight_rot': apply_boresight_rot, 'brake_hwp': brake_hwp, 'cryo_stabilization_time': cryo_stabilization_time},
        { 'name': 'sat.det_setup'       , 'sched_mode': SchedMode.PreCal, 'apply_boresight_rot': apply_boresight_rot, 'iv_cadence':iv_cadence,
        'det_setup_duration': det_setup_duration},
        { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreCal, 'disable_hwp': disable_hwp, 'brake_hwp': brake_hwp},
        { 'name': 'sat.source_scan'     , 'sched_mode': SchedMode.InCal, },
        { 'name': 'sat.bias_step'       , 'sched_mode': SchedMode.PostCal, 'bias_step_cadence': bias_step_cadence},
    ]
    cmb_ops += [
        { 'name': 'sat.setup_boresight' , 'sched_mode': SchedMode.PreObs, 'apply_boresight_rot': apply_boresight_rot, 'brake_hwp': brake_hwp, 'cryo_stabilization_time': cryo_stabilization_time},
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

    wiregrid_ops = []
    if not disable_hwp:
        wiregrid_ops += [
            { 'name': 'sat.setup_boresight' , 'sched_mode': SchedMode.PreCal, 'apply_boresight_rot': apply_boresight_rot, 'brake_hwp': brake_hwp, 'cryo_stabilization_time': cryo_stabilization_time},
            { 'name': 'sat.det_setup'       , 'sched_mode': SchedMode.PreWiregrid, 'apply_boresight_rot': apply_boresight_rot, 'iv_cadence':iv_cadence,},
            { 'name': 'sat.hwp_spin_up'     , 'sched_mode': SchedMode.PreWiregrid, 'disable_hwp': disable_hwp, 'brake_hwp': brake_hwp},
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
    az_offset=0.,
    el_offset=0.,
    xi_offset=0.,
    eta_offset=0.,
    boresight_override=None,
    hwp_override=None,
    brake_hwp=True,
    disable_hwp=False,
    az_motion_override=False,
    az_branch_override=None,
    allow_partial_override=False,
    drift_override=True,
    wiregrid_az=180,
    wiregrid_el=48,
    **op_cfg
):
    blocks = make_blocks(master_file, 'sat-cmb')
    geometries = make_geometry(xi_offset, eta_offset)

    det_setup_duration = 20*u.minute

    operations = make_operations(
        az_speed, az_accel,
        az_motion_override,
        iv_cadence, bias_step_cadence,
        det_setup_duration,
        brake_hwp,
        disable_hwp,
        **op_cfg
    )

    sun_policy = {
        'min_angle': 49,
        'min_sun_time': 1980,
        'min_el': 40,
        'max_el': 90,
        'min_az': -45,
        'max_az': 405,
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
        'disable_hwp': disable_hwp,
        'az_motion_override': az_motion_override,
        'az_speed' : az_speed,
        'az_accel' : az_accel,
        'az_offset': az_offset,
        'el_offset': el_offset,
        'iv_cadence' : iv_cadence,
        'bias_step_cadence' : bias_step_cadence,
        'min_hwp_el' : min_hwp_el,
        'max_cmb_scan_duration' : max_cmb_scan_duration,
        'az_branch_override': az_branch_override,
        'allow_partial_override': allow_partial_override,
        'drift_override': drift_override,
        'wiregrid_az': wiregrid_az,
        'wiregrid_el': wiregrid_el,
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
        az_offset=0.,
        el_offset=0.,
        xi_offset=0.,
        eta_offset=0.,
        boresight_override=None,
        hwp_override=None,
        brake_hwp=True,
        disable_hwp=False,
        az_motion_override=False,
        az_branch_override=None,
        allow_partial_override=False,
        drift_override=True,
        wiregrid_az=180,
        wiregrid_el=48,
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
            az_offset,
            el_offset,
            xi_offset,
            eta_offset,
            boresight_override,
            hwp_override,
            brake_hwp,
            disable_hwp,
            az_motion_override,
            az_branch_override,
            allow_partial_override,
            drift_override,
            wiregrid_az,
            wiregrid_el,
            **op_cfg
        ))
        return x

    def add_cal_target(self, *args, **kwargs):
        self.cal_targets.append(make_cal_target(*args, **kwargs))

    def init_cal_seqs(self, cfile, wgfile, blocks, t0, t1, anchor_time=None, ignore_wafers=None):
        # source -> boresight -> allow_partial
        array_focus = {
            'jupiter': {
                0 : {
                    'ws6': True,
                    'ws1,ws0': True,
                    'ws2': True
                },
                -45 : {
                    'ws6,ws0': True,
                    'ws5': True,
                },
                45 : {
                    'ws2,ws0': True,
                    'ws3': True,
                }
            },
            'taua': {
                0 : {
                    'ws6': True,
                    'ws1,ws0': True,
                    'ws2': True
                },
                -45 : {
                    'ws6,ws0': True,
                    'ws5': True,
                },
                45 : {
                    'ws2,ws0': True,
                    'ws3': True,
                }
            },
            'saturn': {
                0 : {
                    'ws0,ws4': False,
                },
                -45 : {
                    'ws0,ws3': False,
                },
                45 : {
                    'ws0,ws5': False,
                }
            },

            'moon': {
                0 : {
                    'ws3,ws2': False,
                    'ws0,ws1,ws4': False,
                    'ws5,ws6': False,
                    'ws1,ws2,ws6': False,
                },
                45 : {
                    'ws3,ws4': False,
                    'ws2,ws0,ws5': False,
                    'ws1,ws6': False,
                    'ws1,ws2,ws3': False,
                },
                -45 : {
                    'ws1,ws2': False,
                    'ws6,ws0,ws3': False,
                    'ws4,ws5': False,
                    'ws1,ws6,ws5': False,
                },
            }
        }

        if ignore_wafers is not None:
            for outer_key in list(array_focus.keys()):
                for inner_key in list(array_focus[outer_key].keys()):
                    new_entries = {}
                    for wafers_str, val in array_focus[outer_key][inner_key].items():
                        wafers = wafers_str.split(',')
                        filtered = [w for w in wafers if w not in ignore_wafers]
                        if filtered:
                            new_key = ','.join(filtered)
                            new_entries[new_key] = val
                    array_focus[outer_key][inner_key] = new_entries

        # get cal targets
        if cfile is not None:
            cal_targets = parse_cal_targets_from_toast_sat(cfile)
            # keep all cal targets within range (don't restrict cal_target.t1 to t1 so we can keep partial scans)
            cal_targets[:] = [cal_target for cal_target in cal_targets if cal_target.t0 >= t0 and cal_target.t0 < t1]
            # ensure cal_target source is in array_focus
            cal_targets[:] = [cal_target for cal_target in cal_targets if cal_target.source in array_focus.keys()]

            # find nearest cmb block either before or after the cal target
            for i, cal_target in enumerate(cal_targets):
                if cal_target.boresight_rot is None:
                    candidates = [block for block in blocks['baseline']['cmb'] if block.t0 < cal_target.t0]
                    if candidates:
                        block = max(candidates, key=lambda x: x.t0)
                    else:
                        candidates = [block for block in blocks['baseline']['cmb'] if block.t0 > cal_target.t0]
                        if candidates:
                            block = min(candidates, key=lambda x: x.t0)
                        else:
                            raise ValueError("Cannot find nearby block for cal target")

                if self.boresight_override is None:
                    if cal_target.boresight_rot is None:
                        cal_targets[i] = replace(cal_targets[i], boresight_rot=block.boresight_angle)
                else:
                    cal_targets[i] = replace(cal_targets[i], boresight_rot=self.boresight_override)

                # get wafers to observe based on source name and boresight
                if cal_target.array_query is None:
                    focus_str = array_focus[cal_targets[i].source][cal_targets[i].boresight_rot]
                    index = u.get_cycle_option(cal_target.t0, list(focus_str.keys()), anchor_time)
                    # order list so current date's array_query is tried first
                    array_query = list(focus_str.keys())[index:] + list(focus_str.keys())[:index]
                    #array_query = list(focus_str.keys())[index]
                    cal_targets[i] = replace(cal_targets[i], array_query=array_query)

                    allow_partial = list(focus_str.values())[index:] + list(focus_str.values())[:index]
                else:
                    allow_partial = array_focus[cal_targets[i].source][cal_targets[i].boresight_rot][cal_target.array_query]
                cal_targets[i] = replace(cal_targets[i], allow_partial=allow_partial)

                if self.az_branch_override is not None:
                    cal_targets[i] = replace(cal_targets[i], az_branch=self.az_branch_override)

                cal_targets[i] = replace(cal_targets[i], az_speed=0.8, az_accel=1.0)
                cal_targets[i] = replace(cal_targets[i], drift=self.drift_override)

            self.cal_targets += cal_targets

        # get wiregrid file
        if wgfile is not None and not self.disable_hwp:
            wiregrid_candidates = parse_wiregrid_targets_from_file(wgfile)
            # since we don't restrict the wiregrid observations, make sure they end before t1
            wiregrid_candidates[:] = [wiregrid_candidate for wiregrid_candidate in wiregrid_candidates if wiregrid_candidate.t0 >= t0 and wiregrid_candidate.t1 <= t1]

            for i, wiregrid_candidate in enumerate(wiregrid_candidates):
                candidates = [block for block in blocks['baseline']['cmb'] if block.t0 < wiregrid_candidate.t0]
                if candidates:
                    block = max(candidates, key=lambda x: x.t0)
                else:
                    candidates = [block for block in blocks['baseline']['cmb'] if block.t0 > wiregrid_candidate.t0]
                    if candidates:
                        block = min(candidates, key=lambda x: x.t0)
                    else:
                        raise ValueError("Cannot find nearby block for cal target")

                if self.boresight_override is None:
                    wiregrid_candidates[i] = replace(wiregrid_candidates[i], boresight_rot=block.boresight_angle)
                else:
                    wiregrid_candidates[i] = replace(wiregrid_candidates[i], boresight_rot=self.boresight_override)

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
                        hwp_dir=self.hwp_override if self.hwp_override is not None else None,
                        boresight_angle=cal_target.boresight_rot
                    )
                )
        blocks['calibration']['wiregrid'] = wiregrid_candidates

        return blocks

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
