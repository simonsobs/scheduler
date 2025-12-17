from dataclasses import dataclass
from .sat import SATPolicy
from .. import utils as u
from .. import commands as cmd, instrument as inst


logger = u.init_logger(__name__)

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


@dataclass
class SATP3Policy(SATPolicy):
    platform: str = "satp3"

    def __post_init__(self):
        if self.boresight_override is not None:
            logger.warning("Boresight Override does nothing for SATp3")
            self.boresight_override = None

        cmds_uxm_relock = [
            "",
            "####################### Relock #######################",
            "run.smurf.zero_biases()",
            "time.sleep(120)",
            "run.smurf.uxm_relock(concurrent=True)",
            "################## Relock Over #######################",
            "",
        ]

        cmds_det_setup = [
            "",
            "################### Detector Setup######################",
            "with disable_trace():",
            "    run.initialize()",
            "run.smurf.iv_curve(concurrent=True)",
            "run.smurf.bias_dets(rfrac=0.5, concurrent=True)",
            "time.sleep(300)",
            "run.smurf.bias_step(concurrent=True)",
            "#################### Detector Setup Over ####################",
            "",
        ]

        self.blocks = self.make_blocks('sat-cmb')
        self.geometries = self.make_geometry()
        self.operations = self.make_operations(cmds_uxm_relock=cmds_uxm_relock, cmds_det_setup=cmds_det_setup)

        if self.elevation_override is not None:
            self.stages["build_op"]["plan_moves"]["el_limits"] = 2*[self.elevation_override]
        elif self.force_max_hwp_el and self.max_hwp_el is not None:
            self.stages["build_op"]["plan_moves"]["el_limits"][1] = self.max_hwp_el

    def add_cal_target(self, *args, **kwargs):
        array_focus = {
            'left': 'ws3,ws2',
            'middle': 'ws0,ws1,ws4',
            'right': 'ws5,ws6',
            'top': 'ws3,ws4,ws5',
            'toptop': 'ws4',
            'center': 'ws0',
            'bottom': 'ws1,ws2,ws6',
            'bottombottom': 'ws1',
            'all': 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
        }
        self.cal_targets.append(self.make_cal_target(array_focus, *args, **kwargs))
