from dataclasses import dataclass
from .sat import SATPolicy, sat_move_to
from .. import utils as u
from .. import commands as cmd, instrument as inst


logger = u.init_logger(__name__)

@cmd.operation(name="move_to", return_duration=True)
def move_to(state, az, el, az_offset=0, el_offset=0, min_el=48, max_el=60, brake_hwp=True, force=False):
    state, duration, cmd = sat_move_to(state, az, el, az_offset, el_offset, min_el, max_el, brake_hwp, force)
    if cmd:
        cmd += [
            f"acu.stop_and_clear()",
        ]
    return state, duration, cmd


@dataclass
class SATP2Policy(SATPolicy):
    platform: str = "satp2"

    def __post_init__(self):
        super().__post_init__()

    def add_cal_target(self, *args, **kwargs):
        array_focus = {
            0: {
                'left': 'ws3,ws2',
                'middle': 'ws0,ws1,ws4',
                'right': 'ws5,ws6',
                'bottom': 'ws1,ws2,ws6',
                'all': 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
            },
            45: {
                'left': 'ws3,ws4',
                'middle': 'ws2,ws0,ws5',
                'right': 'ws1,ws6',
                'bottom': 'ws1,ws2,ws3',
                'all': 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
            },
            -45: {
                'left': 'ws1,ws2',
                'middle': 'ws6,ws0,ws3',
                'right': 'ws4,ws5',
                'bottom': 'ws1,ws6,ws5',
                'all': 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
            },
        }

        if 'boresight' in kwargs:
            array_focus = array_focus[kwargs['boresight']]
        elif len(args) >= 3:
            array_focus = array_focus[args[2]]
        else:
            raise ValueError("Missing required parameter 'boresight'.")

        self.cal_targets.append(self.make_cal_target(array_focus, *args, **kwargs))
