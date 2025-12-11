from dataclasses import dataclass
from .sat import SATPolicy
from .. import utils as u


logger = u.init_logger(__name__)


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
