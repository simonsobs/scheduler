from dataclasses import dataclass
from .sat import SATPolicy, make_geometry
from .. import utils as u


logger = u.init_logger(__name__)


@dataclass
class SATp3Policy(SATPolicy):
    platform: str = "satp3"

    def __post_init__(self):
        if self.boresight_override is not None:
            logger.warning("Boresight Override does nothing for SATp3")
            self.boresight_override = None

        super().__post_init__()

    def add_cal_target(self, *args, **kwargs):
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
        self.cal_targets.append(self.make_cal_target(array_focus, *args, **kwargs))
