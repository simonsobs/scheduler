from dataclasses import dataclass
from .sat import SATPolicy
from .. import utils as u


logger = u.init_logger(__name__)


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

        assert_cmds = [
            "assert OCSClient('power-iboot-smurf-1').acq().session['data']['outletStatus_0']['status'] == 1, 'Readout fan shutter/vent is not open'",
        ]

        self.blocks = self.make_blocks('sat-cmb')
        self.geometries = self.make_geometry()
        self.operations = self.make_operations(cmds_uxm_relock=cmds_uxm_relock, cmds_det_setup=cmds_det_setup, assert_cmds=assert_cmds)

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
