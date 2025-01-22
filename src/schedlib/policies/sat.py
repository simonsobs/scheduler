"""A production-level implementation of the SAT policy

"""
import numpy as np
import yaml
import os.path as op
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

import datetime as dt
from typing import List, Union, Optional, Dict, Any, Tuple
import jax.tree_util as tu
from functools import reduce, partial

from . import tel
from .. import config as cfg, core, source as src, rules as ru
from .. import commands as cmd, instrument as inst, utils as u
from ..thirdparty import SunAvoidance
from .stages import get_build_stage

logger = u.init_logger(__name__)


@dataclass_json
@dataclass(frozen=True)
class State(tel.State):
    """
    State relevant to SAT operation scheduling. Inherits other fields:
    (`curr_time`, `az_now`, `el_now`, `az_speed_now`, `az_accel_now`)
    from the base State defined in `schedlib.commands`.

    Parameters
    ----------
    hwp_spinning : bool
        Whether the high-precision measurement wheel is spinning or not.
    hwp_dir : bool
        Current direction of HWP.  True is forward, False is backwards.
    last_ufm_relock : Optional[datetime.datetime]
        The last time the UFM was relocked, or None if it has not been relocked.
    last_bias_step : Optional[datetime.datetime]
        The last time a bias step was performed, or None if no bias step has been performed.
    is_det_setup : bool
        Whether the detectors have been set up or not.
    """
    hwp_spinning: bool = False
    hwp_dir: bool = None


# ----------------------------------------------------
#                  Register operations
# ----------------------------------------------------
# Note: to avoid naming collisions. Use appropriate prefixes
# whenver necessary. For example, all satp1 specific
# operations should start with `satp1`.
#
# Registered operations can be three kinds of functions:
#
# 1. for operations with static duration, it can be defined as a function
#    that returns a list of commands, with the static duration specified in
#    the decorator
# 2. for operations with dynamic duration, meaning the duration is determined
#    at runtime, it can be defined as a function that returns a tuple of
#    duration and commands; the decorator should be informed with the option
#    `return_duration=True`
# 3. for operations that depends and/or modifies the state, the operation
#    function should take the state as the first argument (no renaming allowed)
#    and return a new state before the rest of the return values
#
# For example the following are all valid definitions:
#  @cmd.operation(name='my-op', duration=10)
#  def my_op():
#      return ["do something"]
#
#  @cmd.operation(name='my-op', return_duration=True)
#  def my_op():
#      return 10, ["do something"]
#
#  @cmd.operation(name='my-op')
#  def my_op(state):
#      return state, ["do something"]
#
#  @cmd.operation(name='my-op', return_duration=True)
#  def my_op(state):
#      return state, 10, ["do something"]

@cmd.operation(name="sat.preamble", duration=0)
def preamble():
    base = tel.preamble()
    append = ["sup = OCSClient('hwp-supervisor')", "",]
    return base + append

@cmd.operation(name='sat.ufm_relock', return_duration=True)
def ufm_relock(state, commands=None):
    return tel.ufm_relock(state)

@cmd.operation(name='sat.det_setup', return_duration=True)
def det_setup(state, block, commands=None, apply_boresight_rot=True, iv_cadence=None):
    return tel.det_setup(state, block, commands, apply_boresight_rot, iv_cadence)

@cmd.operation(name='sat.setup_boresight', return_duration=True)
def setup_boresight(state, block, apply_boresight_rot=True):
    commands = []
    duration = 0

    if apply_boresight_rot and (
            state.boresight_rot_now is None or state.boresight_rot_now != block.boresight_angle
        ):
        if state.hwp_spinning:
            state = state.replace(hwp_spinning=False)
            duration += cmd.HWP_SPIN_DOWN
            commands += [
                "run.hwp.stop(active=True)",
                "sup.disable_driver_board()",
            ]

        assert not state.hwp_spinning
        commands += [f"run.acu.set_boresight({block.boresight_angle})"]
        state = state.replace(boresight_rot_now=block.boresight_angle)
        duration += 1*u.minute

    return state, duration, commands

@cmd.operation(name='sat.bias_step', return_duration=True)
def bias_step(state, block, bias_step_cadence=None):
    return tel.bias_step(state, block, bias_step_cadence)

@cmd.operation(name='sat.hwp_spin_up', return_duration=True)
def hwp_spin_up(state, block, disable_hwp=False):
    cmds = []
    duration = 0

    if disable_hwp:
        return state, 0, ["# hwp disabled"]

    elif state.hwp_spinning:
        # if spinning in opposite direction, spin down first
        if block.hwp_dir is not None and state.hwp_dir != block.hwp_dir:
            duration += cmd.HWP_SPIN_DOWN
            cmds += [
            "run.hwp.stop(active=True)",
            "sup.disable_driver_board()",
            ]
        else:
            return state, 0, [f"# hwp already spinning with forward={state.hwp_dir}"]

    hwp_dir = block.hwp_dir if block.hwp_dir is not None else state.hwp_dir
    state = state.replace(hwp_dir=hwp_dir)
    state = state.replace(hwp_spinning=True)

    freq = 2 if hwp_dir else -2
    return state, duration + cmd.HWP_SPIN_UP, cmds + [
        "sup.enable_driver_board()",
        f"run.hwp.set_freq(freq={freq})",
    ]

@cmd.operation(name='sat.hwp_spin_down', return_duration=True)
def hwp_spin_down(state, disable_hwp=False):
    if disable_hwp:
        return state, 0, ["# hwp disabled"]
    elif not state.hwp_spinning:
        return state, 0, ["# hwp already stopped"]
    else:
        state = state.replace(hwp_spinning=False)
        return state, cmd.HWP_SPIN_DOWN, [
            "run.hwp.stop(active=True)",
            "sup.disable_driver_board()",
        ]

@cmd.operation(name='sat.cmb_scan', return_duration=True)
def cmb_scan(state, block):
    return tel.cmb_scan(state, block)

@cmd.operation(name='sat.source_scan', return_duration=True)
def source_scan(state, block):
    return tel.source_scan(state, block)

@cmd.operation(name="move_to", return_duration=True)
def move_to(state, az, el, min_el=48, force=False):
    if not force and (state.az_now == az and state.el_now == el):
        return state, 0, []

    duration = 0
    cmd = []

    if state.hwp_spinning and el < min_el:
        state = state.replace(hwp_spinning=False)
        duration += HWP_SPIN_DOWN
        cmd += [
            "run.hwp.stop(active=True)",
            "sup.disable_driver_board()",
        ]

    cmd += [
        f"run.acu.move_to(az={round(az, 3)}, el={round(el, 3)})",
    ]
    state = state.replace(az_now=az, el_now=el)

    return state, duration, cmd


@dataclass
class SATPolicy(tel.TelPolicy):
    """a more realistic SAT policy.

    Parameters
    ----------
    hwp_override : bool
        a bool to override the hwp direction from the master schedules.  True is forward, False is
        reverse.
    min_hwp_el : float
        the minimum elevation a move command to go to without stopping the hwp first
    """
    hwp_override: Optional[bool] = None
    min_hwp_el : float = 48 # deg

    @classmethod
    def from_config(cls, config: Union[Dict[str, Any], str]):
        """
        Constructs a policy object from a YAML configuration file, a YAML string, or a dictionary.

        Parameters
        ----------
        config : Union[dict, str]
            The configuration to populate the policy object.

        Returns
        -------
        The constructed policy object.
        """
        if isinstance(config, str):
            loader = cfg.get_loader()
            if op.isfile(config):
                with open(config, "r") as f:
                    config = yaml.load(f.read(), Loader=loader)
            else:
                config = yaml.load(config, Loader=loader)
        return cls(**config)

    def init_seqs(self, t0: dt.datetime, t1: dt.datetime) -> core.BlocksTree:
        """
        Initialize the sequences for the scheduler to process.

        Parameters
        ----------
        t0 : datetime.datetime
            The start time of the sequences.
        t1 : datetime.datetime
            The end time of the sequences.

        Returns
        -------
        BlocksTree (nested dict / list of blocks)
            The initialized sequences
        """

        # construct seqs by traversing the blocks definition dict
        blocks = tu.tree_map(
            partial(self.construct_seq, t0=t0, t1=t1),
            self.blocks,
            is_leaf=lambda x: isinstance(x, dict) and 'type' in x
        )

        # override hwp direction
        if self.hwp_override is not None:
            blocks['baseline'] = core.seq_map(
                lambda b: b.replace(
                    hwp_dir=self.hwp_override
                ), blocks['baseline']
            )

        # by default add calibration blocks specified in cal_targets if not already specified
        for cal_target in self.cal_targets:
            if isinstance(cal_target, tel.CalTarget):
                source = cal_target.source
                if source not in blocks['calibration']:
                    blocks['calibration'][source] = src.source_gen_seq(source, t0, t1)

        # trim to given time range
        blocks = core.seq_trim(blocks, t0, t1)

        # ok to drop Nones
        blocks = tu.tree_map(
            lambda x: [x_ for x_ in x if x_ is not None],
            blocks,
            is_leaf=lambda x: isinstance(x, list)
        )

        return blocks

    def init_state(self, t0: dt.datetime) -> State:
        """
        Initializes the observatory state with some reasonable guess.
        In practice it should ideally be replaced with actual data
        from the observatory controller.

        Parameters
        ----------
        t0 : float
            The initial time for the state, typically representing the current time in a specific format.

        Returns
        -------
        State
        """
        return State(
            curr_time=t0,
            az_now=180,
            el_now=48,
            boresight_rot_now=0,
            hwp_spinning=False,
        )

# ------------------------
# utilities
# ------------------------

def simplify_hwp(op_seq):
    # if hwp is spinning up and down right next to each other, we can just remove them
    core.seq_assert_sorted(op_seq)
    def rewriter(seq_prev, b_next):
        if len(seq_prev) == 0:
            return [b_next]
        b_prev = seq_prev[-1]
        if (b_prev.name == 'sat.hwp_spin_up' and b_next.name == 'sat.hwp_spin_down') or \
           (b_prev.name == 'sat.hwp_spin_down' and b_next.name == 'sat.hwp_spin_up'):
            return seq_prev[:-1] + [cmd.OperationBlock(
                name='wait-until', 
                t0=b_prev.t0, 
                t1=b_next.t1, 
            )]
        else:
            return seq_prev+[b_next]
    return reduce(rewriter, op_seq, [])
