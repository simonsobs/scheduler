import numpy as np
import datetime as dt
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, replace as dc_replace
from dataclasses_json import dataclass_json

from abc import ABC, abstractmethod
import inspect

from . import core, utils as u
logger = u.init_logger(__name__)


MIN_DURATION = 0.01
HWP_SPIN_UP = 7*u.minute
HWP_SPIN_DOWN = 15*u.minute

@dataclass_json
@dataclass(frozen=True)
class State:
    """
    A dataclass representing the state of an observatory at a specific time. It should
    contain all relevant information for planning operations,

    Parameters
    ----------
    curr_time : datetime.datetime
        The current timestamp of the state.
    az_now : float
        The current azimuth position in degrees.
    el_now : float
        The current elevation position in degrees.
    az_speed_now : Optional[float], optional
        The current azimuth speed in degrees per second. Default is None.
    az_accel_now : Optional[float], optional
        The current azimuth acceleration in degrees per second squared. Default is None.
    prev_state : Optional[State], optional
        A reference to the previous state for tracking state evolution. Default is None and not included in string representation.

    Methods
    -------
    replace(**kwargs)
        Returns a new State instance with the specified attributes replaced with new values.
    increment_time(dt)
        Returns a new State instance with the current time incremented by a datetime.timedelta.
    increment_time_sec(dt_sec)
        Returns a new State instance with the current time incremented by a specific number of seconds.

    """
    curr_time: dt.datetime
    az_now: Optional[float] = None
    el_now: Optional[float] = None
    az_speed_now: Optional[float] = None
    az_accel_now: Optional[float] = None
    prev_state: Optional["State"] = field(default=None, repr=False)

    def clear_history(self):
        return dc_replace(self, **{"prev_state": None} )

    def save(self, fname):
        out = self.clear_history()
        np.save( fname, out.to_dict() )

    @classmethod
    def load(cls, fname):
        return cls.from_dict( np.load(fname, allow_pickle=True).item() )

    def replace(self, **kwargs):
        """
        Creates a new instance of the State class with specified attributes replaced with new values,
        while automatically setting 'prev_state' to the current state to maintain history.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments corresponding to the attributes of the State class which are to be replaced.

        Returns
        -------
        State
            A new instance of the State class with updated attributes.

        Examples
        --------
        >>> new_state = old_state.replace(az_now=180)
        """
        kwargs = {**kwargs, "prev_state": self}
        return dc_replace(self, **kwargs)

    def increment_time(self, dt: dt.timedelta) -> "State":
        """
        Returns a new State instance with the current time incremented by a specified datetime.timedelta.

        Parameters
        ----------
        dt : datetime.timedelta
            The amount of time to increment the current time.

        Returns
        -------
        State
            A new State instance with the incremented current time.
        """
        return self.replace(curr_time=self.curr_time+dt)

    def increment_time_sec(self, dt_sec: float) -> "State":
        """
        Increments the current time of the State by a specified number of seconds
        and returns a new State instance with this updated time.

        Parameters
        ----------
        dt_sec : float
            The number of seconds to increment the current time by.

        Returns
        -------
        State
            A new State instance with the current time incremented by the specified number of seconds.
        """
        return self.replace(curr_time=self.curr_time+dt.timedelta(seconds=dt_sec))

# -------------------------------------------------------------------------
#                         Register operations
# -------------------------------------------------------------------------
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
#

@dataclass(frozen=True)
class Operation(ABC):
    """
    An operation is a callable object that produces a set of commands to be executed by the observatory.

    This block takes an observatory state (dict) as input and returns a tuple containing a new state object
    as the effect of the command, and a list of commands for the telescope. Each command in the list is a
    string-like object.

    Parameters
    ----------
    state : object
        The current state of the observatory.

    Returns
    -------
    tuple
        A tuple containing a new state, an estimated duration, and a list of command strings.

    Notes
    -----
    This is a base class for all operations. It should not be used directly.
    state will be mutated by the operation in place, therefore no "parallelism"
    is allowed on the same state object.

    Examples
    --------
    op = Operation()
    state = get_init_state()
    duration, commands = op(state)

    """
    @abstractmethod
    def __call__(self, state: State) -> Tuple[State, float, List[str]]:
        ...

OPERATION_REGISTRY = {}

def register_operation_cls(name, operation_cls):
    """
    Register a new operation in the operation registry.

    Parameters
    ----------
    name : str
        The name of the operation to register.
    operation_cls : class
        The class of the operation to register.

    Returns
    -------
    class
        The class of the operation that was registered.

    """
    if name in OPERATION_REGISTRY:
        logger.warning(f"Operation {name} already exists in the registry.")
    OPERATION_REGISTRY[name] = operation_cls
    return operation_cls

def get_operation_cls(name):
    """
    Get an operation class from the operation registry.

    Parameters
    ----------
    name : str
        The name of the operation to retrieve.

    Returns
    -------
    class
        The class of the operation that was retrieved.

    """
    if name not in OPERATION_REGISTRY:
        raise ValueError(f"Operation {name} does not exist in the registry.")
    return OPERATION_REGISTRY[name]

def operation_cls(name):
    """
    A decorator that registers an operation class in the operation registry.

    Parameters
    ----------
    name : str
        The name of the operation to register.

    Returns
    -------
    class
        The class of the operation that was registered.

    """
    def wrapper(operation_cls):
        return register_operation_cls(name, operation_cls)
    return wrapper

def operation(name, duration=0, return_duration=False):
    """
    A decorator that registers a function as an operation
    in the operation registry.

    Parameters
    ----------
    name : str
        The name of the operation to register.

    Returns
    -------
    class
        The class of the operation that was registered.

    """
    def wrapper(operation_fun):
        # check whether the operation function wants to update the state dict
        sig = inspect.signature(operation_fun)
        args = list(sig.parameters.keys())
        alter_state = len(args) > 0 and args[0] == "state"

        # sometimes the block is automatically passed in for
        # per-block operations, but the operation may not needed it.
        # here we will explicitly remove it from the kwargs if it is
        # not in the original function signature
        want_block = "block" in args

        class _Operation(Operation):
            def __init__(self, *args, **kwargs):
                self.args = args
                if not want_block: kwargs.pop("block", None)
                self.kwargs = kwargs
            def __call__(self, state):
                # decide whether duration is provided or will be computed
                # as part of the operation function
                _duration = duration
                if alter_state:
                    state, *rest = operation_fun(state, *self.args, **self.kwargs)
                    if len(rest) == 1: rest = rest[0]
                else:
                    rest = operation_fun(*self.args, **self.kwargs)

                if return_duration:
                    _duration, commands = rest
                else:
                    commands = rest

                # avoid 0 duration to prevent sorting problems
                _duration = max(MIN_DURATION, _duration)
                return state, _duration, commands

        return register_operation_cls(name, _Operation)

    return wrapper

def make_op(name, *args, **kwargs):
    """
    Make an operation object from the operation registry.

    Parameters
    ----------
    name : str
        The name of the operation to create.
    *args : list
        The arguments to pass to the operation constructor.
    **kwargs : dict
        The keyword arguments to pass to the operation constructor.

    Returns
    -------
    Operation
        An operation object.

    """
    op_cls = get_operation_cls(name)
    return op_cls(*args, **kwargs)

# common operations
@operation(name='wait_until', return_duration=True)
def wait_until(state, t1: dt.datetime):
    return state, max((t1-state.curr_time).total_seconds(), 0), [
        f"run.wait_until('{t1.isoformat()}')"
    ]

@operation(name='start_time')
def start_time(state):
    return state, [
        f"run.wait_until('{state.curr_time.isoformat()}', tolerance=3600)"
    ]

@operation(name="move_to", return_duration=True)
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

@operation(name='set_scan_params', duration=0)
def set_scan_params(state, az_speed, az_accel):
    if az_speed != state.az_speed_now or az_accel != state.az_accel_now:
        state = state.replace(az_speed_now=az_speed, az_accel_now=az_accel)
        return state, [ f"run.acu.set_scan_params({az_speed}, {az_accel})"]
    return state, []
