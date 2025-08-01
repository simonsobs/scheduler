"""Optimization pass that resolves overlapping of operations and
make prioritization between calibration sequence and baseline sequence.

It begins by converting a sequence of ScanBlock into an intermediate
representation with each block surrounded by operation blocks.
This representation will be subject to several optimization at this
level without actually being lowered into commands.

"""
import numpy as np
import datetime as dt
import copy
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field, replace as dc_replace
from schedlib import core, commands as cmd, utils as u, rules as ru, instrument as inst
from schedlib.thirdparty.avoidance import get_sun_tracker
import socs.agents.acu.avoidance as avoidance

logger = u.init_logger(__name__)

def get_traj(az0, az1, el0, el1, wrap_north=False):
    # catching wrap questions
    if az0 < 180 and az1 >= 180: # ccw wrap
        if not wrap_north:
            mid_az = np.mean([az0, az1])
        else:
            az += -360.
            mid_az = np.mean([az0, az1])
    elif az0 > 180 and az1 <= 180: # cw wrap
        mid_az = np.mean([az0, az1])
    else:
        mid_az = az1

    mid_el = np.mean([el0, el1])

    return [az0, mid_az, az1], [el0, mid_el, el1]


def get_traj_ok_time(az0, az1, alt0, alt1, t0, sun_policy):
    # Returns the timestamp until which the move from
    # (az0, alt0) to (az1, alt1) is sunsafe.
    sun_tracker = get_sun_tracker(u.dt2ct(t0), policy=sun_policy)
    az = np.linspace(az0, az1, 101)
    el = np.linspace(alt0, alt1, 101)
    if alt0 != alt1 or az0 != az1:
        az1 = (az[:,None] + el[None,:]*0).ravel()
        el1 = (az[:,None]*0 + el[None,:]).ravel()
        az, el = az1, el1

    sun_safety = sun_tracker.check_trajectory(t=u.dt2ct(t0), az=az, el=el)
    if (sun_safety['sun_time'] < sun_policy['min_sun_time']
        or sun_safety['sun_dist_min'] < sun_policy['min_angle']):
        return t0

    return u.ct2dt(u.dt2ct(t0) + sun_safety['sun_time'])

def get_traj_ok_time_socs(az0, az1, alt0, alt1, t0, sun_policy, block0=None):
    # Returns the timestamp until which the move from (az0, alt0) to
    # (az1, alt1) is sunsafe using socs sun-safety checker.  If block0 is
    # passed it will check the start and end azimuths of a scan if it can
    # returns t0 if no move could be found
    policy = avoidance.DEFAULT_POLICY
    policy['min_el'] = sun_policy['min_el']
    policy['max_el'] = sun_policy['max_el']
    policy['min_az'] = sun_policy['min_az']
    policy['max_az'] = sun_policy['max_az']
    policy['min_sun_time'] = sun_policy['min_sun_time']
    policy['exclusion_radius'] = sun_policy['min_angle']

    sungod = avoidance.SunTracker(policy=policy, base_time=t0.timestamp()-100, compute=True)
    az_range, el_range = get_traj(az0, az1, alt0, alt1)

    d = sungod.check_trajectory(az_range, el_range, t=t0.timestamp())
    if d['sun_dist_min'] <= policy['exclusion_radius'] or d['sun_time'] <= policy['min_sun_time']:
        return t0

    moves = sungod.analyze_paths(az_range[0], el_range[0], az_range[-1], el_range[-1], t=t0.timestamp())
    move, _ = sungod.select_move(moves)

    if move is None:
        return t0

    if block0 is not None and hasattr(block0, 'block'):
        drifted_az = block0.block.az + block0.block.throw + block0.block.az_drift * ((block0.block.t1 - block0.block.t0).total_seconds())
        az_range, el_range = get_traj(drifted_az, az1, alt0, alt1)

        d = sungod.check_trajectory(az_range, el_range, t=t0.timestamp())
        if d['sun_dist_min'] <= policy['exclusion_radius'] or d['sun_time'] <= policy['min_sun_time']:
            return t0

        end_moves = sungod.analyze_paths(az_range[0], el_range[0], az_range[-1], el_range[-1], t=t0.timestamp())
        end_move, _ = sungod.select_move(end_moves)

        if end_move is None:
            return t0

    if block0 is not None and hasattr(block0, 'az_drift'):
        return u.ct2dt(u.dt2ct(t0) + min(move['sun_time'], end_move['sun_time']))
    else:
        return u.ct2dt(u.dt2ct(t0) + move['sun_time'])

def get_parking(t0, t1, alt0, sun_policy, az_parking=180, alt_parking=None):
    # gets a safe parking location for the time range and
    # Do the move in two steps, parking at az_parking (180
    # is most likely to be sun-safe).  Identify a spot that
    # is safe for the duration of t0 to t1.
    if alt_parking is None:
        alt_range = alt0, sun_policy['min_el']
        n_alts = max(2, int(round(abs(alt_range[1] - alt_range[0]) / 4. + 1)))
        trial_alts = np.linspace(alt_range[0], alt_range[1], n_alts)
    else:
        trial_alts = [alt_parking]

    for trial_alt in trial_alts:
        safet = get_traj_ok_time_socs(az_parking, az_parking, trial_alt, trial_alt,
                                      t0, sun_policy)
        if safet >= t1:
            break
    else:
        if alt_parking is None:
            raise ValueError(f"Sun-safe parking spot not found. az {az_parking} "
                                f"el {trial_alt} is safe only until {safet}")
        else:
            return None

    # Now bracket the moves, hopefully with ~5 minutes on each end.
    buffer_t = min(300, int((t1 - t0).total_seconds() / 2))
    t0_parking = t0 + dt.timedelta(seconds=buffer_t)
    t1_parking = t1 - dt.timedelta(seconds=buffer_t)

    return (az_parking, trial_alt, t0_parking, t1_parking)


def get_safe_gaps(block0, block1, sun_policy, el_limits, is_end=False, max_delay=300):
    # Check the move
    t1 = get_traj_ok_time_socs(block0.az, block1.az, block0.alt, block1.alt,
                               block0.t1, sun_policy, block0)
    # if previous block runs into or beyond next block
    if block0.t1 >= block1.t0:
        # allow for t1 = max(block0.t1, block1.t0) if same source,
        # otherwise strictly t1 > max(block0.t1, block1.t0)
        same_source = (block0.az==block1.az and block0.alt==block1.alt)
        if ((t1 >= max(block0.t1, block1.t0) and same_source)
            or (t1 > max(block0.t1, block1.t0) and not same_source)):
            return []

    if t1 > block1.t0 and (block0.t1 < block1.t0):
        return [IR(name='gap', subtype=IRMode.Gap, t0=block0.t1, t1=block1.t0,
                    az=block1.az, alt=block1.alt,
                    az_offset=block1.az_offset, alt_offset=block1.alt_offset)]

    # elevations to check. search in order of cur_el -> min_el then from cur_el -> max_el
    alt_step = 8
    if block0.alt <= 90:
        alt_lower = np.arange(block0.alt, sun_policy['min_el'] - alt_step, -alt_step)
        alt_lower[-1] = sun_policy['min_el']
    else:
        alt_lower = np.arange(block0.alt, el_limits[-1] + alt_step, alt_step)
        alt_lower[-1] = el_limits[-1]

    if block0.alt <= 90:
        alt_upper = np.arange(block0.alt + alt_step, el_limits[-1] + alt_step, alt_step)
        alt_upper[-1] = el_limits[-1]
    else:
        alt_upper = np.arange(block0.alt, sun_policy['min_el'] - alt_step, -alt_step)
        alt_upper[-1] = sun_policy['min_el']

    alt_range = np.concatenate((alt_lower, alt_upper))

    # check 180, next, and current azimuths for parking
    az_range = np.array([block0.az, block1.az, 180])

    _, idx = np.unique(az_range, return_index=True)
    az_range = az_range[np.sort(idx)]

    for az_test in az_range:
        for alt_test in alt_range:
            logger.info(f"trying to park at ({az_test}, {alt_test})")
            # try parking position at (az_test, alt_test)
            parking = get_parking(block0.t1, block1.t0, block0.alt, sun_policy,
                                    az_parking=az_test, alt_parking=alt_test)

            if parking is not None:
                az_parking, alt_parking, t0_parking, t1_parking = parking
            else:
                continue

            # you might need to rush away from final position...
            move_away_by = get_traj_ok_time_socs(block0.az, az_parking, block0.alt,
                                                 alt_parking, block0.t1, sun_policy,
                                                 block0)

            if move_away_by < t0_parking:
                if move_away_by <= block0.t1:
                    continue
                else:
                    t0_parking = move_away_by + (move_away_by - block0.t1) / 2

            # You might need to wait until the last second before going to new pos
            shift = 10.
            while t1_parking < block1.t0 + dt.timedelta(seconds=max_delay):
                ok_until = get_traj_ok_time_socs(
                    az_parking, block1.az, alt_parking, block1.alt, t1_parking, sun_policy, block1)
                if ok_until >= block1.t0 and ok_until > t1_parking:
                    break
                t1_parking = t1_parking + dt.timedelta(seconds=shift)
            else:
                continue

            if t1_parking > block1.t0:
                logger.warning("sun-safe parking delays move to next field by "
                            f"{(t1_parking - block1.t0).total_seconds()} seconds")

            return [IR(name='gap', subtype=IRMode.Gap, t0=block0.t1, t1=t0_parking,
                    az=az_parking, alt=alt_parking,
                    az_offset=block0.az_offset, alt_offset=block0.alt_offset),
                    IR(name='gap', subtype=IRMode.Gap, t0=t0_parking, t1=t1_parking,
                    az=az_parking, alt=alt_parking,
                    az_offset=block1.az_offset, alt_offset=block1.alt_offset),
                    IR(name='gap', subtype=IRMode.Gap, t0=t1_parking, t1=block1.t0,
                    az=block1.az, alt=block1.alt,
                    az_offset=block1.az_offset, alt_offset=block1.alt_offset),
                    ]

# some additional auxilary command classes that will be mixed
# into the IR to represent some intermediate operations. They
# don't need to contain all the fields of a regular IR
@dataclass(frozen=True)
class Aux: pass

@dataclass(frozen=True)
class MoveTo(Aux):
    az: float
    alt: float
    az_offset: float = 0.
    alt_offset: float = 0.
    subtype: str = "aux"
    def __repr__(self):
        return f"# move to az={self.az:.2f}"

@dataclass(frozen=True)
class WaitUntil(Aux):
    t1: dt.datetime
    az: float
    alt: float
    az_offset: float = 0.
    alt_offset: float = 0.
    subtype: str = "aux"
    def __repr__(self):
        return f"# wait until {self.t1} at az = {self.az:.2f}"

# full intermediate representation of operation used in this
# build stage
@dataclass(frozen=True)
class IR(core.Block):
    name: str
    subtype: str
    t0: dt.datetime
    t1: dt.datetime
    az: float
    alt: float
    block: Optional[core.Block] = field(default=None, hash=False, repr=False)
    operations: List[Dict[str, Any]] = field(default_factory=list, hash=False, repr=False)
    az_offset: float = 0.
    alt_offset: float = 0.

    def __repr__(self):
        az = f"{self.az:>7.2f}" if self.az is not None else f"{'None':>7}"
        return f"{self.name[:15]:<15} ({self.subtype[:8]:<8}) az = {az}: {self.t0.strftime('%y-%m-%d %H:%M:%S')} -> {self.t1.strftime('%y-%m-%d %H:%M:%S')}"

    def replace(self, **kwargs):
        """link `replace` in the wrapper block with the block it contains.
        Note that when IR is produced, we assume no trimming needs to happen,
        so we use `dc_replace` instead of `super().replace` which accounts for
        trimming effect on drift scans. It is not necessary here as we are
        merely solving for different unwraps for drift scan.

        We allow the option of changing the block or block.subtype here because
        sometimes we need to run a master schedule but mark things as
        calibration. Most important example: drone calibration campaigns
        """
        if self.block is not None and 'block' not in kwargs:
            block_kwargs = {k: v for k, v in kwargs.items() if k in ['t0', 't1', 'az', 'alt', 'subtype']}
            new_block = dc_replace(self.block, **block_kwargs)
            kwargs['block'] = new_block
        return dc_replace(self, **kwargs)

# some mode enums relevant for IR building in this stage
# didn't use built-in enum in python as they turn things into
# objects and not str.
class IRMode:
    PreSession = 'pre_session'
    PreBlock = 'pre_block'
    InBlock = 'in_block'
    PostBlock = 'post_block'
    PostSession = 'post_session'
    Gap = 'gap'
    Aux = 'aux'

# custom exceptions
class SunSafeError(Exception):
    def __init__(self, message, block0=None, block1=None):
        super().__init__(message)
        self.block0 = block0
        self.block1 = block1

    def __str__(self):
        base_message = super().__str__()
        if self.block0 and self.block1:
            return f"{base_message} (Block: {self.block} -> {self.block1})"
        elif self.block0:
            return f"{base_message} (Block: {self.block0})"
        else:
            return base_message

@dataclass(frozen=True)
class BuildOpSimple:
    """try to simplify the block -> op process logic"""
    policy_config: Dict[str, Any]
    max_pass: int = 3
    max_reject: int = 3
    min_duration: float = 1 * u.minute
    min_cmb_duration: float = 10 * u.minute
    plan_moves: Dict[str, Any] = field(default_factory=dict)
    simplify_moves: Dict[str, Any] = field(default_factory=dict)

    def merge_adjacent_blocks(self, seq, max_dt=dt.timedelta(minutes=60), min_dt=dt.timedelta(minutes=20)):
        for i in range(1, len(seq)):
            current, previous = seq[i], seq[i-1]
            # skip previously merged blocks
            if current is None or previous is None:
                continue
            # don't merge blocks that are too far apart in time
            time_gap = (current.t0 - previous.t1).total_seconds()
            combined_duration = (current.duration + previous.duration).total_seconds()
            max_combined_duration = (max_dt + min_dt).total_seconds()
            # if blocks were split from same block and are close in time
            if current.tag == previous.tag and time_gap <= min_dt.total_seconds():
                # don't merge blocks that are longer than the max length
                if combined_duration <= max_combined_duration:
                    seq[i-1] = previous.extend_right(current.duration)
                    seq[i] = None
                    # add some or all of time gaps (likely from det_setup)
                    if time_gap > 0:
                        if (seq[i-1].duration.total_seconds() + time_gap) <= max_combined_duration:
                            seq[i-1] = seq[i-1].extend_right(dt.timedelta(seconds=time_gap))
                        else:
                            seq[i-1] = seq[i-1].extend_right(dt.timedelta(seconds=(max_combined_duration - seq[i-1].duration.total_seconds())))
        return seq

    def apply(self, seq, t0, t1, state):
        init_state = state

        # when something fails to plan, we reject the block and try again
        # `max_reject` determines how many times we could do this before
        # giving up
        n_reject = 0
        reject_list = []
        while True:
            if len(reject_list) > 0:
                reject_block = reject_list.pop(0)
                logger.info(f"rejecting block: {reject_block}")
                seq_after_reject = [b for b in seq_ if b['block'] != reject_block]
                # find the block in seq_ right after the reject_block
                assert len(seq_after_reject) == len(seq_) - 1, "reject block failed, need investigation..."
                seq_ = seq_after_reject
            else:
                seq_ = seq

            for i in range(self.max_pass):
                logger.info(f"================ pass {i+1} ================")
                seq_new = self.round_trip(seq_, t0, t1, init_state)
                if seq_new == seq_:
                    logger.info(f"round_trip: converged in pass {i+1}, lowering...")
                    break
                seq_ = seq_new

                cmb_blocks = self.merge_adjacent_blocks([s['block'] for s in seq_ if s['block'].subtype == 'cmb'],
                             dt.timedelta(seconds=self.policy_config.max_cmb_scan_duration))

                seq_temp = []
                cmb_index = 0
                for s in seq_:
                    if s['block'].subtype == 'cmb':
                        if cmb_blocks[cmb_index] is not None:
                            s = s.copy()
                            s['block'] = cmb_blocks[cmb_index]
                            seq_temp.append(s)
                        cmb_index += 1
                    else:
                        seq_temp.append(s)

                seq_ = seq_temp
            else:
                logger.warning(f"round_trip: ir did not converge after {self.max_pass} passes, proceeding anyway")

            logger.info(f"================ lowering ================")

            ir = self.lower(seq_, t0, t1, init_state)
            assert ir[-1].t1 <= t1, "Going beyond our schedule limit, something is wrong!"

            logger.info(f"================ solve moves ================")
            logger.info("step 1: solve sun-safe moves")
            try:
                ir = PlanMoves(**self.plan_moves).apply(ir, t1)
            except SunSafeError as e:
                logger.exception(f"unable to plan sun-safe moves: {e}")

                # append to reject list
                # (latter block will be rejected first)
                if e.block1 is not None:
                    assert isinstance(e.block1, IR), f"unexpected block type: {e.block1}"
                    to_reject = e.block1.block  # dereference to original block
                    if to_reject not in reject_list:
                        reject_list.append(to_reject)
                if e.block0 is not None:
                    assert isinstance(e.block0, IR), f"unexpected block type: {e.block0}"
                    to_reject = e.block0.block  # dereference to original block
                    if to_reject not in reject_list:
                        reject_list.append(to_reject)

                n_reject += 1
                if n_reject >= self.max_reject:
                    logger.error(f"max reject reached, giving up")
                    raise e
            else:
                logger.info("sun-safe moves found, continuing...")
                break

        logger.info("step 2: simplify moves")
        ir = SimplifyMoves(**self.simplify_moves).apply(ir)

        # in full generality we should do some round-trips to make sure
        # the moves are still valid when we include the time costs of
        # moves. Here I'm working under the assumption that the moves
        # are very small and the time cost is negligible.

        # now we do lowering further into full ops
        logger.info(f"================ lowering (ops) ================")
        ir_ops, out_state = self.lower_ops(ir, init_state)
        logger.info(u.pformat(ir_ops))

        logger.info(f"================ done ================")

        return ir_ops, out_state

    def lower(self, seq, t0, t1, state):
        # group operations by priority
        priorities = sorted(list(set(b['priority'] for b in seq)), reverse=False)

        # process each priority group
        init_state = state

        for priority in priorities:
            logger.info(f"processing priority group: {priority}")
            state = init_state

            # update constraint to avoid overlapping with previously planned blocks
            seq_ir = [b for b in seq if isinstance(b, IR)]
            # if nestedness is used, we can use this
            # seq_ir = core.seq_sort(core.seq_filter(lambda b: isinstance(b, IR), seq), flatten=True)
            if len(seq_ir) > 0:
                constraints = core.seq_remove_overlap(core.Block(t0=t0, t1=t1), seq_ir)
            else:
                constraints = [core.Block(t0=t0, t1=t1)]

            seq_out = []
            for b in seq:
                # if it's already an planned, just execute it, otherwise plan it
                # if isinstance(b, list) and all(isinstance(x, IR) for x in b):
                #     for x in b:
                #         state, _, _ = self._apply_ops(state, x.operations, block=x.block)
                #     seq_out += [b]
                #     continue
                if isinstance(b, IR):
                    state, _, _ = self._apply_ops(state, b.operations, block=b.block)
                    seq_out += [b]
                    continue

                # what's our constraint? find the one that (partially) covers the block
                constraints_ = [x for x in constraints if core.block_overlap(x, b['block'])]
                if len(constraints_) == 0:
                    logger.info(f"--> block {b['block']} doesn't fit within constraint, skipping...")
                    continue

                # we always fit the operations within the largest window that covers the block in the constraint
                # i.e. find the window with largest overlap with the block
                constraint = sorted(constraints_, key=lambda x: core.block_intersect(x, b['block']).duration.total_seconds())[-1]

                # now plan the operations for the given block within our specified constraint
                state, ir = self._plan_block_operations(
                    state, block=b['block'], constraint=constraint,
                    pre_ops=b['pre'], post_ops=b['post'], in_ops=b['in'],
                    causal=not(b['priority'] == priority)
                )
                if len(ir) == 0:
                    logger.info(f"--> block {b['block']} has nothing that can be planned, skipping...")
                    continue

                # higher priority group is planned first, and the constraint is updated
                # to the end of the previously planned block
                if b['priority'] == priority:
                    logger.info(f"-> {b['name'][:5]:<5} ({b['block'].subtype:<3}): {b['block'].t0.strftime('%d-%m-%y %H:%M:%S')} -> {b['block'].t1.strftime('%d-%m-%y %H:%M:%S')}")
                    seq_out += ir
                    constraints = core.seq_flatten(core.seq_trim(constraints, t0=state.curr_time, t1=t1))
                elif b['priority'] > priority:
                    # lower priority item will pass through to be planned in the next round
                    seq_out += [b]
                else:
                    raise ValueError(f"unexpected priority: {b['priority']}")
            seq = seq_out

        return seq

    def round_trip(self, seq, t0, t1, state):
        """lower the sequence and lift it back to original data structure"""
        # 1. lower the sequence into IRs
        ir = self.lower(seq, t0, t1, state)

        # 2. lift the IRs back to the original data structure
        trimmed_blocks = core.seq_sort(
            core.seq_map(lambda b: b.block if b.subtype == IRMode.InBlock else None, ir),
            flatten=True
        )
        # match input blocks with trimmed blocks: since we are trimming the blocks
        # each block in ir should match one or none of the trimmed blocks.
        # this assumes no splitting is done in lowering process, which can be supported
        # but needs more work
        seq_out = []
        min_dur_filter = ru.MinDuration(self.min_duration)
        min_cmb_dur_filter = ru.MinDuration(self.min_cmb_duration)
        for b in seq:
            if b.get('pinned', False):
                seq_out += [b]
                continue
            matched = [x for x in trimmed_blocks if core.block_overlap(x, b['block'])]
            assert len(matched) <= 1, f"unexpected match: {matched=}"
            if len(matched) == 1:
                # does it meet our minimum duration requirement? drop if it doesn't
                # if min_dur_filter(matched[0]) == matched[0]:
                if min_dur_filter(matched[0]) == matched[0]:
                    if b['block'].subtype == 'cmb':
                        if min_cmb_dur_filter(matched[0]) == matched[0]:
                            b = b | {'block': matched[0]}
                            seq_out.append(b)
                        else:
                            logger.info(f"--> dropping {b['name']} due to min cmb duration requirement")
                    else:
                        b = b | {'block': matched[0]}
                        seq_out.append(b)
                else:
                    logger.info(f"--> dropping {b['name']} due to min duration requirement")
        return seq_out

    def _apply_ops(self, state, op_cfgs, block=None, az=None, alt=None):
        """
        Apply a series of operations to the current planning state, computing
        the updated state, the total duration, and resulting commands of the
        operations.

        Parameters
        ----------
        state : State
            The current planning state. It must be an object capable of tracking
            the mission's state and supporting time increment operations.
        op_cfgs : list of operation configs (dict)
            A list of operation configurations, where each configuration is a
            dictionary specifying the operation's parameters. Each operation
            configuration dict must have a 'sched_mode' key
        block : Optional[core.Block], optional
            per-block operations such as PreCal, PreObs, etc. require a block
            as part of the operation configuration.

        Returns
        -------
        new_state : State
            The new state after all operations have been applied.
        total_duration : int
            The total duration of all operations in seconds.
        commands : list of str
            A list of strings representing the commands generated by each
            operation. Commands are preconditioned by operation-specific
            indentation.

        """
        op_blocks = []
        duration = 0
        if (az is None or alt is None) and (block is not None):
            az, alt = block.az, block.alt

        for op_cfg_ in op_cfgs:
            op_cfg = op_cfg_.copy()

            # sanity check
            for k in ['name', 'sched_mode']:
                assert k in op_cfg, f"operation config must have a '{k}' key"

            # pop some non-operation kwargs
            op_name = op_cfg.pop('name')
            sched_mode = op_cfg.pop('sched_mode')

            # not needed now -> needed only during lowering
            op_cfg.pop('indent', None)
            op_cfg.pop('divider', None)

            # add block to the operation config if provided
            block_cfg = {'block': block} if block is not None else {}

            op_cfg = {**op_cfg, **block_cfg}  # make copy

            # apply operation
            t_start = state.curr_time
            op = cmd.make_op(op_name, **op_cfg)
            state, dur, _ = op(state)

            duration += dur
            state = state.increment_time(dt.timedelta(seconds=dur))

            op_blocks += [IR(
                name=op_name,
                subtype=sched_mode,
                t0=t_start,
                t1=state.curr_time,
                az=az,
                alt=alt,
                block=block,
                operations=[op_cfg_]
            )]

        return state, duration, op_blocks

    def _plan_block_operations(self, state, block, constraint,
                               pre_ops, in_ops, post_ops, causal=True):
        """
        Plan block operations based on the current state, block information, constraint, and operational sequences.

        The function takes in sequences of operations to be planned before, within, and after the block, and returns the
        updated state and the planned sequence of operations.

        Parameters
        ----------
        state : State
            The current state of the system.
        block : Block or list of Block
            Block information containing start and end times.
        constraint : Block
            Constraint information containing start and end times.
        pre_ops : list
            List of operations to be planned immediately before block.t0.
        in_ops : list
            List of operations to be planned within the block, i.e., from block.t0 to block.t1.
        post_ops : list
            List of operations to be planned immediately after block.t1.

        Returns
        -------
        state : State
            The updated state after planning the block operations.
        planned_sequence : list of IR
            The sequence of operations planned for the block.

        """
        # fast forward to within the constrained time block
        # state = state.replace(curr_time=min(constraint.t0, block.t0))
        # - during causal planning: fast forward state is allowed
        # - during non-causal planning (e.g. during prioritized planning):
        #   time backtracking is allowed
        if causal:
            state = state.replace(curr_time=max(constraint.t0, state.curr_time))
        else:
            state = state.replace(curr_time=constraint.t0) # min(constraint.t0, block.t0))

        # if we already pass the block or our constraint, nothing to do
        if state.curr_time >= block.t1 or state.curr_time >= constraint.t1:
            logger.info(f"--> skipping block {block.name} because it's already past")
            return state, []

        shift = 10
        safet = get_traj_ok_time(block.az, block.az, block.alt, block.alt, state.curr_time, self.plan_moves['sun_policy'])
        while safet <= state.curr_time:
            state = state.replace(curr_time=state.curr_time + dt.timedelta(seconds=shift))
            safet = get_traj_ok_time(block.az, block.az, block.alt, block.alt, state.curr_time, self.plan_moves['sun_policy'])

        initial_state = state

        logger.debug(f"--> with constraint: planning {block.name} from {state.curr_time} to {block.t1}")

        op_seq = []

        # +++++++++++++++++++++
        # pre-block operations
        # +++++++++++++++++++++

        logger.debug(f"--> planning pre-block operations")

        state, pre_dur, _ = self._apply_ops(state, pre_ops, block=block)

        logger.debug(f"---> pre-block ops duration: {pre_dur} seconds")
        logger.debug(f"---> pre-block curr state: {u.pformat(state)}")

        # what time are we starting?
        # -> start from t_start or block.t0-duration, whichever is later
        # -> overwrite block if we extended into the block
        # -> if we extended past the block, skip operation

        # did we extend into the block?
        if state.curr_time > block.t0:
            logger.debug(f"---> curr_time extended into block {block.name}")
            # did we extend past entire block?
            if state.curr_time < block.t1:
                logger.debug(f"---> curr_time did not extend past block {block.name}")
                delta_t = (state.curr_time - block.t0).total_seconds()
                block = block.trim_left_to(state.curr_time)
                logger.debug(f"---> trimmed block: {block}")
                pre_block_name = "pre_block (into)"
                logger.info(f"--> trimming left by {delta_t} seconds to fit pre-block operations")
            else:
                logger.info(f"--> not enough time for pre-block operations for {block.name}, skipping...")
                return initial_state, []
        else:
            logger.debug(f"---> gap is large enough for pre-block operations")
            state = state.replace(curr_time=block.t0)
            pre_block_name = "pre_block"

        logger.debug(f"--> post pre-block state: {u.pformat(state)}")
        logger.debug(f"--> post pre-block op_seq: {u.pformat(op_seq)}")

        # +++++++++++++++++++
        # in-block operations
        # +++++++++++++++++++

        logger.debug(f"--> planning in-block operations from {state.curr_time} to {block.t1}")
        logger.debug(f"--> pre-planning state: {u.pformat(state)}")

        state, in_dur, _ = self._apply_ops(state, in_ops, block=block)

        logger.debug(f"---> in-block ops duration: {in_dur} seconds")
        logger.debug(f"---> in-block curr state: {u.pformat(state)}")

        # sanity check: if fail, it means post-cal operations are
        # mixed into in-cal operations
        assert state.curr_time <= block.t1, \
            "in-block operations are probably mixed with post-cal operations"

        # advance to the end of the block
        state = state.replace(curr_time=block.t1)

        logger.debug(f"---> post in-block state: {u.pformat(state)}")

        # +++++++++++++++++++++
        # post-block operations
        # +++++++++++++++++++++

        state, post_dur, _ = self._apply_ops(state, post_ops, block=block)

        logger.debug(f"---> post-block ops duration: {post_dur} seconds")
        logger.debug(f"---> post-block curr state: {u.pformat(state)}")

        # have we extended past our constraint?
        post_block_name = "post_block"
        if state.curr_time > constraint.t1:
            logger.debug(f"---> post-block ops extended past constraint")
            # shrink our block to make space for post-block operation and
            # revert to an old state before retrying
            delta_t = (state.curr_time - constraint.t1).total_seconds()
            block = block.shrink_right(state.curr_time - constraint.t1)

            # if we extends passed the block.t0, there is not enough time to do anything
            # -> revert to initial state
            logger.info(f"--> trimming right by {delta_t} seconds to fit post-block operations")
            if block is None:
                logger.info(f"--> skipping because post-block op couldn't fit inside constraint")
                return initial_state, []
            post_block_name = "post_block (into)"
            state = state.replace(curr_time=constraint.t1)

        # block has been trimmed properly, so we can just do this
        if len(pre_ops) > 0:
            op_seq += [
                IR(name=pre_block_name,
                subtype=IRMode.PreBlock,
                t0=block.t0-dt.timedelta(seconds=pre_dur),
                t1=block.t0,
                az=block.az,
                alt=block.alt,
                az_offset=block.az_offset,
                alt_offset=block.alt_offset,
                block=block,
                operations=pre_ops),
            ]
        if len(in_ops) > 0:
            op_seq += [
                IR(name=block.name,
                subtype=IRMode.InBlock,
                t0=block.t0,
                t1=block.t1,
                az=block.az,
                alt=block.alt,
                az_offset=block.az_offset,
                alt_offset=block.alt_offset,
                block=block,
                operations=in_ops),
            ]
        if len(post_ops) > 0:
            op_seq += [
                IR(name=post_block_name,
                subtype=IRMode.PostBlock,
                t0=block.t1,
                t1=block.t1+dt.timedelta(seconds=post_dur),
                az=block.az,
                alt=block.alt,
                az_offset=block.az_offset,
                alt_offset=block.alt_offset,
                block=block,
                operations=post_ops)
            ]

        return state, op_seq

    def lower_ops(self, irs, state):
        # `lower` generates a basic plan, here we work with ir to resolve
        # all operations within each blocks
        def resolve_block(state, ir):
            if isinstance(ir, WaitUntil):
                op_cfgs = [{'name': 'wait_until', 'sched_mode': IRMode.Aux, 't1': ir.t1}]
                state, _, op_blocks = self._apply_ops(state, op_cfgs, az=ir.az, alt=ir.alt)
            elif isinstance(ir, MoveTo):
                # aux move_to should be enforced'
                op_cfgs = [{'name': 'move_to', 'sched_mode': IRMode.Aux, 'az': ir.az, 'el': ir.alt,
                            'az_offset': ir.az_offset, 'el_offset': ir.alt_offset, 'force': True}]
                # add min hwp elevation if present
                if hasattr(self.policy_config, 'min_hwp_el'):
                    op_cfgs[0]['min_el'] = self.policy_config.min_hwp_el
                if hasattr(self.policy_config, 'brake_hwp'):
                    op_cfgs[0]['brake_hwp'] = self.policy_config.brake_hwp
                state, _, op_blocks = self._apply_ops(state, op_cfgs, az=ir.az, alt=ir.alt)
            elif ir.subtype in [IRMode.PreBlock, IRMode.InBlock, IRMode.PostBlock]:
                if ir.block.name in ['pre-session', 'post-session']:
                    state, _, op_blocks = self._apply_ops(state, ir.operations, az=ir.az, alt=ir.alt, block=ir.block)
                else:
                    tolerance=dt.timedelta(seconds=0)
                    if ir.subtype == IRMode.PreBlock:
                        wait_time = ir.t0
                        # add a tolerance for wait commands during scan pre-blocks
                        tolerance=dt.timedelta(seconds=1200)
                    elif ir.subtype == IRMode.InBlock:
                        wait_time = ir.block.t0
                    else:
                         wait_time = ir.block.t1
                    op_cfgs = [{'name': 'wait_until', 'sched_mode': IRMode.Aux, 't1': wait_time, 'tolerance': tolerance}]
                    state, _, op_blocks_wait = self._apply_ops(state, op_cfgs, az=ir.az, alt=ir.alt)
                    state, _, op_blocks_cmd = self._apply_ops(state, ir.operations, block=ir.block)
                    op_blocks = op_blocks_wait + op_blocks_cmd
            elif ir.subtype == IRMode.Gap:
                op_cfgs = [{'name': 'wait_until', 'sched_mode': IRMode.Gap, 't1': ir.t1}]
                state, _, op_blocks = self._apply_ops(state, op_cfgs, az=ir.az, alt=ir.alt)
            else:
                raise ValueError(f"unexpected block type: {ir}")
            return state, op_blocks

        ir_lowered = []
        for ir in irs:
            state, op_blocks = resolve_block(state, ir)
            ir_lowered += op_blocks
        return ir_lowered, state

@dataclass(frozen=True)
class PlanMoves:
    """solve moves to make seq possible"""
    sun_policy: Dict[str, Any]
    stow_position: Dict[str, Any]
    el_limits: Tuple[float, float]
    az_step: float = 1
    az_limits: Tuple[float, float] = (-90, 450)

    def apply(self, seq, t_end):
        """take a list of IR from BuildOp as input to solve for optimal sun-safe moves"""

        seq = core.seq_sort(seq, flatten=True)

        # go through the sequence and wrap az if falls outside limits
        logger.info(f"checking if az falls outside limits")
        seq_ = []
        for b in seq:
            drifted_az = b.az + b.block.throw + b.block.az_drift * ((b.t1 - b.t0).total_seconds())
            if b.az < self.az_limits[0] or b.az > self.az_limits[1] or (drifted_az < self.az_limits[0]) or (drifted_az > self.az_limits[1]):
                logger.info(f"block az ({b.az}) outside limits, unwrapping...")
                az_unwrap = find_unwrap(b.az, az_limits=self.az_limits)[0]
                logger.info(f"-> unwrapping az: {b.az} -> {az_unwrap}")
                seq_ += [b.replace(az=az_unwrap)]
            else:
                seq_ += [b]
        seq = seq_

        logger.info(f"planning moves...")
        seq_ = [seq[0]]
        for i in range(1, len(seq)):
            gaps = get_safe_gaps(seq[i-1], seq[i], self.sun_policy, self.el_limits,
                                 is_end=(i==(len(seq)-1)), max_delay=0)
            if gaps is None:
                # repeat with 20 minute delay
                gaps = get_safe_gaps(seq[i-1], seq[i], self.sun_policy, self.el_limits,
                                is_end=(i==(len(seq)-1)), max_delay=1200)
            if gaps is None:
                raise ValueError("No sun-safe gap found between {seq[i-1]} and {seq[i]}")
            seq_.extend(gaps)
            seq_.append(seq[i])

        # find sun-safe parking if not stowing at end of schedule
        if seq[-1].block.name != 'post-session':
            block = seq[-1]
            safet = get_traj_ok_time(block.az, block.az, block.alt, block.alt, block.t1, self.sun_policy)
            # if current position is safe until end of schedule
            if safet >= t_end:
                seq_.extend([IR(name='gap', subtype=IRMode.Gap, t0=block.t1, t1=t_end,
                    az=block.az, alt=block.alt, az_offset=block.az_offset, alt_offset=block.alt_offset)])
            else:
                movet = block.t1 #max(safet, block.t1)
                buffer_t = dt.timedelta(seconds=min(300, int((t_end - movet).total_seconds() / 2)))
                parking = get_parking(movet, t_end + buffer_t, block.alt, self.sun_policy)

                if parking is not None:
                    az_parking, alt_parking, t0_parking, t1_parking = parking
                else:
                    raise ValueError(f"Sun-safe parking spot not found for {block}.")

                move_away_by = get_traj_ok_time(
                    block.az, az_parking, block.alt, alt_parking, movet, self.sun_policy)
                if move_away_by < t0_parking:
                    if move_away_by < movet:
                        raise ValueError("Sun-safe parking spot not accessible from prior scan.")
                    else:
                        t0_parking = move_away_by + (move_away_by - movet) / 2

                seq_.append(IR(name='gap', subtype=IRMode.Gap, t0=t0_parking, t1=t1_parking,
                        az=az_parking, alt=alt_parking,
                        az_offset=block.az_offset, alt_offset=block.alt_offset))

        # Replace gaps with Wait, Move, Wait.
        seq_, seq = [], seq_
        last_az, last_alt = None, None

        # Combine, but skipping first and last blocks, which are init/shutdown.
        for i, b in enumerate(seq):
            if b.block is not None and b.block.name in ['pre-session']:
                # Pre/post-ambles, leave it alone.
                seq_ += [b]
                continue
            elif b.name == 'gap':
                # For a gap, always seek to the stated gap position.
                # But not until the gap is supposed to start.  Since
                # gaps may be used to manage Sun Avoidance, it's
                # important to be in that place for that time period.
                seq_ += [
                    WaitUntil(t1=b.t0, az=b.az, alt=b.alt),
                    MoveTo(az=b.az, alt=b.alt, az_offset=b.az_offset, alt_offset=b.alt_offset),
                    WaitUntil(t1=b.t1, az=b.az, alt=b.alt)]
                last_az, last_alt = b.az, b.alt
            else:
                if (last_az is None
                    or np.round(b.az - last_az, 3) != 0
                    or np.round(b.alt - last_alt, 3) != 0):
                    seq_ += [MoveTo(az=b.az, alt=b.alt, az_offset=b.az_offset, alt_offset=b.alt_offset)]
                    last_az, last_alt = b.az, b.alt
                else:
                    if (b.block != seq[i-1].block) & (i>0):
                        seq_ += [MoveTo(az=b.az, alt=b.alt, az_offset=b.az_offset, alt_offset=b.alt_offset)]
                seq_ += [b]

        return seq_


@dataclass(frozen=True)
class SimplifyMoves:
    def apply(self, ir):
        """simplify moves by removing redundant MoveTo blocks"""
        i_pass = 0
        while True:
            logger.info(f"simplify_moves: {i_pass=}")
            ir_new = self.round_trip(ir)
            #ir_new = ir
            if ir_new == ir:
                logger.info("simplify_moves: IR converged")
                return ir
            ir = ir_new
            i_pass += 1

    def round_trip(self, ir):
        def without(i):
            return ir[:i] + ir[i+1:]
        for bi in range(len(ir)-1):
            b1, b2 = ir[bi], ir[bi+1]
            if isinstance(b1, MoveTo) and isinstance(b2, MoveTo):
                # repeated moves will be replaced by the last move
                return without(bi)
            elif isinstance(b1, WaitUntil) and isinstance(b2, WaitUntil):
                # repeated wait untils will be replaced by the longer wait
                if b1.t1 < b2.t1:
                    return without(bi)
                return without(bi+1)
            elif (isinstance(b1, IR) and b1.subtype == IRMode.Gap) and isinstance(b2, WaitUntil):
                # gap followed by wait until will be replaced by the wait until
                return without(bi)
            # remove redundant move->wait->move if they are all at the same az/alt
            for bi in range(len(ir) - 3):
                if (
                    isinstance(ir[bi], MoveTo)
                    and isinstance(ir[bi + 2], MoveTo)
                    and ir[bi] == ir[bi + 2]
                    and isinstance(ir[bi + 1], WaitUntil)
                ):
                    return ir[:bi + 1] + ir[bi + 3:]

        return ir


def find_unwrap(az, az_limits) -> List[float]:
    az = (az - az_limits[0]) % 360 + az_limits[0]  # min az becomes az_limits[0]
    az_unwraps = list(np.arange(az, az_limits[1], 360))
    return az_unwraps

def az_ranges_intersect(
    r1: List[Tuple[float, float]],
    r2: List[Tuple[float, float]],
    *,
    az_limits: Tuple[float, float],
    az_step: float
) -> List[Tuple[float, float]]:
    az_full = np.arange(az_limits[0], az_limits[1]+az_step, az_step)
    mask1 = np.zeros_like(az_full, dtype=bool)
    mask2 = np.zeros_like(az_full, dtype=bool)
    for s in r1:
        mask1 = np.logical_or(mask1, (az_full >= s[0]) * (az_full <= s[1]))
    for s in r2:
        mask2 = np.logical_or(mask2, (az_full >= s[0]) * (az_full <= s[1]))
    sints_both = [(az_full[s[0]], az_full[s[1]-1]) for s in u.mask2ranges(mask1*mask2)]
    return sints_both

def az_distance(az1: float, az2: float) -> float:
    return abs(az1 - az2)
    # return abs((az1 - az2 + 180) % 360 - 180)

def az_ranges_contain(ranges: List[Tuple[float,float]], az: float) -> bool:
    for r in ranges:
        if r[0] <= az <= r[1]:
            return True
    return False

def az_ranges_cover(ranges: List[Tuple[float,float]], range_: Tuple[float, float]) -> bool:
    for r in ranges:
        if r[0] <= range_[0] and r[1] >= range_[1]:
            return True
    return False
