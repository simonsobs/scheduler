from __future__ import annotations
from jax import tree_util as tu
import pandas as pd
from typing import List, TypeVar, Union, Dict, Optional
import numpy as np
from functools import reduce
from dataclasses import dataclass
from so3g.proj import quat

from . import core, utils as u


@dataclass(frozen=True)
class ScanBlock(core.NamedBlock):
    """
    Dataclass representing a scan block.

    Parameters
    ----------
    az : float
        Azimuth angle in degrees.
    alt : float
        Altitude angle in degrees.
    throw : float
        Throw angle in degrees.
    az_drift : float, optional
        Azimuth drift rate in degrees per second (default is 0).
    az_speed : float, optional
        Azimuth speed in degrees per second (default is 1).
    az_accel : float, optional
        Azimuth acceleration in degrees per second squared (default is 2).
    boresight_angle : float, optional
        Boresight angle in degrees (default is None).
    hwp_dir : bool, optional
        HWP direction for SATs. Forward is True, backwards if False.
        Default is None.
    subtype : str, optional
        Subtype of the scan block (default is an empty string).
    tag : str, optional
        Tag for the scan block (default is an empty string).
    """
    az: float        # deg
    alt: float       # deg
    throw: float     # deg
    az_drift: float = 0. # deg / s
    az_speed: float = 1. # deg / s
    az_accel: float = 2. # deg / s**2
    boresight_angle: Optional[float] = None # deg
    hwp_dir: Optional[bool] = None
    subtype: str = ""
    tag: str = ""

    def replace(self, **kwargs) -> "ScanBlock":
        """
        Update the parameters of the ScanBlock and ensure consistency between azimuth and drift when t0 is changed.

        Parameters
        ----------
        kwargs : keyword arguments
            Keyword arguments representing the parameters to update.

        Returns
        -------
        ScanBlock
            Updated ScanBlock object.

        Notes
        -----
        Updating the t0 parameter will trigger the recalculation of azimuth (az) based on drift (az_drift) if it is not zero. 
        If azimuth (az) is also provided in the kwargs, the consistency between azimuth and t0 will be checked.

        """
        # when t0 is changed, az should be updated to reflect the drift
        if "t0" in kwargs and self.az_drift != 0:
            new_az = (kwargs["t0"] - self.t0).total_seconds() * self.az_drift + self.az
            if "az" in kwargs:
                assert np.isclose(kwargs["az"], new_az), "inconsistent az and t0"
            kwargs['az'] = new_az
        return super().replace(**kwargs)

    def get_az_alt(self, time_step=1, ctimes=None):
        """
        Calculate the azimuth and altitude for the scan block.

        Parameters
        ----------
        time_step : float, optional
            The time step between each calculated azimuth and altitude.
            Default is 1.
        ctimes : iterable, optional
            A list of times to calculate the azimuth and altitude for.
            Default is None, in which case the times are calculated
            automatically. Otherwise time_step is ignored.

        Returns
        -------
        t : numpy.ndarray
            A 1D array of times.
        az : numpy.ndarray
            A 1D array of azimuths.
        alt : numpy.ndarray
            A 1D array of altitudes.

        """
        t0, t1 = u.dt2ct(self.t0), u.dt2ct(self.t1)
        # Additional buffer distance for turn-arounds (like ACU Agent)
        turn = self.az_speed**2 / self.az_accel

        # allow passing in a list of ctimes
        if ctimes is not None:
            t = ctimes
        else:
            t = np.arange(t0, t1+time_step, time_step)  # inclusive

        # find left and right az limits, accounting for drift
        drift = self.az_drift * (t-t0)
        left = self.az + drift - turn
        right = self.az + self.throw + drift + turn

        # calculate the phase of the scan, assuming it
        # moves at a constant speed from az to az+throw
        phase = (t - t0) / (self.throw / self.az_speed) % 2
        phase[m] = 2 - phase[(m:=(phase>1))]
        az = left*(1-phase) + right*phase
        return t, az, az*0 + self.alt

    def get_az_alt_extent(self, time_step=60, ctimes=None):
        """Calculate the azimuth and altitude extent for a scan block.

        Parameters
        ----------
        time_step : float, optional
            The time step between each calculated azimuth and altitude.
            Default is 1.
        ctimes : iterable, optional
            A list of times to calculate the azimuth and altitude for.
            Default is None, in which case the times are calculated
            automatically. Otherwise time_step is ignored.

        Returns
        -------
        t : numpy.ndarray
            A 1D array of times.
        az_min : numpy.ndarray
            A 1D array of min azimuth that might be hit at time t.
        az_max : numpy.ndarray
            A 1D array of max azimuth that might be hit at time t.
        alt_min : numpy.ndarray
            A 1D array of min altitude that might be hit at time t.
        alt_max : numpy.ndarray
            A 1D array of max altitude that might be hit at time t.

        """
        t0, t1 = u.dt2ct(self.t0), u.dt2ct(self.t1)
        # Additional buffer distance for turn-arounds (like ACU Agent)
        turn = self.az_speed**2 / self.az_accel

        # allow passing in a list of ctimes
        if ctimes is not None:
            t = ctimes
        else:
            t = np.arange(t0, t1+time_step, time_step)  # inclusive

        # find left and right az limits, accounting for drift
        drift = self.az_drift * (t-t0)
        left = self.az + drift - turn
        right = self.az + self.throw + drift + turn
        alt = self.alt + 0*t
        return t, left, right, alt, alt

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, {self.t0.strftime('%y-%m-%d %H:%M:%S')} -> {self.t1.strftime('%y-%m-%d %H:%M:%S')}, az={self.az:.2f}, el={self.alt:.2f}, throw={self.throw:.2f}, drift={self.az_drift:.5f}))"

@dataclass(frozen=True)
class StareBlock(ScanBlock):
    throw: float = 0.

    def get_az_alt(self, time_step=1, ctimes=None):
        """
        Calculate the azimuth and altitude for the scan block.

        Parameters
        ----------
        time_step : float, optional
            The time step between each calculated azimuth and altitude.
            Default is 1.
        ctimes : iterable, optional
            A list of times to calculate the azimuth and altitude for.
            Default is None, in which case the times are calculated
            automatically. Otherwise time_step is ignored.

        Returns
        -------
        t : numpy.ndarray
            A 1D array of times.
        az : numpy.ndarray
            A 1D array of azimuths.
        alt : numpy.ndarray
            A 1D array of altitudes.

        """
        t0, t1 = u.dt2ct(self.t0), u.dt2ct(self.t1)

        # allow passing in a list of ctimes
        if ctimes is not None:
            t = ctimes
        else:
            t = np.arange(t0, t1+time_step, time_step)  # inclusive

        return t, t*0+self.az, t*0+self.alt

# dummy type variable for readability
Spec = TypeVar('Spec')
SpecsTree = Dict[str, Union[Spec, "SpecsTree"]]

# SpecsTree can be an arbitrarily nested dict, with each leaf node being a dict
# with the following keys:
#  {
#       'bounds_x: [-1.0, 1.0],
#       'bounds_y: [-1.0, 1.0],
#  }
# To query a specific set of spec, use get_spec with a list of strings as query, where
# each string will be matched with the dot-separated path of a leaf node. Leaves that
# match *any* of the provided queries will be collected and reduced to a single leaf
# node.
def get_spec(specs: SpecsTree, query: List[str], merge=True) -> Union[Spec, SpecsTree]:
    """return a reduced spec (if merge=True) from all specs that match
    one of the queries. return all matches if merge=False"""
    is_leaf = lambda x: isinstance(x, dict) and 'bounds_x' in x
    match_p = lambda key: any([p in key for p in query])
    def reduce_fn(l, r):
        res = {}
        for k in ['bounds_x', 'bounds_y']:
            res[k] = [min(l[k][0], r[k][0]), max(l[k][1], r[k][1])]
        return res
    all_matches = tu.tree_leaves(
        tu.tree_map_with_path(lambda path, x: x if match_p(u.path2key(path)) else None, specs, is_leaf=is_leaf), 
        is_leaf=is_leaf
    )  # None is not a leaf, so it will be filtered out
    if not merge: return all_matches
    if len(all_matches) == 0: return {}
    return reduce(reduce_fn, all_matches[1:], all_matches[0])

def get_bounds_x_tilted(bounds_x: List[float], bounds_y: List[float], phi_tilt: Union[float, core.Arr[float]], shape: str):
    """get the effective bounds of the x-axis of the spec when covering a tilted patch"""
    assert shape in ['ellipse', 'rect']  # more to implement
    a = (bounds_x[1] - bounds_x[0])/2
    b = (bounds_y[1] - bounds_y[0])/2
    if shape == 'ellipse':
        w_proj = a * np.sqrt(1 + b**2 / a**2 * np.tan(phi_tilt)**2)
    elif shape == 'rect':
        w_proj = b * np.tan(phi_tilt) + a
    else:
        raise NotImplementedError
    return np.array([-w_proj, w_proj]) + (bounds_x[0] + bounds_x[1])/2

def make_circular_cover(xi0, eta0, R, count=50, degree=True):
    """make a circular cover centered at xi0, eta0 with radius R"""
    if degree: xi0, eta0, R = np.deg2rad([xi0, eta0, R])
    dphi = 2*np.pi/count
    phi = np.arange(count) * dphi
    L = 1.01*R / np.cos(dphi/2)
    xi, eta = L * np.cos(phi), L * np.sin(phi)
    xi, eta, _ = quat.decompose_xieta(quat.rotation_xieta(xi0, eta0) * quat.rotation_xieta(xi, eta))
    return {
        'center': (xi0, eta0),
        'cover': np.array([xi, eta])
    }

def array_info_merge(arrays):
    center = np.mean(np.array([a['center'] for a in arrays]), axis=0)
    cover = np.concatenate([a['cover'] for a in arrays], axis=1)
    return {
        'center': center,
        'cover': cover
    }

def array_info_from_query(geometries, query):
    """make an array info with geometries that match the query"""
    is_leaf = lambda x: isinstance(x, dict) and 'center' in x
    matched = tu.tree_leaves(tu.tree_map_with_path(
        lambda path, x: x if u.match_query(path, query) else None,
        geometries,
        is_leaf=is_leaf
    ), is_leaf=is_leaf)
    arrays = [make_circular_cover(*g['center'], g['radius']) for g in matched]
    return array_info_merge(arrays)

def parse_sequence_from_toast(ifile):
    """
    Parameters
    ----------
    ifile : str
        Path to the input master schedule from toast.

    Returns
    -------
    list of ScanBlock
        List of ScanBlock objects parsed from the input file.

    """
    #columns = ["start_utc", "stop_utc", "rotation", "patch", "az_min", "az_max", "el", "pass", "sub"]
    #columns = ["start_utc", "stop_utc", "rotation", "az_min", "az_max", "el", "pass", "sub", "patch"]
    #columns = ["start_utc", "stop_utc", "hwp_dir", "rotation", "az_min", "az_max", "el", "pass", "sub", "patch"]
    columns = ["start_utc", "stop_utc", "hwp_dir", "rotation", "az_min", "az_max",
               "el", "speed", "accel", "pass", "sub", "uid", "patch"]

    # count the number of lines to skip
    with open(ifile) as f:
        for i, l in enumerate(f):
            if l.startswith('#'):
                continue
            else:
                break
    df = pd.read_csv(ifile, skiprows=i, delimiter="|", names=columns, comment='#')
    blocks = []
    for _, row in df.iterrows():
        block = ScanBlock(
            name=row['patch'].strip(),
            t0=u.str2datetime(row['start_utc']),
            t1=u.str2datetime(row['stop_utc']),
            alt=row['el'],
            az=row['az_min'],
            az_speed=row['speed'],
            az_accel=row['accel'],
            throw=np.abs(row['az_max'] - row['az_min']),
            boresight_angle=row['rotation'],
            tag=row['uid'].strip(),
            hwp_dir=(row['hwp_dir'] == 1)
        )
        blocks.append(block)
    return blocks
