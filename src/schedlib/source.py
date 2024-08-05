# -*- coding: utf-8 -*-

from __future__ import annotations
import logging
from dataclasses import dataclass
import ephem
from ephem import to_timezone
import datetime as dt
from typing import Union, Callable, NamedTuple, List, Tuple, Optional
import numpy as np
from scipy import interpolate, optimize
from so3g.proj import quat, CelestialSightLine

from . import core, utils as u, instrument as inst

UTC = dt.timezone.utc


class Location(NamedTuple):
    """Location given in degrees and meters"""
    lat: float
    lon: float
    elev: float

    def at(self, date: dt.datetime) -> ephem.Observer:
        """Always get new object to avoid side effects."""
        obs = ephem.Observer()
        obs.lat = str(self.lat)
        obs.lon = str(self.lon)
        obs.elev = self.elev
        obs.date = ephem.date(date)
        return obs


def _debabyl(deg, arcmin, arcsec):
    return deg + arcmin/60 + arcsec/3600

SITES = {
    'act':   Location(lat=-22.9585, lon=-67.7876, elev=5188),
    'lat':   Location(lat=-_debabyl(22, 57, 39.47), lon=-_debabyl(67, 47, 15.68), elev=5188),
    'satp1': Location(lat=-_debabyl(22, 57, 36.38), lon=-_debabyl(67, 47, 18.11), elev=5188),
    'satp2': Location(lat=-_debabyl(22, 57, 36.35), lon=-_debabyl(67, 47, 17.28), elev=5188),
    'satp3': Location(lat=-_debabyl(22, 57, 35.97), lon=-_debabyl(67, 47, 16.53), elev=5188),
}
DEFAULT_SITE = Location(lat=-22.958, lon=-67.786, elev=5200)

def get_site(site='lat') -> Location:
    """use lat as default following so3g convention"""
    return SITES[site]

# source needs to be callable to avoid side effects
SOURCES = {
    'sun': ephem.Sun,
    'moon': ephem.Moon,
    'mercury': ephem.Mercury,
    'venus': ephem.Venus,
    'mars': ephem.Mars,
    'jupiter': ephem.Jupiter,
    'saturn': ephem.Saturn,
    'uranus': ephem.Uranus,
    'neptune': ephem.Neptune,
}

Source = Union[ephem.Body, ephem.FixedBody]

def get_source(name: str) -> Source:
    # always get new object to avoid side effects
    if name.lower() == 'taua':
        obj = ephem.FixedBody()
        obj._ra = 5.5755*np.pi/12.
        obj._dec = 22.0167*np.pi/180.
        return obj
    if name.lower() == 'galcenter':
        obj = ephem.FixedBody()
        obj._ra = 17.7611*np.pi/12.
        obj._dec = -28.95*np.pi/180.
        return obj
    return SOURCES[name]()

def _source_get_az_alt(source: str, times: List[dt.datetime]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the azimuth and altitude of a source at given times.

    Parameters
    ----------
    source : str
        The name of the celestial source.
    times : List[datetime.datetime]
        A list of timezone-aware datetime objects for which to calculate the source's azimuth and altitude.

    Returns
    -------
    az, alt: np.ndarray, np.ndarray

    """
    assert len(times) > 0, "Need at least one time"
    if times[0].tzinfo is None:
        raise ValueError("Need timezone-aware datetime")
    observer = get_site().at(times[0])
    source = get_source(source)
    az, alt = [], []
    for t in times:
        observer.date = ephem.date(t)
        source.compute(observer)
        az.append(np.rad2deg(source.az))
        alt.append(np.rad2deg(source.alt))
    az, alt = np.array(az), np.array(alt)
    az = np.unwrap(az, period=360)
    # prefer close to 0
    az = (az + 180) % 360 - 180
    az = (az - az.min()) % 360 + az.min()
    return az, alt


def _source_az_alt_interpolators(
    source: str,
    t0: dt.datetime,
    t1: dt.datetime,
    time_step: dt.timedelta = dt.timedelta(seconds=30)
) -> Tuple[Callable, Callable]:
    """
    Create azimuth and altitude interpolators for a source between two time points.

    Parameters
    ----------
    source : str
        The name of the source.
    t0 : datetime.datetime
        The starting time for the interpolator.
    t1 : datetime.datetime
        The ending time for the interpolator.
    time_step : datetime.timedelta
        The time step between each interpolation point. Default is 30 seconds.

    Returns
    -------
    interp_az : Callable
        An interpolator function for the azimuth of the source over the given time range.
    interp_alt : Callable
        An interpolator function for the altitude of the source over the given time range.

    """
    times = [t0 + i * time_step for i in range(int((t1 - t0) / time_step))]
    az, alt = _source_get_az_alt(source, times)
    times = [u.dt2ct(t) for t in times]
    interp_az = interpolate.interp1d(times, az, kind='cubic', fill_value='extrapolate')
    interp_alt = interpolate.interp1d(times, alt, kind='cubic', fill_value='extrapolate')
    return interp_az, interp_alt

# global registry of precomputed sources
PRECOMPUTED_SOURCES = {}

class _PrecomputedSource(NamedTuple):
    """
    A object to store precomputed source information for efficient retrieval.

    Attributes
    ----------
    t0 : datetime.datetime
        The starting time for the precomputed source data.
    t1 : datetime.datetime
        The ending time for the precomputed source data.
    interp_az : Callable[[int], float]
        An interpolator function for the azimuth of the source over the given time range.
    interp_alt : Callable[[int], float]
        An interpolator function for the altitude of the source over the given time range.
    blocks : core.Blocks
        SourceBlocks associated with the source.

    Methods
    -------
    for_(name: str, t0: datetime.datetime, t1: datetime.datetime,
         buf: datetime.timedelta = datetime.timedelta(days=1),
         time_step: datetime.timedelta = datetime.timedelta(seconds=30)) -> Source
        Class method to retrieve or compute the precomputed source data.
    for_block(block: SourceBlock, buf: datetime.timedelta = datetime.timedelta(days=1),
              time_step: datetime.timedelta = datetime.timedelta(seconds=30)) -> Source
        Class method to retrieve or compute the precomputed source data for a given block.

    """
    t0: dt.datetime
    t1: dt.datetime
    interp_az: Callable[[int], float]
    interp_alt: Callable[[int], float]
    blocks: core.Blocks

    @classmethod
    def for_(
        cls, name: str, t0: dt.datetime, t1: dt.datetime,
        buf: dt.timedelta = dt.timedelta(days=1),
        time_step: dt.timedelta = dt.timedelta(seconds=30)
    ) -> _PrecomputedSource:
        """
        Retrieve or compute the precomputed source data.

        Parameters
        ----------
        name : str
            The name of the celestial source.
        t0 : datetime.datetime
            The starting time for the interpolation.
        t1 : datetime.datetime
            The ending time for the interpolation.
        buf : datetime.timedelta, optional
            The buffer time to add to the ending time, by default 1 day.
        time_step : datetime.timedelta, optional
            The time step between each interpolation point, by default 30 seconds.

        Returns
        -------
        Source
            The precomputed source
        """
        reuse = False
        if name in PRECOMPUTED_SOURCES:
            precomputed = PRECOMPUTED_SOURCES[name]
            reuse = precomputed.t0 <= t0 and precomputed.t1 >= t1
        if not reuse:
            # future is more important than past
            t0, t1 = t0, t1 + buf
            az_interp, alt_interp = _source_az_alt_interpolators(name, t0, t1, time_step)
            # precompute blocks of observation time
            blocks = source_get_blocks(name, t0, t1)
            PRECOMPUTED_SOURCES[name] = cls(t0, t1, az_interp, alt_interp, blocks)
        return PRECOMPUTED_SOURCES[name]

    @classmethod
    def for_block(
        cls,
        block: SourceBlock,
        buf: dt.timedelta = dt.timedelta(days=1),
        time_step: dt.timedelta = dt.timedelta(seconds=30)
    ) -> _PrecomputedSource:
        """
        Retrieve or compute the precomputed source for a given block.

        Parameters
        ----------
        block : SourceBlock
            The block of observation data for which to retrieve or compute the source data.
        buf : datetime.timedelta, optional
            The buffer time to add to the ending time of the block, by default 1 day.
        time_step : datetime.timedelta, optional
            The time step between each interpolation point, by default 30 seconds.

        Returns
        -------
        _PrecomputedSource

        """
        return cls.for_(block.name, block.t0, block.t1, buf=buf, time_step=time_step)

@dataclass(frozen=True)
class SourceBlock(core.NamedBlock):
    mode: str

    def __post_init__(self):
        if not self.mode in ["rising", "setting", "both"]:
            raise ValueError("mode must be rising or setting or both")

    def get_az_alt(self, time_step=30, ctimes=None):
        """Return times, az, alt for a source block at a given time step"""
        source = _PrecomputedSource.for_block(self, time_step=dt.timedelta(seconds=time_step))

        # if ctimes is not provided, we will calculate it
        # based on the time step
        if ctimes is None:
            t0, t1 = u.dt2ct(self.t0), u.dt2ct(self.t1)
            ctimes = np.arange(t0, t1+time_step, time_step)

        az = source.interp_az(ctimes)
        alt = source.interp_alt(ctimes)
        az = np.unwrap(az, period=360)
        # prefer close to 0
        # az_min = (az_min + 180) % 360 - 180
        az = (az + 180) % 360 - 180
        az = (az - az.min()) % 360 + az.min()
        return ctimes, az, alt

    def trim_by_az_alt_range(self, az_range: Optional[Tuple[float, float]] = None,
                             alt_range: Optional[Tuple[float, float]] = None,
                             time_step: float = 30):
        """
        Trim a source block by azimuth and altitude ranges

        Parameters
        ----------
        alt_range: (alt_min, alt_max) in degrees
        z_range: (az_min, az_max) in degrees

        """
        if az_range is None and alt_range is None:
            # not sure why one would want to do this though
            return [self]
        times, az, alt = self.get_az_alt(time_step=time_step)
        mask = np.ones_like(az, dtype=bool)
        if az_range is not None:
            az_min, az_max = az_range
            mask *= (az_min <= az) * (az <= az_max)
        if alt_range is not None:
            alt_min, alt_max = alt_range
            mask *= (alt_min <= alt) * (alt <= alt_max)
        if not mask.any():
            return []  # need blocks type
        blocks = []
        for (i0, i1) in u.mask2ranges(mask):
            t0 = u.ct2dt(times[i0])
            t1 = u.ct2dt(times[i1-1])  # i1 is non-inclusive
            blocks.append(self.replace(t0=t0, t1=t1))
        return blocks

    @property
    def t(self):
        return self.get_az_alt()[0]

    @property
    def az(self):
        return self.get_az_alt()[1]

    @property
    def alt(self):
        return self.get_az_alt()[2]

    def get_az_alt_interpolators(self):
        source = _PrecomputedSource.for_block(self)
        return source.interp_az, source.interp_alt

def source_get_blocks(name: str, t0: dt.datetime, t1: dt.datetime) -> core.Blocks:
    """
    Generate a list of SourceBlocks for the given source name during the time interval
    defined by t0 and t1, each SourceBlock representing a period of time when the
    source is either rising or setting.

    Note that this function does actual computation. In practice, it is almost always
    preferable to use `source_gen_seq` instead, which will try to load from cache first.

    Parameters
    ----------
    name : str
        The name of the source to generate blocks for.
    t0 : dt.datetime
        The start of the time interval for generating blocks.
    t1 : dt.datetime
        The end of the time interval for generating blocks.

    Returns
    -------
    list[SourceBlocks]
        A list of SourceBlocks representing the rising and setting periods of the
        given source during the time interval defined by t0 and t1.

    """
    site = get_site()
    source = get_source(name)
    t_block_beg = to_timezone(site.at(t0).previous_rising(source), UTC)
    t_block_mid = to_timezone(site.at(t_block_beg).next_transit(source), UTC)
    t_block_end = to_timezone(site.at(t_block_mid).next_setting(source), UTC)
    blocks = [SourceBlock(t0=t_block_beg, t1=t_block_mid, name=name, mode="rising"),
              SourceBlock(t0=t_block_mid, t1=t_block_end, name=name, mode="setting")]
    while t_block_end < t1:
        t_block_beg = to_timezone(site.at(t_block_end).next_rising(source), UTC)
        t_block_mid = to_timezone(site.at(t_block_beg).next_transit(source), UTC)
        t_block_end = to_timezone(site.at(t_block_mid).next_setting(source), UTC)
        blocks += [SourceBlock(t0=t_block_beg, t1=t_block_mid, name=name, mode="rising"),
                   SourceBlock(t0=t_block_mid, t1=t_block_end, name=name, mode="setting")]
    return blocks

def source_gen_seq(source: str, t0: dt.datetime, t1: dt.datetime) -> core.Blocks:
    """
    Generate a sequence of blocks from the given source for the time range (t0, t1].

    This function is similar to `source_get_blocks` but it will try loading from cache
    first and will return a trimmed sequence of blocks. It is the recommended way to
    generate a sequence of source blocks. TODO: better name!

    Parameters
    ----------
    source : str
        The identifier of the data source.
    t0 : dt.datetime
        The start time of the time range.
    t1 : dt.datetime
        The end time of the time range.

    Returns
    -------
    list[SourceBlocks]
        A list of SourceBlocks representing the rising and setting periods of the
        given source during the time interval defined by t0 and t1.

    """
    blocks = _PrecomputedSource.for_(source, t0, t1).blocks
    return core.seq_flatten(core.seq_trim(blocks, t0, t1))

def block_get_matching_sun_block(block: core.Block) -> SourceBlock:
    """
    Get the corresponding sun block with the same time bounds. It is primarily
    used for sun avoidance calculation.

    Parameters
    ----------
    block : Block
        The input block for which a corresponding sun block will be returned.

    Returns
    -------
    SourceBlock
        A SourceBlock object with the same time bounds as the input block and
        with the name and mode set to 'sun' and 'both', respectively.

    """
    return SourceBlock(t0=block.t0, t1=block.t1, name="sun", mode="both")

@dataclass(frozen=True)
class ObservingWindow(SourceBlock):
    """
    An old convenience class for storing some relevant data to make interpolators
    of az, alt, and throw for a given source. It is not used in the current
    policy implementation, will be removed in the future.

    Attributes:
    -----------
    t_start : core.Arr[float]
        Start time of the observing window.
    obs_length : core.Arr[float]
        Length of the observing window.
    az_bore : core.Arr[float]
        Azimuth value of the boresight at t_start.
    alt_bore : core.Arr[float]
        Altitude value of the boresight at t_start.
    az_throw : core.Arr[float]
        Azimuth value of the throw.
    """
    t_start: core.Arr[float]
    obs_length: core.Arr[float]
    az_bore: core.Arr[float]
    alt_bore: core.Arr[float]
    az_throw: core.Arr[float]

    def get_scan_at_t0(self, t0: dt.datetime) -> inst.ScanBlock:
        """
        Get a possible scan starting at a given time.

        Parameters
        ----------
        t0 : datetime.datetime
            The starting time of the scan.

        Returns
        -------
        scan_block : inst.ScanBlock
            A ScanBlock object with attributes t0, t1, az, alt, and throw
            calculated based on the given t0.
        """
        t_req = int(t0.timestamp())
        # if we start at t0, we can observe for at most obs_length
        obs_length = u.interp_bounded(t_req, self.t_start, self.obs_length)
        t1 = t0 + dt.timedelta(seconds=float(obs_length))
        # if we start at t0, we can observe with these parameters
        az = u.interp_bounded(t_req, self.t_start, self.az_bore)
        alt = u.interp_bounded(t_req, self.t_start, self.alt_bore)
        az_throw = u.interp_bounded(t_req, self.t_start, self.az_throw)
        return inst.ScanBlock(
            name=self.name,
            t0=t0,
            t1=t1,
            az=float(az),
            alt=float(alt),
            throw=float(az_throw),
        )
    def get_scan_at_alt(self, alt: float) -> inst.ScanBlock:
        """
        Get a possible scan at a given altitude.

        Parameters
        ----------
        alt : float
            The altitude at which to get the scan.

        Returns
        -------
        inst.ScanBlock
            A scan block at the given altitude.

        """
        t0 = u.interp_bounded(alt, self.alt_bore, self.t_start)
        return self.get_scan_at_t0(t0)

def _find_az_bore(el_bore, az_src, el_src, q_point, atol: float = 0.01) -> float:
    """
    Find the boresight azimuthal angle (az_bore) at a given boresight elevation (el_bore).

    Parameters
    ----------
    el_bore : float
        Elevation angle of the boresight
    az_src : numpy.ndarray
        Azimuth angles of the source
    el_src : numpy.ndarray
        Elevation angles of the source
    q_point : Quaternion
        Quaternion of the focal plane center
    atol : float
        Absolute tolerance for the optimization

    Returns
    -------
    az_bore : float
        The boresight angle

    """
    az_interp = interpolate.interp1d(el_src, az_src, kind='linear',
                                     fill_value='extrapolate',
                                     assume_sorted=False)
    def fun(az_bore):
        az_center, el_center, _ =  quat.decompose_lonlat(
            quat.rotation_lonlat(-az_bore * u.deg, el_bore * u.deg) * q_point
        )
        az_center *= -1
        az_expect = az_interp(el_center / u.deg)
        diff = (az_expect - az_center/u.deg)
        diff = (diff + 180) % 360 - 180
        return np.abs(diff)

    az_bore_init = az_interp(el_bore)
    res = optimize.minimize(fun, az_bore_init, method='Nelder-Mead')
    az_bore = res.x[0]

    # extra check
    if fun(az_bore) > atol:
        raise ValueError(f"failed to meet convergence tol ({atol}) on where to point the boresight")
    return az_bore

def make_source_ces(
    block, array_info,
    el_bore=50, allow_partial=False,
    v_az=None, boresight_rot=None,
    az_branch=None,
):
    """make a ces scan of a source

    Parameters
    ----------
    block: SourceBlock
        a source block to compute the ces scan for
    array_info: dict
        contains center and cover of the array
    el_bore: float
        elevation of the boresight in degrees
    allow_partial: bool
        if True, allow partial coverage of the array
    v_az: Optional[float]
        az drift speed in az in deg/s, if None, will try to find optimal drift speed
    boresight_rot: Optional[float]
        rotation of the boresight in deg
    az_branch: Optional[flat]
        az value at the center of the preferred observation branch (e.g. 180 if you
        prefer scans in the (0, 360) range).

    Returns
    -------
    inst.ScanBlock
        a scan block that can be used to scan the source

    """
    assert 'center' in array_info and 'cover' in array_info, 'array_info must contain center and cover'
    assert block.mode in ['rising', 'setting'], 'unsupported scan mode'

    q_center = quat.rotation_xieta(*array_info['center'])
    q_cover = quat.rotation_xieta(*array_info['cover'])

    # apply boresight rotation if specified
    if boresight_rot is not None:
        q_bore_rot = quat.euler(2, -np.deg2rad(boresight_rot))
        q_center = q_bore_rot * q_center
        q_cover = q_bore_rot * q_cover

    t, az_src, el_src = block.get_az_alt()  # degs
    # make sure we are in a monotonic range
    if block.mode == 'rising':
        i_beg = np.argmin(el_src)
        i_end = np.argmax(el_src) + 1
    else:
        i_beg = np.argmax(el_src)
        i_end = np.argmin(el_src) + 1
    t, az_src, el_src = t[i_beg:i_end], az_src[i_beg:i_end], el_src[i_beg:i_end]

    t_src_interp = interpolate.interp1d(
        el_src, t, kind='cubic', fill_value='extrapolate', assume_sorted=False
    )

    # work out boresight
    az_bore = _find_az_bore(el_bore, az_src, el_src, q_center)
    q_bore = quat.rotation_lonlat(-az_bore * u.deg, el_bore * u.deg)

    # put array on the sky
    az_cover, el_cover, _ = quat.decompose_lonlat(q_bore * q_cover)
    az_cover *= -1

    # can we cover the full array?
    if not allow_partial:
        if np.max(el_cover) / u.deg > np.max(el_src):
            logging.warning("Source will not cover the top part of the array")
            return None
        if np.min(el_cover) / u.deg < np.min(el_src):
            logging.warning("Source will not cover the bottom part of the array")
            return None

    if block.mode == 'rising':
        t0 = t_src_interp(max(np.min(el_cover) / u.deg, np.min(el_src)))
        t1 = t_src_interp(min(np.max(el_cover) / u.deg, np.max(el_src)))
    elif block.mode == 'setting':
        t0 = t_src_interp(min(np.max(el_cover) / u.deg, np.max(el_src)))
        t1 = t_src_interp(max(np.min(el_cover) / u.deg, np.min(el_src)))
    else:
        raise ValueError(f'unsupported scan mode encountered: {block.mode}')

    # now we have the time bounds, we will find approximate optimal drift
    def _find_approx_az_throw(az_drift, az_src, el_src):
        az_src = az_src - az_drift * (t - t0)
        az_src_interp = interpolate.interp1d(el_src, az_src, kind='cubic')
        # find the az distance to the source for each point on the array cover
        distances = []
        for az_, el_ in zip(az_cover / u.deg, el_cover / u.deg):
            # we'll only be here if allow_partial is True, in which case
            # we safely ignore these points
            if el_ > np.max(el_src) or el_ < np.min(el_src):
                continue
            distance = az_src_interp(el_) - az_
            distances.append(distance)
        distances = np.array(distances)
        if len(distances) == 0:
            logging.error("Source will not cover the array at all")
            raise ValueError("Source will not cover the array at all")
        az0, az1 = np.array([np.min(distances), np.max(distances)]) + az_bore
        throw = az1 - az0
        return az0, throw

    def _set_branch(az0):
        if az_branch is None:
            return az0
        return (az0 - (az_branch - 180)) % 360 + (az_branch - 180)

    # only solve if no az_drift are specified
    if v_az is None:
        try:
            res = optimize.minimize(
                lambda x, *args: _find_approx_az_throw(x, *args)[1],
                0,
                args=(az_src, el_src),
                method='Nelder-Mead'
            )
            if not res.success:
                raise ValueError(
                    "Failed to find optimal drift, using median az speed instead")
            else:
                v_az = res.x[0]

        except ValueError:
            logging.error("Failed to find optimal drift, using median az speed instead")
            v_az = np.median(np.diff(az_src) / np.diff(t))
    try:
        az0, throw = _find_approx_az_throw(v_az, az_src, el_src)
        az0 = _set_branch(az0)
        return inst.ScanBlock(
            name=block.name,
            az=az0,
            alt=el_bore,
            throw=throw,
            t0=u.ct2dt(float(t0)),
            t1=u.ct2dt(float(t1)),
            az_drift=v_az,
            boresight_angle=boresight_rot,
            tag=f"{block.name},{block.mode}"
        )
    except ValueError:
        logging.error("Failed to find optimal drift, using median az speed instead")
        return None

def radial_distance(t, az1, alt1, az2, alt2):
    """
    Calculate the radial distance between two celestial sight lines.

    Parameters
    ----------
    t : np.ndarray
        The time at which the sight lines are taken.
    az1, alt1 : np.ndarray
        The azimuth and altitude of the first object, in degrees.
    az2, alt2 : np.ndarray
        The azimuth and altitude of the second object, in degrees.

    Returns
    -------
    r : np.ndarray
        The radial distance between the two sight lines, in degrees.

    """
    csl1 = CelestialSightLine.az_el(t, az1*u.deg, alt1*u.deg, 0, weather='vacuum')
    csl2 = CelestialSightLine.az_el(t, az2*u.deg, alt2*u.deg, 0, weather='vacuum')
    r = quat.decompose_iso(~csl1.Q * csl2.Q)[0] / u.deg
    return r
