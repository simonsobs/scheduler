Blocks 
========

This is the core of the scheduler setup. Many different types

core.Block
-----------
A generic block of time that you can do math on

* t0
* t1

core.NamedBlock(Block)
-----------------------
Adds a name

* name 

instrument.ScanBlock(core.NamedBlock)
---------------------------------------
Represents scanning for the telescopes 

* az: float        # deg
* alt: float       # deg
* throw: float     # deg
* az_drift: float = 0. # deg / s
* az_speed: float = 1. # deg / s
* az_accel: float = 2. # deg / s**2
* boresight_angle: Optional[float] = None # deg
* hwp_dir: Optional[bool] = None
* subtype: str = ""
* tag: str = ""
* priority: float = 0

source.SourceBlock(core.NamedBlock)
-------------------------------------
Represents a celestial source rising or setting

* mode # 'rising' or 'setting'
* t, az, alt # properties that are computed the first time they're needed
