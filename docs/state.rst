State 
=======

The state class is what we are using to keep track of the state of the 
telescopes


commands.State
---------------
This top level class tracks time and the positions of azimuth and elevation

* curr_time
* az_now
* el_now
* az_speed_now
* az_accel_now
* prev_state


tel.State( commands.State )
----------------------------
Telescope state adds in the variables that are relevant for detector setup. tel.
State will depend on a `get_boresight()` function that must be implemented by the SAT and LAT states because boresight is a derived parameter for the LAT.

* last_ufm_relock
* last_bias_step
* last_bias_step_boresight
* last_bias_step_elevation
* last_iv
* last_iv_boresight
* last_iv_elevation
* is_det_setup
* get_boresight() 

sat.State( tel.State ) 
-----------------------
The sat state adds in HWP 

* hwp_spinning
* hwp_dir
* boresight_rot_now

lat.State( tel.State )
-----------------------
The LAT does not have a boresight axis but it does have a co-rotator that can 
do weird boresight rotations. boresight = elevation - 60 - corotator. 

* corotator_now