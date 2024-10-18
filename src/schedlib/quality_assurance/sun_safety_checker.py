#!/usr/bin/env python

import numpy as np
import argparse
import datetime

## going to require the actual agent code here
import socs.agents.acu.avoidance as avoidance

import schedlib.utils as u
logger = u.init_logger(__name__)

## Code to check sun safety

class SunCrawler:
    def __init__(self, platform, path=None, cmd_txt=None):
        assert platform in ['satp1', 'satp2', 'satp3'], (
            f"{platform} is not an implemented platform, choose from satp1, "
             "satp2, or satp3"
        )
    
        match platform:
            case "satp1":
                from schedlib.policies.satp1 import make_config
            case "satp2":
                from schedlib.policies.satp2 import make_config
            case "satp3":
                from schedlib.policies.satp3 import make_config

        self.configs = make_config(
            master_file='None',
            az_speed=None, az_accel=None,
            cal_targets=None,
        )['rules']['sun-avoidance']
            
        if not path is None:
            self.from_cmds = False
            self.schedf = open(path, 'r')
        elif not cmd_txt is None:
            self.from_cmds = True
            self.cmd_n = 0
            self.cmd_list = cmd_txt.split("\n")
        else:
            raise ValueError(
                "SunCrawler needs either a path to a schedule file or "
                "a list of commands."
            )

        self.cur_time = 0
        self.cur_az = 0
        self.cur_el = 0

        self.MAX_SUN_MAP_TDELTA = 6.*3600.

        self._get_initial_pos()
        self._generate_sun_solution()
        #self._test_sungod()
        
    def next_line(self):
        if self.from_cmds:
            if self.cmd_n == len(self.cmd_list):
                return ''
            self.cmd_n += 1
            l = self.cmd_list[self.cmd_n-1]+"\n"
        else:
            l = self.schedf.readline()
        if len(l)>0 and l[0] == "#":
            return self.next_line()
        return l


    def _move_to_parse(self, l):
        try:
            az = float(l.split('az=')[1].split(',')[0])
        except IndexError:
            print('Bad input!', l)
            az = None

        try:
            el = float(l.split('el=')[1].rstrip(')\n'))
        except IndexError:
            try:
                el = float(l.split(',')[1].rstrip(')\n'))
            except IndexError:
                print('Bad input!', l)
                el = None

        return az, el

    def _wait_parse(self, l):
        t = l.split("('")[1].rstrip("')\n").split(",")[0].strip("'")
        time = datetime.datetime.fromisoformat(t).timestamp()
        return time

    def _scan_parse(self, b):

        stop = 0.
        width = 0.
        drift = 0.
        for l in b:
            if 'stop_time' in l:
                stop = datetime.datetime.fromisoformat(l.split("stop_time='")[1].rstrip("', \n")).timestamp()
            if 'width' in l and 'drift' in l:
                width = float(l.split(', ')[0].split('=')[1])
                drift = float(l.split(', ')[1].split('=')[1].rstrip(', \n'))
            elif 'width' in l and 'drift' not in l:
                width = float(l.split('=')[1].rstrip(', \n'))
            elif 'drift' in l and 'width' not in l:
                drift = float(l.split('=')[1].rstrip(', \n'))
                
        init_az_range = [self.cur_az, self.cur_az + width]
        drifted_az = self.cur_az + drift * (stop - self.cur_time)
        end_az_range = [drifted_az, drifted_az + width]

        if self.cur_time - self.sungod.base_time > self.MAX_SUN_MAP_TDELTA:
            print('Resetting sun god!')
            self.sungod.reset(base_time=self.cur_time-100.)
        
        d1 = self.sungod.check_trajectory(init_az_range, [self.cur_el, self.cur_el], t=self.cur_time)
        d2 = self.sungod.check_trajectory(end_az_range, [self.cur_el, self.cur_el], t=stop)
        assert((d1['sun_dist_min'] > self.policy['exclusion_radius']) and (d2['sun_dist_min'] > self.policy['exclusion_radius']))
        assert((d1['sun_time'] > self.policy['min_sun_time']) and (d2['sun_time'] > self.policy['min_sun_time']))

        #print('Min scan distance to sun at start', d1['sun_dist_min'])
        print('Min scan distance to sun at end', d2['sun_dist_min'])


        logger.debug(f"azimuth : {self.cur_az} --> {drifted_az + width}")
        logger.debug(f"timestamp : {self.cur_time} --> {stop}")
        
        self.cur_az = drifted_az + width
        # Need to determine which az will be most stringent for next slew
        if d2['sun_dist_start'] >= d2['sun_dist_stop']:
            logger.debug(f"azimuth : {self.cur_az} --> {drifted_az + width}")
            self.cur_az = drifted_az + width
        else:
            logger.debug(f"azimuth : {self.cur_az} --> {drifted_az}")
            self.cur_az = drifted_az

        logger.debug(f"timestamp : {self.cur_time} --> {stop}")

        self.cur_time = stop

    def _get_initial_pos(self):
        time_flag = False
        pos_flag = False
        while True:
            l = self.next_line()
            if 'move_to' in l:
                az, el = self._move_to_parse(l)
                self.cur_az = az
                self.cur_el = el
                pos_flag = True
                
            if 'wait_until' in l:
                ts = self._wait_parse(l)
                self.cur_time = ts
                time_flag = True

            if pos_flag and time_flag:
                break

    def _generate_sun_solution(self):
        self.policy = avoidance.DEFAULT_POLICY
        self.policy['min_el'] = 48.
        self.policy['min_sun_time'] = self.configs['min_sun_time']
        self.policy['exclusion_radius'] = self.configs['min_angle']

        self.sungod = avoidance.SunTracker(policy=self.policy, base_time=self.cur_time, compute=True)

    def _test_sungod(self):
        out = self.sungod.get_sun_pos(t=self.cur_time+500.)
        print(out)

    def _get_traj(self, az, el, wrap_north=False):
        if az == 0:
            az = self.next_az

        # catching wrap questions
        if self.cur_az < 180 and az >= 180: # ccw wrap
            if not wrap_north:
                mid_az = np.mean([self.cur_az, az])
            else:
                az += -360.
                mid_az = np.mean([self.cur_az, az])
                print('Special az wrap to north', mid_az, az)
        elif self.cur_az > 180 and az <= 180: # cw wrap
            mid_az = np.mean([self.cur_az, az])
        else:
            mid_az = az

        mid_el = np.mean([self.cur_el, el])

        return [self.cur_az, mid_az, az], [self.cur_el, mid_el, el]

    def step_thru_schedule(self):
        logger.info("Checking Sun Safety")
        cur_block = []
        scan_flag = False
        
        while True:
            l = self.next_line()

            if 'wait_until' in l:
                ts = self._wait_parse(l)
                logger.debug(f"timestamp: '{self.cur_time} --> {ts}")
                self.cur_time = ts
            
            if 'move_to' in l:
                az, el = self._move_to_parse(l)

                if az is not None and el is not None:
                    az_range, el_range = self._get_traj(az, el)
                    logger.debug(f"azimuth : {self.cur_az} --> {az}")
                    logger.debug(f"elevation : {self.cur_el} --> {el}")
                    
                    self.cur_az = az
                    self.cur_el = el
                elif el is not None:
                    az_range, el_range = self._get_traj(0, el)


                    logger.debug(
                        f"azimuth :{self.cur_az} --> {self.next_az}"
                    )
                    logger.debug(f"elevation :' {self.cur_el} --> {el}")

                    # if np.round(self.cur_az,2) == np.round(self.next_az,2):
                    #     self.cur_el = el
                    #     continue
                    
                    self.cur_az = self.next_az
                    self.cur_el = el
                    
                #print('Ranges', az_range, el_range)
                if self.cur_time - self.sungod.base_time > self.MAX_SUN_MAP_TDELTA:
                    logger.info('Resetting sun god!')
                    self.sungod.reset(base_time=self.cur_time-100.)
                
                d = self.sungod.check_trajectory(az_range, el_range, t=self.cur_time)

                logger.info(f"Min slew distance to sun {d['sun_dist_min']}")
                try:
                    assert(d['sun_dist_min'] > self.policy['exclusion_radius'])
                except AssertionError as e:
                    self.raise_failure(e, l)
                logger.info(f"Min sun clear time {d['sun_time']}")
                try:
                    assert(d['sun_time'] > self.policy['min_sun_time'])
                except AssertionError as e:
                    self.raise_failure(e, l)

                moves = self.sungod.analyze_paths(az_range[0], el_range[0], az_range[-1], el_range[-1], t=self.cur_time)
                move, decisions = self.sungod.select_move(moves)
                try:
                    assert(move is not None)
                except AssertionError as e:
                    self.raise_failure(e, l, moves)

            if 'az = ' in l:
                az = float(l.split('az = ')[1].split('+')[0])
                try:
                    self.next_az = az
                except AttributeError:
                    self.next_az = 0.
                    self.next_az = az

            if 'run.seq.scan' in l:
                assert(l[-2:] == '(\n')
                scan_flag = True

            if l.strip() == ')':
                scan_flag = False
                self._scan_parse(cur_block)
                cur_block = []

            if scan_flag:
                cur_block.append(l)

            if l == '':
                break

    def raise_failure(self, e, line, moves=None):
        out = self.sungod.get_sun_pos(t=self.cur_time)
        logger.info(f'Sun position at failure time {out}')
        logger.error('Sun-safe motions not solved!')
        t = datetime.datetime.utcfromtimestamp(self.cur_time)
        l = line.strip('\n')
        logger.error(
            f"Error on Line \'{l}\' at time {t.isoformat()}"
        )
        if moves is not None:
            logger.error(
                'Move info (min sun dist, min sun time, min el, max el):'
            )
            logger.error('\n'.join([', '.join(map(str, [m['sun_dist_min'], m['sun_time'], min(m['moves'].get_traj(res=1.0)[1]), max(m['moves'].get_traj(res=1.0)[1])])) for m in moves]))
        raise(e)            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('platform')
    parser.add_argument('sched_path')

    args = parser.parse_args()
    sc = SunCrawler(args.platform, args.sched_path)
    sc.step_thru_schedule()
