import numpy as np 
import casadi as ca 
from matplotlib import pyplot as plt
from numba import njit
 
import LibFunctions as lib
from Simulator import ScanSimulator


class OptimalAgent:
    def __init__(self, config):
        self.name = "Optimal Agent: Following target references"
        self.env_map = None
        self.path_name = None
        self.wpts = None
        self.vs = None

        self.pind = 1
        self.target = None
        self.steps = 0

        mu = config['car']['mu']
        self.m = config['car']['m']
        g = config['car']['g']
        safety_f = config['pp']['force_f']
        self.f_max = mu * self.m * g #* safety_f

        self.wpts = None
        self.vs = None

        self.lookahead = config['pp']['lookahead']
        self.v_gain = config['pp']['v_gain']
        self.wheelbase =  config['car']['l_f'] + config['car']['l_r']

        self.current_v_ref = None
        self.current_phi_ref = None

        # self.scan_sim = ScanSimulator(20)

    def init_agent(self, env_map):
        self.env_map = env_map

        self.path_name = "DataRecords/" + self.env_map.name + "_path.npy" # move to setup call
 
        self.reset_lap()
        
        # self.wpts = self.env_map.get_reference_path()

        r_line = self.wpts
        ths = [lib.get_bearing(r_line[i], r_line[i+1]) for i in range(len(r_line)-1)]
        alphas = [lib.sub_angles_complex(ths[i+1], ths[i]) for i in range(len(ths)-1)]
        lds = [lib.get_distance(r_line[i], r_line[i+1]) for i in range(1, len(r_line)-1)]

        self.deltas = np.arctan(2*0.33*np.sin(alphas)/lds)

        self.pind = 1

        return self.wpts

    def act(self, obs):

        v_ref, d_ref = self.get_target_references(obs)


        return [v_ref, d_ref]

    def get_corridor_references(self, obs):
        ranges = obs[5:]
        max_range = np.argmax(ranges)
        dth = np.pi / 9
        theta_dot = dth * max_range - np.pi/2

        L = 0.33
        delta_ref = np.arctan(theta_dot * L / (obs[3]+0.001))

        v_ref = 6

        return v_ref, delta_ref

    def get_target_references(self, obs):
        self._set_target(obs)

        target = self.wpts[self.pind]
        th_target = lib.get_bearing(obs[0:2], target)
        alpha = lib.sub_angles_complex(th_target, obs[2])

        # pure pursuit
        ld = lib.get_distance(obs[0:2], target)
        delta_ref = np.arctan(2*0.33*np.sin(alpha)/ld)

        # v_ref = float(self.vs[self.pind]) * self.v_gain * 0.8
        # delta_ref = self.limit_inputs(v_ref, delta_ref)

        ds = self.deltas[min(self.pind, len(self.deltas)-1)]
        max_d = abs(ds)

        max_friction_force = 3.74 * 9.81 * 0.523 *0.6
        d_plan = max(abs(delta_ref), abs(obs[4]), max_d)
        theta_dot = abs(obs[3] / 0.33 * np.tan(d_plan))
        v_ref = max_friction_force / (3.74 * max(theta_dot, 0.01)) 
        v_ref = min(v_ref, 8.5)
        # v_ref = 3

        return v_ref, delta_ref

    def control_system(self, obs):
        v_ref = self.current_v_ref
        d_ref = self.current_phi_ref

        kp_a = 10
        a = (v_ref - obs[3]) * kp_a
        
        kp_delta = 40
        d_dot = (d_ref - obs[4]) * kp_delta

        return a, d_dot

    def _set_target(self, obs):
        dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
        shift_distance = 1
        while dis_cur_target < shift_distance: # how close to say you were there
            if self.pind < len(self.wpts)-2:
                self.pind += 1
                dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
            else:
                self.pind = 0

    def reset_lap(self):
        self.wpts, self.vs = self.env_map.get_optimal_path()
        self.pind = 0

    def show_vehicle_history(self):
        pass

    def limit_inputs(self, speed, steering_angle):
        max_steer = np.arctan(self.f_max * self.wheelbase / (speed**2 * self.m))
        new_steer = np.clip(steering_angle, -max_steer, max_steer)

        if max_steer < abs(steering_angle):
            print(f"Problem, Steering clipped from: {steering_angle} --> {max_steer}")

        return new_steer


class TunerCar:
    def __init__(self, config) -> None:
        self.config = config
        self.name = "TunerCar Agent: Following PP references"
        self.env_map = None
        self.path_name = None

        mu = config['car']['mu']
        self.m = config['car']['m']
        g = config['car']['g']
        safety_f = config['pp']['force_f']
        self.f_max = mu * self.m * g #* safety_f

        self.wpts = None
        self.vs = None

        self.lookahead = config['pp']['lookahead']
        self.vgain = config['pp']['v_gain']
        self.wheelbase =  config['car']['l_f'] + config['car']['l_r']

    def init_agent(self, env_map):
        self.env_map = env_map

        self.reset_lap()

    def _get_current_waypoint(self, position, theta):
        # nearest_pt, nearest_dist, t, i = nearest_point_on_trajectory_py2(position, self.wpts)
        nearest_pt, nearest_dist, t, i = self.nearest_pt(position)

        if nearest_dist < self.lookahead:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, self.lookahead, self.wpts, i+t, wrap=True)
            if i2 == None:
                return None
            i = i2
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = self.wpts[i2]
            # speed
            current_waypoint[2] = self.vs[i]
            return current_waypoint
        elif nearest_dist < 20:
            return np.append(self.wpts[i], self.vs[i])

    def act(self, obs):
        pose_th = obs[2]
        pos = np.array(obs[0:2], dtype=np.float)

        lookahead_point = self._get_current_waypoint(pos, pose_th)

        if lookahead_point is None:
            return 4.0, 0.0

        self.env_map.targets.append(lookahead_point[0:2])

        speed, steering_angle = self.get_actuation(pose_th, lookahead_point, pos)
        speed = self.vgain * speed

        # print(f"Speed: {speed} --> Steer: {steering_angle}")
        # avg_speed = max(speed, obs[3])
        # steering_angle = self.limit_inputs(avg_speed, steering_angle)

        return [speed, steering_angle]

    def limit_inputs(self, speed, steering_angle):
        max_steer = np.arctan(self.f_max * self.wheelbase / (speed**2 * self.m))
        new_steer = np.clip(steering_angle, -max_steer, max_steer)

        # if max_steer < abs(steering_angle):
        #     print(f"Problem, Steering clipped from: {steering_angle} --> {max_steer}")

        return new_steer

    def reset_lap(self):
        # self.wpts, self.vs = self.env_map.get_optimal_path()
        self.wpts, self.vs = self.env_map.get_reference_path()

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

    def get_actuation(self, pose_theta, lookahead_point, position):
        waypoint_y = np.dot(np.array([np.cos(pose_theta), np.sin(-pose_theta)]), lookahead_point[0:2]-position)
        
        # print(f"Wpt_y: {waypoint_y} --> pos: {position} --> lookahead: {lookahead_point}")

        speed = lookahead_point[2]
        if np.abs(waypoint_y) < 1e-6:
            return speed, 0.
        radius = 1/(2.0*waypoint_y/self.lookahead**2)
        steering_angle = np.arctan(self.wheelbase/radius)

        return speed, steering_angle

    def nearest_pt(self, point):
        dots = np.empty((self.wpts.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - self.wpts[i, :]), self.diffs[i, :])
        t = dots / self.l2s

        t = np.clip(dots / self.l2s, 0.0, 1.0)
        projections = self.wpts[:-1,:] + (t*self.diffs.T).T
        dists = np.linalg.norm(point - projections, axis=1)

        min_dist_segment = np.argmin(dists)
        return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

# @njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.
    Assumes that the first segment passes within a single radius of the point
    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

    # print min_dist_segment, dists[min_dist_segment], projections[min_dist_segment]


class FollowTheGap:
    def __init__(self, config):
        self.name = "Follow The Gap"
        self.config = config
        self.env_map = None
        self.map = None
        self.cur_scan = None
        self.cur_odom = None
    
        self.max_speed = config['lims']['max_v']
        self.max_steer = config['lims']['max_steer']
        self.wheelbase = config['car']['l_r'] + config['car']['l_f']
        mu = config['car']['mu']
        self.m = config['car']['m']
        g = config['car']['g']
        safety_f = config['pp']['force_f']
        self.f_max = mu * self.m * g * safety_f

        # n_beams = config['sim']['beams']
        n_beams = 20
        self.scan_sim = ScanSimulator(n_beams, np.pi)
        self.n_beams = n_beams
        
    def init_agent(self, env_map):
        self.scan_sim.set_check_fcn(env_map.check_scan_location)
        self.env_map = env_map


    def act(self, obs):
        scan = self.scan_sim.get_scan(obs[0], obs[1], obs[2])
        ranges = np.array(scan, dtype=np.float)
        o_ranges = ranges
        angle_increment = np.pi / len(ranges)

        max_range = 1
        # ranges = preprocess_lidar(ranges, max_range)

        bubble_r = 0.1
        ranges = create_zero_bubble(ranges, bubble_r)
        
        start_i, end_i = find_max_gap(ranges)

        aim = find_best_point(start_i, end_i, ranges[start_i:end_i])

        half_pt = len(ranges) /2
        steering_angle =  angle_increment * (aim - half_pt)

        val = ranges[aim] * 4
        th = lib.add_angles_complex(obs[2], steering_angle)
        pt = lib.theta_to_xy(th) * val
        self.env_map.targets.append(pt)

        speed = self.max_speed * ranges[aim] / max_range * 0.5
        # steering_angle = self.limit_inputs(speed, steering_angle)

        return np.array([speed, steering_angle])

    def limit_inputs(self, speed, steering_angle):
        max_steer = np.arctan(self.f_max * self.wheelbase / (speed**2 * self.m))
        new_steer = np.clip(steering_angle, -max_steer, max_steer)

        if max_steer < abs(steering_angle):
            print(f"Problem, Steering clipped from: {steering_angle} --> {max_steer}")

        return new_steer


@njit
def preprocess_lidar(ranges, max_range):
    ranges = np.array([min(ran, max_range) for ran in ranges])
    
    # moving_avg
    # n = 3
    # cumsum = np.cumsum(np.insert(ranges, 0, 0))
    # proc_ranges = (cumsum[n:] - cumsum[:-n])/float(n)

    proc_ranges = ranges

    return proc_ranges

# @njit
def create_zero_bubble(input_vector, bubble_r):
    centre = np.argmin(input_vector)
    min_dist = input_vector[centre]
    input_vector[centre] = 0
    size = len(input_vector)

    current_idx = centre
    while(current_idx < size -1 and input_vector[current_idx] < (min_dist + bubble_r)):
        input_vector[current_idx] = 0
        current_idx += 1
    
    current_idx = centre
    while(current_idx > 0  and input_vector[current_idx] < (min_dist + bubble_r)):
        input_vector[current_idx] = 0
        current_idx -= 1

    return input_vector
    
# @njit
def find_max_gap(input_vector):
    max_start = 0
    max_size = 0

    current_idx = 0
    size = len(input_vector)

    # exclude gaps that are smaller than this. Currently 1m
    min_distance = 0.5

    while current_idx < size:
        current_start = current_idx
        current_size = 0
        while current_idx< size and input_vector[current_idx] > min_distance:
            current_size += 1
            current_idx += 1
        if current_size > max_size:
            max_start = current_start
            max_size = current_size
            current_size = 0
        current_idx += 1
    if current_size > max_size:
        max_start = current_start
        max_size = current_size

    if max_size == 1:
        # max_start -= 1
        max_size = 3

    return max_start, max_start + max_size - 1


# @njit  
def find_best_point(start_i, end_i, ranges):
    # return best index to goto
    mid_i = (start_i + end_i) /2
    best_i = np.argmax(ranges)  
    best_i = (mid_i + (best_i + start_i)) /2

    return int(best_i)





