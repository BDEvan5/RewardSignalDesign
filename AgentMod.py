import numpy as np 
from matplotlib import pyplot as plt

from ModelsRL import TD3

import LibFunctions as lib
from Simulator import ScanSimulator



class BaseMod:
    def __init__(self, config, agent_name) -> None:
        self.config = config
        self.name = agent_name
        self.env_map = None
        self.path_name = None

        mu = config['car']['mu']
        self.m = config['car']['m']
        g = config['car']['g']
        safety_f = config['pp']['force_f']
        self.f_max = mu * self.m * g #* safety_f
        self.max_v = config['lims']['max_v']
        self.max_d = config['lims']['max_steer']
        self.lookahead = config['pp']['lookahead']
        self.vgain = config['pp']['v_gain']
        self.wheelbase =  config['car']['l_f'] + config['car']['l_r']

        self.wpts = None
        self.vs = None
        self.steps = 0

        self.mod_history = []
        self.d_ref_history = []
        self.reward_history = []
        self.critic_history = []

        n_beams = config['sim']['beams']
        self.scan_sim = ScanSimulator(n_beams, np.pi)
        self.n_beams = n_beams


    def init_agent(self, env_map):
        self.env_map = env_map
        self.scan_sim.set_check_fcn(self.env_map.check_scan_location)


        self.reset_lap()

    def _get_current_waypoint(self, position):
        nearest_pt, nearest_dist, t, i = self.nearest_pt(position)

        # if nearest_dist < self.lookahead:
        #     lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, self.lookahead, self.wpts, i+t, wrap=True)
        #     if i2 == None:
        #         return None
        #     i = i2
        # elif nearest_dist < 20:
        #     return np.append(self.wpts[i], self.vs[i])
        return np.append(self.wpts[i], self.vs[i])

    def act_pp(self, obs):
        pose_th = obs[2]
        pos = np.array(obs[0:2], dtype=np.float)

        lookahead_point = self._get_current_waypoint(pos)

        if lookahead_point is None:
            return 4.0, 0.0
        self.env_map.targets.append(lookahead_point[0:2])

        speed, steering_angle = self.get_actuation(pose_th, lookahead_point, pos)
        speed = self.vgain * speed

        steering_angle = self.limit_inputs(max(speed, obs[3]), steering_angle)

        return speed, steering_angle

    def limit_inputs(self, speed, steering_angle):
        max_steer = np.arctan(self.f_max * self.wheelbase / (speed**2 * self.m))
        new_steer = np.clip(steering_angle, -max_steer, max_steer)

        # if max_steer < abs(steering_angle):
            # print(f"Problem, Steering clipped from: {steering_angle} --> {max_steer}")

        return new_steer

    def reset_lap(self):
        # self.wpts, self.vs = self.env_map.get_optimal_path()
        self.wpts, self.vs = self.env_map.get_reference_path()

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

        self.steps = 0
        self.mod_history = []


    def get_actuation(self, pose_theta, lookahead_point, position):
        waypoint_y = np.dot(np.array([np.cos(pose_theta), np.sin(-pose_theta)]), lookahead_point[0:2]-position)
        
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

    def show_vehicle_history(self):
        # plt.figure(1)
        # plt.clf()
        # plt.title("Mod History")
        # plt.ylim([-1.1, 1.1])
        # plt.plot(self.mod_history)
        # np.save('Vehicles/mod_hist', self.mod_history)
        # # plt.plot(self.d_ref_history)
        # plt.legend(['NN'])


        plt.figure(3)
        plt.clf()
        plt.title('Rewards')
        plt.ylim([-1.5, 4])
        plt.plot(self.reward_history, 'x', markersize=12)
        plt.plot(self.critic_history)
        plt.pause(0.001)

    def transform_obs(self, obs):
        v_ref, d_ref = self.act_pp(obs)

        cur_v = [obs[3]/self.max_v]
        cur_d = [obs[4]/self.max_d]
        vr_scale = [(v_ref)/self.max_v]
        dr_scale = [d_ref/self.max_d]

        scan = self.scan_sim.get_scan(obs[0], obs[1], obs[2])

        nn_obs = np.concatenate([cur_v, cur_d, vr_scale, dr_scale, scan])

        return nn_obs

    def modify_references(self, nn_action, v_ref, d_ref, obs):
        d_max = 0.4 #- use this instead
        d_phi = d_max * nn_action[0] # rad
        d_new = d_ref + d_phi
        d_new = np.clip(d_new, -d_max, d_max)

        if abs(d_new) > abs(d_ref):
            max_friction_force = 3.74 * 9.81 * 0.523 *0.5
            d_plan = max(abs(d_ref), abs(obs[4]), abs(d_new))
            theta_dot = abs(obs[3] / 0.33 * np.tan(d_plan))
            v_ref_new = max_friction_force / (3.74 * max(theta_dot, 0.01)) 
            v_ref_mod = min(v_ref_new, self.max_v)
        else:
            v_ref_mod = v_ref


        return v_ref_mod, d_new


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




class ModVehicleTrain(BaseMod):
    def __init__(self, config, name, load=False):
        BaseMod.__init__(self, config, name)

        self.current_v_ref = None
        self.current_phi_ref = None

        state_space = 4 + self.n_beams
        self.agent = TD3(state_space, 1, 1, name)
        h_size = config['nn']['h']
        self.agent.try_load(load, h_size)

        self.m1 = None
        self.m2 = None

    def act(self, obs):
        # v_ref, d_ref = self.get_target_references(obs)
        v_ref, d_ref = self.act_pp(obs)

        nn_obs = self.transform_obs(obs)
        nn_action = self.agent.act(nn_obs)
        self.cur_nn_act = nn_action

        self.d_ref_history.append(d_ref)
        self.mod_history.append(self.cur_nn_act[0])
        self.critic_history.append(self.agent.get_critic_value(nn_obs, nn_action))
        self.state_action = [nn_obs, self.cur_nn_act]

        v_ref, d_ref = self.modify_references(self.cur_nn_act, v_ref, d_ref, obs)

        self.steps += 1

        return [v_ref, d_ref]

    def add_memory_entry(self, new_reward, done, s_prime, buffer):
        self.prev_nn_act = self.state_action[1][0]

        nn_s_prime = self.transform_obs(s_prime)

        mem_entry = (self.state_action[0], self.state_action[1], nn_s_prime, new_reward, done)

        buffer.add(mem_entry)

    def get_deviation(self):
        return self.cur_nn_act[0]


class ModVehicleTest(BaseMod):
    def __init__(self, config, name):
        path = 'Vehicles/' + name + ''
        state_space = 4 
        self.agent = TD3(state_space, 1, 1, name)
        self.agent.load(directory=path)

        print(f"NN: {self.agent.actor.type}")

        nn_size = self.agent.actor.l1.in_features
        n_beams = nn_size - 4
        BaseMod.__init__(self, config, name)
        print(f"Agent loaded: {name}")

        self.current_v_ref = None
        self.current_phi_ref = None

    def act(self, obs):
        v_ref, d_ref = self.act_pp(obs)

        nn_obs = self.transform_obs(obs)
        nn_action = self.agent.act(nn_obs, noise=0)
        self.cur_nn_act = nn_action

        self.d_ref_history.append(d_ref)
        self.mod_history.append(self.cur_nn_act[0])
        self.critic_history.append(self.agent.get_critic_value(nn_obs, nn_action))
        self.state_action = [nn_obs, self.cur_nn_act]

        v_ref, d_ref = self.modify_references(self.cur_nn_act, v_ref, d_ref, obs)

        self.steps += 1

        return [v_ref, d_ref]


