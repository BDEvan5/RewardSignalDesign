import numpy as np 
import random
from matplotlib import pyplot as plt

from ModelsRL import TD3

import LibFunctions as lib
from Simulator import ScanSimulator



class BaseGenAgent:
    def __init__(self, config, name):
        self.config = config
        self.name = name
        self.env_map = None
        self.wpts = None

        self.path_name = 'Vehicles/%s' % self.name
        self.pind = 1
        self.target = None

        # history
        self.steer_history = []
        self.vel_history = []
        self.d_ref_history = []
        self.reward_history = []
        self.critic_history = []
        self.steps = 0

        self.max_v = config['lims']['max_v']
        self.max_d = config['lims']['max_steer']
        self.mu = config['car']['mu']
        self.m = config['car']['m']
        self.max_friction_force = 9.81 * self.m * self.mu 

        # agent stuff 
        self.state_action = None
        self.cur_nn_act = None
        self.prev_nn_act = 0

        n_beams = config['sim']['beams']
        self.scan_sim = ScanSimulator(n_beams, np.pi)
        self.n_beams = n_beams

    def init_agent(self, env_map):
        self.env_map = env_map
        
        self.scan_sim.set_check_fcn(self.env_map.check_scan_location)

        # self.wpts = self.env_map.get_min_curve_path()
        self.wpts = self.env_map.get_reference_path()

        self.prev_dist_target = lib.get_distance(self.env_map.start, self.env_map.end)

        return self.wpts

    def show_vehicle_history(self):
        plt.figure(1)
        plt.clf()
        plt.title("Steer History")
        plt.ylim([-1.1, 1.1])
        # np.save('Vehicles/mod_hist', self.steer_history)
        plt.plot(self.steer_history)
        plt.plot(self.vel_history)
        plt.legend(['Steer', 'Velocity'])

        plt.pause(0.001)

        # plt.figure(3)
        # plt.clf()
        # plt.title('Rewards')
        # plt.ylim([-1.5, 4])
        # plt.plot(self.reward_history, 'x', markersize=12)
        # plt.plot(self.critic_history)

    def transform_obs(self, obs):
        cur_v = [obs[3]/self.max_v]
        cur_d = [obs[4]/self.max_d]

        th_target = lib.get_bearing(obs[0:2], self.env_map.end)
        alpha = lib.sub_angles_complex(th_target, obs[2])
        th_scale = [(alpha)*2/np.pi]

        scan = self.scan_sim.get_scan(obs[0], obs[1], obs[2])

        nn_obs = np.concatenate([cur_v, cur_d, th_scale, scan])

        return nn_obs

    def reset_lap(self):
        self.steer_history.clear()
        self.vel_history.clear()
        self.d_ref_history.clear()
        self.reward_history.clear()
        self.critic_history.clear()
        self.steps = 0
        self.pind = 1

    def generate_references(self, nn_action, space=None):
        v_ref = (nn_action[0] + 1) / 2 * self.max_v # change the min from -1 to 0
        d_ref = nn_action[1] * self.max_d

        return v_ref, d_ref



class GenVehicle(BaseGenAgent):
    def __init__(self, config, name, load=False):
        BaseGenAgent.__init__(self, config, name)

        state_space = 3 + self.n_beams
        self.agent = TD3(state_space, 2, 1, name)
        h_size = config['nn']['h']
        self.agent.try_load(load, h_size, path=self.path_name)

    def act(self, obs):
        nn_obs = self.transform_obs(obs)
        nn_action = self.agent.act(nn_obs)

        self.critic_history.append(self.agent.get_critic_value(nn_obs, nn_action))
        self.state_action = [nn_obs, nn_action]

        v_ref, d_ref = self.generate_references(nn_action, obs)
        self.steer_history.append(d_ref/self.max_d)
        self.vel_history.append(v_ref/self.max_v)

        self.steps += 1

        return [v_ref, d_ref]

    def add_memory_entry(self, new_reward, done, s_prime, buffer):
        self.prev_nn_act = self.state_action[1][0]

        nn_s_prime = self.transform_obs(s_prime)

        mem_entry = (self.state_action[0], self.state_action[1], nn_s_prime, new_reward, done)

        buffer.add(mem_entry)



"""Test Vehicles"""
class GenTest(BaseGenAgent):
    def __init__(self, config, name):
        path = 'Vehicles/' + name + ''
        self.agent = TD3(1, 1, 1, name)
        self.agent.load(directory=path)

        print(f"NN: {self.agent.actor.type}")

        nn_size = self.agent.actor.l1.in_features
        n_beams = nn_size - 3
        BaseGenAgent.__init__(self, config, name)

    def act(self, obs):
        nn_obs = self.transform_obs(obs)
        nn_action = self.agent.act(nn_obs, noise=0)

        v_ref, d_ref = self.generate_references(nn_action)

        return [v_ref, d_ref]




