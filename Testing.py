

from matplotlib import pyplot as plt
from HistoryStructs import TrainHistory
from Simulator import ForestSim, TrackSim
from SimMaps import  ForestMap, SimMap

from ModelsRL import ReplayBufferDQN, ReplayBufferTD3
import numpy as np
import csv


import LibFunctions as lib



"""Train"""
def TrainVehicle(config, agent_name, vehicle, reward, steps=20000, env_kwarg='forest', show=False):
    path = 'Vehicles/' + agent_name
    buffer = ReplayBufferTD3()

    if env_kwarg == 'forest':
        env_map = ForestMap(config)
        env = ForestSim(env_map)
    else:
        env_map = SimMap(config)
        env = TrackSim(env_map)


    t_his = TrainHistory(agent_name)

    print_n = 500
    add_obs = True

    done = False
    state, wpts, vs = env.reset(add_obs=add_obs)
    vehicle.init_agent(env_map)

    for n in range(steps):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step(a)

        deviation = vehicle.get_deviation()
        new_r = reward(state, a, s_prime, r, deviation)
        vehicle.add_memory_entry(new_r, done, s_prime, buffer)
        t_his.add_step_data(new_r)

        state = s_prime
        vehicle.agent.train(buffer, 2)
        
        # env.render(False)

        if n % print_n == 0 and n > 0:
            t_his.print_update()
            vehicle.agent.save(directory=path)
        
        if done:
            if show:
                t_his.lap_done(True)
                env.render(wait=False, save=False)
                # vehicle.show_vehicle_history()
            else:
                t_his.lap_done(False)

            vehicle.reset_lap()
            state, wpts, vs = env.reset(add_obs=add_obs)


    vehicle.agent.save(directory=path)
    t_his.save_csv_data()

    print(f"Finished Training: {agent_name}")

    return t_his.rewards


"""Testing Function"""
class TestData:
    def __init__(self) -> None:
        self.endings = None
        self.crashes = None
        self.completes = None
        self.lap_times = None

        self.names = []
        self.lap_histories = None

        self.N = None

    def init_arrays(self, N, laps):
        self.completes = np.zeros((N))
        self.crashes = np.zeros((N))
        self.lap_times = np.zeros((laps, N))
        self.endings = np.zeros((laps, N)) #store env reward
        self.lap_times = [[] for i in range(N)]
        self.N = N
 
    def save_txt_results(self):
        test_name = 'Evals/' + self.eval_name + '.txt'
        with open(test_name, 'w') as file_obj:
            file_obj.write(f"\nTesting Complete \n")
            file_obj.write(f"Map name:  \n")
            file_obj.write(f"-----------------------------------------------------\n")
            file_obj.write(f"-----------------------------------------------------\n")
            for i in range(self.N):
                file_obj.write(f"Vehicle: {self.vehicle_list[i].name}\n")
                file_obj.write(f"Crashes: {self.crashes[i]} --> Completes {self.completes[i]}\n")
                percent = (self.completes[i] / (self.completes[i] + self.crashes[i]) * 100)
                file_obj.write(f"% Finished = {percent:.2f}\n")
                file_obj.write(f"Avg lap times: {np.mean(self.lap_times[i])}\n")
                file_obj.write(f"-----------------------------------------------------\n")

    def print_results(self):
        print(f"\nTesting Complete ")
        print(f"-----------------------------------------------------")
        print(f"-----------------------------------------------------")
        for i in range(self.N):
            if len(self.lap_times[i]) == 0:
                self.lap_times[i].append(0)
            print(f"Vehicle: {self.vehicle_list[i].name}")
            print(f"Crashes: {self.crashes[i]} --> Completes {self.completes[i]}")
            percent = (self.completes[i] / (self.completes[i] + self.crashes[i]) * 100)
            print(f"% Finished = {percent:.2f}")
            print(f"Avg lap times: {np.mean(self.lap_times[i])}")
            print(f"-----------------------------------------------------")
        
    def save_csv_results(self):
        test_name = 'Evals/'  + self.eval_name + '.csv'

        data = [["#", "Name", "%Complete", "AvgTime", "Std"]]
        for i in range(self.N):
            v_data = [i]
            v_data.append(self.vehicle_list[i].name)
            v_data.append((self.completes[i] / (self.completes[i] + self.crashes[i]) * 100))
            v_data.append(np.mean(self.lap_times[i]))
            v_data.append(np.std(self.lap_times[i]))
            data.append(v_data)

        with open(test_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)

    # def load_csv_data(self, eval_name):
    #     file_name = 'Vehicles/Evals' + eval_name + '.csv'

    #     with open(file_name, 'r') as csvfile:
    #         csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
            
    #         for lines in csvFile:  
    #             self.
    #             rewards.append(lines)


    # def plot_eval(self):
    #     pass


class TestVehicles(TestData):
    def __init__(self, config, eval_name, env_kwarg='forest') -> None:
        self.config = config
        self.eval_name = eval_name
        self.vehicle_list = []
        self.N = None
        self.env_kwarg = env_kwarg

        TestData.__init__(self)

    def add_vehicle(self, vehicle):
        self.vehicle_list.append(vehicle)

    def run_eval(self, laps=100, show=False, add_obs=True, save=False, wait=False):
        N = self.N = len(self.vehicle_list)
        self.init_arrays(N, laps)

        if self.env_kwarg == 'forest':
            env_map = ForestMap(self.config)
            env = ForestSim(env_map)    
        else:
            env_map = SimMap(self.config)
            env = TrackSim(env_map)

        path = 'Evals/imgs/'


        for i in range(laps):
            for j in range(N):
                vehicle = self.vehicle_list[j]

                r, steps = self.run_lap(vehicle, env, show, add_obs, wait)
                
                if save:
                    plt.figure(4)
                    plt.title(vehicle.name)
                    plt.savefig(path + f"lap_{i}_{vehicle.name}")

                
                print(f"#{i}: Lap time for ({vehicle.name}): {env.steps} --> Reward: {r}")
                self.endings[i, j] = r
                if r == -1 or r == 0:
                    self.crashes[j] += 1
                else:
                    self.completes[j] += 1
                    self.lap_times[j].append(steps)

        self.print_results()
        self.save_txt_results()
        self.save_csv_results()

    def run_lap(self, vehicle, env, show, add_obs, wait):
        state, wpts, vs = env.reset(add_obs)
        # env.render(wait=True)
        vehicle.init_agent(env.env_map)
        done = False
        while not done:
            a = vehicle.act(state)
            s_p, r, done, _ = env.step(a)
            state = s_p
            # env.render(False)

        if show:
            # vehicle.show_vehicle_history()
            # env.show_history()
            # env.history.show_history()
            env.render(wait=False)
            if wait:
                env.render(wait=True)

        return r, env.steps

