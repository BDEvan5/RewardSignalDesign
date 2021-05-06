from Testing import TestVehicles
import numpy as np
from HistoryStructs import RewardAnalyser, TrainHistory

import timeit
import yaml

from Simulator import ForestSim
from SimMaps import  ForestMap
from ModelsRL import  ReplayBufferTD3
import LibFunctions as lib
from LibFunctions import load_config
from Rewards import *

from AgentOptimal import OptimalAgent, TunerCar

from AgentMod import ModVehicleTest, ModVehicleTrain
from RefGen import GenVehicle, GenTest


config_sf = "small_forest"
config_std = "std_config"
config_med = "med_forest"



"""Train"""
def TrainVehicle(config, agent_name, vehicle, reward, steps=20000):
    path = 'Vehicles/' + agent_name
    buffer = ReplayBufferTD3()

    # env_map = SimMap(name)
    # env = TrackSim(env_map)

    env_map = ForestMap(config)
    env = ForestSim(env_map)

    t_his = TrainHistory(agent_name)
    print_n = 500

    done = False
    state, wpts, vs = env.reset()
    vehicle.init_agent(env_map)
    reward.init_reward(wpts, vs)

    for n in range(steps):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step(a)

        new_r = reward(state, a, s_prime, r)
        vehicle.add_memory_entry(new_r, done, s_prime, buffer)
        t_his.add_step_data(new_r)

        state = s_prime
        vehicle.agent.train(buffer, 2)
        
        # env.render(False)

        if n % print_n == 0 and n > 0:
            t_his.print_update()
            vehicle.agent.save(directory=path)
        
        if done:
            t_his.lap_done(True)
            # vehicle.show_vehicle_history()
            env.render(wait=False, save=False)

            vehicle.reset_lap()
            state, wpts, vs = env.reset()
            reward.init_reward(wpts, vs)


    vehicle.agent.save(directory=path)
    t_his.save_csv_data()

    print(f"Finished Training: {agent_name}")

    return t_his.rewards

"""General test function"""
def testVehicle(config, vehicle, show=False, laps=100):
    # env_map = SimMap(name)
    # env = TrackSim(env_map)

    env_map = ForestMap(config)
    env = ForestSim(env_map)

    crashes = 0
    completes = 0
    lap_times = [] 

    state, w, v = env.reset()
    vehicle.init_agent(env_map)
    done, score = False, 0.0
    for i in range(laps):
        print(f"Running lap: {i}")
        while not done:
            a = vehicle.act(state)
            s_p, r, done, _ = env.step(a)
            state = s_p
            # env.render(False, vehicle.scan_sim)
        print(f"Lap time updates: {env.steps}")
        if show:
            # env.history.show_history(vs=env_map.vs)
            # env.history.show_forces()
            env.render(wait=False)
            # plt.pause(1)
            # env.render(wait=True)

        if r == -1:
            crashes += 1
        else:
            completes += 1
            lap_times.append(env.steps)
        state, w, v = env.reset()
        
        vehicle.reset_lap()
        done = False

    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {(completes / (completes + crashes) * 100):.2f} %")
    print(f"Lap times: {lap_times} --> Avg: {np.mean(lap_times)}")


""" Training sets"""
def train_gen_time():
    load = False

    agent_name = "GenTime_test"
    config = load_config(config_med)
    vehicle = GenVehicle(config, agent_name, load)
    reward = TimeReward(config, 0.06)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def train_gen_cth():
    load = False

    agent_name = "GenCth_test"
    config = load_config(config_med)
    vehicle = GenVehicle(config, agent_name, load)
    reward = CthReward(config, 0.4, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def train_gen_steer():
    load = False

    agent_name = "GenSteer_test"
    config = load_config(config_med)
    vehicle = GenVehicle(config, agent_name, load)
    reward = SteerReward(config, 0.1, 0.1)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

"""Mod training"""
def train_mod_steer():
    load = False

    agent_name = "ModSteer_test"
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = SteerReward(config, 0.1, 0.1)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def train_mod_time():
    load = False

    agent_name = "ModTime_test"
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.06)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def train_mod_cth():
    load = False

    agent_name = "ModCth_test"
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = CthReward(config, 0.4, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)




"""Total functions"""
def test_Gen():
    agent_name = "GenCth_test"
    # agent_name = "GenTime_test"
    # agent_name = "GenSteer_test"
    

    config = load_config(config_med)
    vehicle = GenTest(config, agent_name)

    testVehicle(config, vehicle, True, 10)
    
def test_Mod():
    agent_name = "ModSteer_test"
    # agent_name = "ModCth_test"
    # agent_name = "ModTime_test"
    
    config = load_config("std_config")
    config = load_config(config_med)
    # config = load_config(config_sf)
    vehicle = ModVehicleTest(config, agent_name)

    testVehicle(config, vehicle, True, laps=20)


def testOptimal():

    # config = load_config(config_std)
    config = load_config(config_med)
    vehicle = TunerCar(config)
    # vehicle = OptimalAgent(config) # to be deprecated for tuner car

    testVehicle(config, vehicle, True, 10)

def test_compare():
    config = load_config(config_med)
    # config = load_config(config_std)
    # test = TestVehicles(config, "test_compare_mod")
    # test = TestVehicles(config, "test_compare_gen")
    test = TestVehicles(config, "test_compare")

    # mod
    agent_name = "ModTime_test"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModCth_test"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModSteer_test"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # gen
    agent_name = "GenTime_test"
    vehicle = GenTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "GenCth_test"
    vehicle = GenTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "GenSteer_test"
    vehicle = GenTest(config, agent_name)
    test.add_vehicle(vehicle)

    # PP
    vehicle = TunerCar(config)
    test.add_vehicle(vehicle)

    test.run_eval(100, True)


        

def timing():
    # t = timeit.timeit(stmt=RunModAgent, number=1)
    # print(f"Time: {t}")
    
    t = timeit.timeit(stmt=testOptimal, number=1)
    print(f"Time: {t}")


if __name__ == "__main__":

    # train_gen_time()
    # train_gen_steer()
    # train_gen_cth()

    # train_mod_steer()
    # train_mod_cth()
    # train_mod_time()


    # train_mod_time_sweep_m1()
    # train_mod_time_sweep_m2()
    # train_mod_time_sweep_mt()

    # test_mod_time_sweep_m1()
    # test_mod_time_sweep_m2()
    # test_mod_time_sweep_mt()

    # test_Gen()
    test_Mod()
    # testOptimal()
    # test_compare()

    # timing()

    # RunMpcAgent()
    # test_mapping()





    
