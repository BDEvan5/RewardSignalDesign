import numpy as np
import csv, yaml
from Rewards import CthReward, TimeReward, SteerReward


import LibFunctions as lib
from LibFunctions import load_config

# from AgentOptimal import OptimalAgent
from AgentOptimal import FollowTheGap, TunerCar
from AgentMod import ModVehicleTest, ModVehicleTrain


config_sf = "small_forest"
config_std = "std_config"


from Testing import TestVehicles, TrainVehicle

config_sf = "small_forest"
config_std = "std_config"
config_med = "med_forest"
config_old = "old_med"


"""Mod training"""
def train_mod_steer():
    agent_name = "ModSteer_test_omr"

    config = load_config(config_old)
    vehicle = ModVehicleTrain(config, agent_name, load=False)
    reward = SteerReward(config, 0.1, 0.1)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def train_mod_time():
    agent_name = "ModTime_test_f"
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name)
    reward = TimeReward(config, 0.12)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def train_mod_cth():
    agent_name = "ModCth_test_f"
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name)
    reward = CthReward(config, 0.4, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)


def train_mod_dev():
    agent_name = "ModDev_test_f"
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name)

    reward = DevReward(config)

    TrainVehicle(config, agent_name, vehicle, reward, 20000, 'track')

def train_mod_std():
    agent_name = "ModStd_test_f"
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name)

    reward = StdReward(config)

    TrainVehicle(config, agent_name, vehicle, reward, 20000, 'track')

def train_mod_old():
    agent_name = "ModOld_test_rt"
    config = load_config(config_med)
    vehicle = ModVehicleTrain(config, agent_name)

    reward = OldReward(config)

    TrainVehicle(config, agent_name, vehicle, reward, 20000, 'track')


"""Tests """

def FullTrainRT():
    # config = load_config(config_med)
    config = load_config(config_med)
    env_name = "porto"
    n_train = 20000

    agent_name = "ModSteer_"  + env_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = SteerReward(config, 0.1, 0.1)

    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # agent_name = "ModTime_" + env_name
    # vehicle = ModVehicleTrain(config, agent_name)
    # reward = TrackTimeReward(config, 0.12)

    # TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    agent_name = "ModCth_" + env_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = CthReward(config, 0.4, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    agent_name = "ModOld_" + env_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = OldReward(config)

    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    agent_name = "ModStd_" + env_name
    vehicle = ModVehicleTrain(config, agent_name)
    reward = StdReward(config)

    TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

    # agent_name = "ModDev_" + env_name
    # vehicle = ModVehicleTrain(config, agent_name)
    # reward = TrackDevReward(config)

    # TrainVehicle(config, agent_name, vehicle, reward, n_train, 'track')

def FullTest():
    # config = load_config(config_med)
    # config = load_config(config_std)
    config = load_config(config_med)

    env_name = "porto"
    test_name = "compare_" + env_name + "_8_noObs"
    test = TestVehicles(config, test_name, 'track')

    # mod
    # agent_name = "ModTime_" + env_name
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    agent_name = "ModCth_" + env_name
    # agent_name = "ModCth_test"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModSteer_" + env_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # old
    # agent_name = "ModDev_" + env_name
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    agent_name = "ModOld_" + env_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModStd_" + env_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # PP
    vehicle = TunerCar(config)
    test.add_vehicle(vehicle)

    # FTG
    # vehicle = FollowTheGap(config)
    # test.add_vehicle(vehicle)

    test.run_eval(1, True, add_obs=False)
    # test.run_eval(10, True, add_obs=True, save=True)
    # test.run_eval(100, False, add_obs=True, save=True)

    # test.run_eval(10, True)

"""Time sweep"""
def train_time_sweep():
    load = False
    config = load_config(config_med)

    agent_name = "ModTime_test_04"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)


    agent_name = "ModTime_test_06"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.06)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModTime_test_08"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.08)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModTime_test_10"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.1)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModTime_test_15"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.15)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModTime_test_18"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.18)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModTime_test_20"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.20)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModTime_test_25"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = TimeReward(config, 0.25)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def test_time_sweep():
    config = load_config(config_med)

    test = TestVehicles(config, "test_time_sweep")

    # mod
    agent_name = "ModTime_test_04"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_06"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_08"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_10"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_15"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_18"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_20"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_test_25"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    test.run_eval(100, False)

"""Steer sweep"""
def train_steer_sweep():
    load = False
    config = load_config(config_med)

    agent_name = "ModSteer_test_004_004"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = SteerReward(config, 0.04, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)
    agent_name = "ModSteer_test_008_008"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = SteerReward(config, 0.08, 0.08)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    # agent_name = "ModSteer_test_01_01"
    # vehicle = ModVehicleTrain(config, agent_name, load)
    # reward = SteerReward(config, 0.1, 0.1)

    # TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModSteer_test_015_015"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = SteerReward(config, 0.15, 0.15)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

    agent_name = "ModSteer_test_02_02"
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = SteerReward(config, 0.2, 0.2)

    TrainVehicle(config, agent_name, vehicle, reward, 4000)

def test_steer_sweep():
    config = load_config(config_med)

    test = TestVehicles(config, "test_steer_sweep")

    # mod
    agent_name = "ModSteer_test_004_004"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModSteer_test_008_008"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModSteer_test_01_01"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModSteer_test_015_015"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModSteer_test_02_02"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)


    test.run_eval(100, False)



"""Smaller tests"""

def test_ftg():
    # config = load_config(config_med)
    config = load_config(config_med)

    vehicle = TunerCar(config)
    # vehicle = FollowTheGap(config)

    test = TestVehicles(config, "FTG", 'track')
    test.add_vehicle(vehicle)
    test.run_eval(10, True, add_obs=False)
    # testVehicle(config, vehicle, True, 10)

def test_mod():
    config = load_config(config_old)
    # agent_name = "ModTime_raceTrack"

    agent_name = "ModSteer_test_omr"
    # agent_name = "ModCth_test_f"
    # agent_name = "ModCth_test"
    # agent_name = "ModDev_test_f"
    # agent_name = "ModDev_test_f"
    # agent_name = "ModOld_test_f"
    # agent_name = "ModTime_test_f"
    # agent_name = "ModTime_medForest"
    # agent_name = "ModDev_raceTrack"
    vehicle = ModVehicleTest(config, agent_name)
    # vehicle = TunerCar(config)

    test = TestVehicles(config, "Mod_test_f")
    test.add_vehicle(vehicle)

    agent_name = "ModSteer_test_om"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # test.run_eval(10, True, add_obs=False)
    test.run_eval(100, True, add_obs=True)


def train():
    pass

    # train_mod_cth()
    # train_mod_time()
    train_mod_steer()

    # train_mod_dev()
    # train_mod_std()
    # train_mod_old()

    # train_time_sweep()
    # train_steer_sweep()


if __name__ == "__main__":
    # train()


    # test_compare()
    # test_compare_mod()
    # test_time_sweep()
    # test_steer_sweep()

    # FullTrainf()
    # FullTest()


    test_mod()
    # test_ftg()
