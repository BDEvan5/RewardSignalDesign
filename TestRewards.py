
import numpy as np
import csv, yaml
import Rewards as r

import LibFunctions as lib
from LibFunctions import load_config
from matplotlib import pyplot as plt

# from AgentOptimal import OptimalAgent
from AgentOptimal import FollowTheGap, TunerCar
from AgentMod import ModVehicleTest, ModVehicleTrain


config_sf = "small_forest"
config_std = "std_config"


from Testing import TestVehicles, TrainVehicle

config_sf = "small_forest"
config_std = "std_config"
config_med = "med_forest"
config_rt = "race_track"





def FullTest():
    config = load_config(config_rt)

    env_name = "porto"
    train_name = "_test"
    test_name = "compare_" + env_name + train_name
    # test_name = "compare_NoObs_" + env_name + train_name
    test = TestVehicles(config, test_name, 'track')

    # 1) no racing reward
    agent_name = "ModEmp_" + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # 3) Distance Centerline
    agent_name = "ModCenterDis_" + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # 6) distance ref
    agent_name = "ModRefDis_" + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # 4) CTH center
    agent_name = "ModCenterCth_" + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)


    # 7) CTH ref
    agent_name = "ModRefCth_" + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # 8) Steering and Velocity
    agent_name = "ModSteer_"  + env_name + train_name
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    # PP
    vehicle = TunerCar(config)
    test.add_vehicle(vehicle)

    # FTG
    vehicle = FollowTheGap(config)
    test.add_vehicle(vehicle)

    test.run_eval(1, False, add_obs=False, save=False)
    test.eval_name += "_Obs"
    test.run_eval(10, False, add_obs=True, save=False)





if __name__ == "__main__":
    FullTest()





