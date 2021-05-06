

from LibFunctions import load_config

from RewardSignalDesign.AgentOptimal import FollowTheGap, PurePursuit
from RewardSignalDesign.AgentMod import ModVehicleTest, ModVehicleTrain

from TestingScripts.TrainTestUtils import TestVehicles

config_rt = "race_track"
env_name = "porto"
train_name = "_test"
test_name = "compare_" + env_name + train_name


def FullTest():
    config = load_config(config_rt)
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
    vehicle = PurePursuit(config)
    test.add_vehicle(vehicle)

    # FTG
    vehicle = FollowTheGap(config)
    test.add_vehicle(vehicle)

    test.run_eval(1, False, add_obs=False, save=False)
    test.eval_name += "_Obs"
    test.run_eval(10, False, add_obs=True, save=False)



if __name__ == "__main__":
    FullTest()





