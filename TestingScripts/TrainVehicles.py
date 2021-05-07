

from RewardSignalDesign.LibFunctions import load_config
import RewardSignalDesign.Rewards as r
from RewardSignalDesign.AgentMod import ModVehicleTrain

from TrainTestUtils import TrainVehicle


config_rt = "race_track"
n_train_itterations = 100000
map_name = "porto"
train_name = "_final"


def train_mod_emp():
    agent_name = "ModEmp_" + map_name + train_name

    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name, load=False)
    reward = r.EmptyR()

    TrainVehicle(config, agent_name, vehicle, reward, n_train_itterations, 'track', show=False)

def train_distance_center():
    agent_name = "ModCenterDis_" + map_name + train_name
    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.CenterDistanceReward(config, 0.02)

    TrainVehicle(config, agent_name, vehicle, reward, n_train_itterations, 'track')

def train_distance_ref():
    agent_name = "ModRefDis_" + map_name + train_name
    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.CenterCTHReward(config, 0.4, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, n_train_itterations, 'track')

def train_cth_center():
    agent_name = "ModCenterCth_" + map_name + train_name
    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.CenterCTHReward(config, 0.4, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, n_train_itterations, 'track')

def train_cth_ref():
    agent_name = "ModRefCth_" + map_name + train_name
    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name)
    reward = r.RefCTHReward(config, 0.4, 0.04)

    TrainVehicle(config, agent_name, vehicle, reward, n_train_itterations, 'track')

def train_mod_steer():
    agent_name = "ModSteer_" + map_name + train_name

    config = load_config(config_rt)
    vehicle = ModVehicleTrain(config, agent_name, load=False)
    reward = r.TrackSteerReward(config, 0, 0.005)

    TrainVehicle(config, agent_name, vehicle, reward, n_train_itterations, 'track', show=False)




def train():
    pass

    # train_mod_emp()
    # train_distance_center()
    # train_distance_ref()
    # train_cth_center()
    train_cth_ref()
    # train_mod_steer()



if __name__ == "__main__":
    train()




