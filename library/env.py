import sys
import numpy as np
sys.path.append("../../")
from UAV_sim.gym_wrapper.gymEnv import UAV_sim

class Dummy_UAV_sim:
    def __init__(self,length=300,bredth=300,max_height=80,max_time=500,
                num_uav=10,num_mine=30,num_target=10,pix=80,uavs_at_plane=4,seed=101):
        self.num_uav=num_uav
        self.env=UAV_sim(length=length,bredth=bredth,max_height=max_height,max_time=max_time,
                    num_uav=num_uav,num_mine=num_mine,num_target=num_target,pix=pix,uavs_at_plane=uavs_at_plane,seed=seed)
        