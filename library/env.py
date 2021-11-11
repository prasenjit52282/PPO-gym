import sys
import numpy as np
sys.path.append("../../")
from UAV_sim.gym_wrapper.gymEnv import UAV_sim
from UAV_sim.gym_wrapper.preprocessor import Preprocessor

class Dummy_UAV_sim:
    def __init__(self,length=300,bredth=300,max_height=150,max_time=500,
                num_uav=10,num_mine=30,num_target=10,pix=80):
        self.num_uav=num_uav
        self.env=UAV_sim(length=length,bredth=bredth,max_height=max_height,max_time=max_time,
                    num_uav=num_uav,num_mine=num_mine,num_target=num_target)
        self.preprocessor=Preprocessor(self.env,pix=pix)

    def preprocess(self,state,padding=True):
        return self.preprocessor.preprocess(state,padding=padding)

    @property
    def curr_state(self):
        return self.preprocess(self.env.curr_state)

    def reset(self):
        state=self.env.reset()
        return self.preprocess(state)
        

    def step(self,act):
        next_state,reward,notdones,(done,info)=self.env.step(act)
        return self.preprocess(next_state),np.array(reward),np.array(notdones),(done,info)

    def close(self):
        pass
        