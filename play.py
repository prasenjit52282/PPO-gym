from library.ppo import PPO2
from library.env import Dummy_UAV_sim
from library.nn import Mlp_AC

max_reward=None
env=Dummy_UAV_sim()

ppo=PPO2(env,
		 Mlp_AC,
		 n_steps=4*512,
		 epochs=10,
		 steps_per_epoch=32,
		 shuffle_buffer_size=1024,
		 gamma=0.99,
		 lam=0.95,
		 vf_coef=0.5,
		 ent_coef=0.01,
		 learning_rate=0.00025,
		 max_grad_norm=0.5,
		 cliprange=0.2,
		 cliprange_vf=None)


ppo.learn(iterations=5000,max_reward=max_reward,test_at_iter=5,num_of_test=5,render_at_test=False)