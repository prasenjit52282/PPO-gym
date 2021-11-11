import tensorflow as tf
from .helper import *
from .memory import Memory
from .logger import TensorboardLogger
class PPO2:
	def __init__(self,
				 env,
				 actor_critic_class,
				 n_steps=512,
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
				 cliprange_vf=None
				 ):
		self.envs=env
		self.test_env=env
		self.ac_network=actor_critic_class()

		self.n_steps=n_steps
		self.epochs=epochs
		self.steps_per_epoch=steps_per_epoch
		self.shuffle_buffer_size=shuffle_buffer_size
		self.mem=Memory(self.envs.num_uav,self.n_steps,self.epochs,self.steps_per_epoch,self.shuffle_buffer_size)

		self.gamma=gamma
		self.lam=lam
		self.vf_coef=vf_coef
		self.ent_coef=ent_coef
		self.learning_rate=learning_rate
		self.max_grad_norm=max_grad_norm
		self.cliprange=cliprange
		self.cliprange_vf=cliprange_vf

		self.optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
		self.logger=TensorboardLogger(loc="./logs/", experiment="PPO")

	def policy_loss(self,new_log_probs,old_log_probs,adv):
		r=tf.math.exp(tf.math.subtract(new_log_probs, old_log_probs))
		surr1=tf.multiply(r, adv)
		surr2=tf.multiply(tf.clip_by_value(r, 1-self.cliprange, 1+self.cliprange), adv)
		loss= -1*tf.reduce_mean(tf.reduce_min([surr1,surr2],axis=0))
		return loss

	def value_loss(self,vpred,old_values,returns):
		if self.cliprange_vf==None:
			return 0.5*tf.reduce_mean(tf.math.squared_difference(returns, vpred))
		else:
			vpred_clipped=old_values+tf.clip_by_value(vpred-old_values,-self.cliprange_vf, self.cliprange_vf)
			vf_losses1=tf.math.squared_difference(returns,vpred)
			vf_losses2=tf.math.squared_difference(returns,vpred_clipped)
			return 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

	def entropy_loss(self,entropy):
		return -1*tf.reduce_mean(entropy)


	def _train(self,img_states,ser_states,actions,old_log_probs,adv,old_values,returns):
		with tf.GradientTape() as t:
			new_log_probs,vpred,entropy=self.ac_network.log_prob_value_entropy((img_states,ser_states),actions)
			pi_loss=self.policy_loss(new_log_probs,old_log_probs,adv)
			v_loss=self.vf_coef*self.value_loss(vpred,old_values,returns)
			ent_loss=self.ent_coef*self.entropy_loss(entropy)
			total_loss=pi_loss+v_loss+ent_loss

		grads=t.gradient(total_loss,self.ac_network.trainable_variables)
		if self.max_grad_norm!=None:
			grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
		self.optimizer.apply_gradients(zip(grads,self.ac_network.trainable_variables))
		return pi_loss.numpy(),v_loss.numpy(),ent_loss.numpy()

	def _test(self,num=1,render=False):
		cum_rewd=0
		for _ in range(num):
			done=False
			reward=0
			curr_state=self.test_env.reset()
			while not done:
				if render:self.test_env.render()
				act=self.ac_network.learned_action(curr_state)
				next_state,rewd,notdone,(done,info)=self.test_env.step(act)
				if not done:curr_state=self.test_env.curr_state
				reward+=rewd.sum()
			cum_rewd+=reward
		avg_rewd=cum_rewd/num
		return avg_rewd


	def learn(self,iterations=5000,max_reward=None,test_at_iter=2,num_of_test=10,render_at_test=False):
		for iteration in range(iterations):
			print("On Iteration:",iteration)
			log_probs = []
			values = []
			img_states = []
			ser_states=[]
			actions = []
			rewards = []
			masks = []

			curr_states=self.envs.reset()
			for _ in range(self.n_steps):
				act,log_prob,val=self.ac_network.action_log_prob_value(curr_states)
				next_states,rewds,notdones,(done,infos)=self.envs.step(act)

				log_probs.append(log_prob)
				values.append(val.ravel())
				img_states.append(curr_states[0])
				ser_states.append(curr_states[1])
				actions.append(act)
				rewards.append(rewds)
				masks.append(notdones)

				if done:
					curr_states=self.envs.reset()
				else:
					curr_states=self.envs.curr_state

			next_value=self.ac_network.value(next_states)
			returns=compute_gae(next_value.ravel(), rewards, masks, values,self.gamma,self.lam)

			returns      = functools_reduce_iconcat(returns).reshape(-1,1)
			log_probs    = functools_reduce_iconcat(log_probs)
			values       = functools_reduce_iconcat(values).reshape(-1,1)
			img_states   = functools_reduce_iconcat(img_states)
			ser_states   = functools_reduce_iconcat(ser_states)
			actions      = functools_reduce_iconcat(actions)
			advantage    = returns - values
			advantage    = normalize(advantage)

			self.mem.initilize((img_states,ser_states,actions,log_probs,advantage,values,returns))
			pi_loss,v_loss,ent_loss=0,0,0
			for batch in self.mem.dataset:
				print("Learning from replayBuffer")
				pi_l,v_l,ent_l=self._train(*batch)
				pi_loss+=pi_l;v_loss+=v_l;ent_loss+=ent_l
			pi_loss/=self.mem.num_of_batch
			v_loss/=self.mem.num_of_batch
			ent_loss/=self.mem.num_of_batch
			total_loss=pi_loss+v_loss+ent_loss
			metrics={"pi_loss":pi_loss,"v_loss":v_loss,"ent_loss":ent_loss,"total_loss":total_loss}
			#print("metrics are:",metrics)
			self.logger.log(iteration,metrics)
			
			if iteration%test_at_iter==0:
				score=self._test(num_of_test,render_at_test)
				self.logger.log(iteration,{"Test Score":score})
				print('On iteration {} score {}'.format(iteration,score))
				if max_reward!=None and score>=max_reward:
					print(f"----- max reward {max_reward} reached-----")
					break
