import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,GlobalAveragePooling2D,LSTM,Reshape,Flatten,Concatenate
import tensorflow_probability as tfp

class Mlp_AC:
	def __init__(self,trainable=True):
		self.trainable=trainable
		self.nn=self.setup_nn(self.trainable)
		self.logits=self.actor_network_head()
		self.val=self.critic_network_head()

	@property
	def trainable_variables(self):
		return self.nn.trainable_variables+self.logits.trainable_variables+self.val.trainable_variables

	def setup_nn(self,trainable):
		nn=self.neuralNetArch()
		nn.trainable=trainable
		return nn

	def Convolutional_network(self):
		cnn=Sequential([
			Conv2D(32, 5, strides=(1, 1), input_shape=(80,80,2), padding='same',activation="relu"),
			MaxPool2D(3, strides=3, padding='same'), #N,27,27,32
			Conv2D(64, 5, strides=(1, 1), padding='same',activation="relu"),
			MaxPool2D(3, strides=3, padding='same'), #N,9,9,64
			Conv2D(256, 3, strides=(1, 1), padding='same',activation="relu"),
			MaxPool2D(2, strides=2, padding='same'), #N,5,5,256
			GlobalAveragePooling2D() #N,256
		],name="cnn")
		return cnn

	def Recurrent_network(self):
		rnn=Sequential([
			LSTM(64, activation="relu", return_sequences=True, input_shape=(None, 9)), #(N,None,64)
			LSTM(64, activation="relu", return_sequences=False) #(N,64)
		],name="rnn")
		return rnn

	def Connector(self):
		return Concatenate()

	def Common_network(self):
		common=Sequential([
			Dense(500, activation="relu", input_shape=(320,)), #N,500
			Reshape((5,5,-1)), #Nx5x5x20
			Conv2D(32, 3, strides=(1, 1), padding='same',activation="relu"), #Nx5x5x32
		],name="common")
		return common
	
	def neuralNetArch(self):
		cnn=self.Convolutional_network()
		rnn=self.Recurrent_network()
		connect=self.Connector()
		common=self.Common_network()
		out=common(connect([cnn.output,rnn.output]))
		nn=Model(inputs=[cnn.input,rnn.input],outputs=[out],name="nn")
		return nn

	def actor_network_head(self):
		actor_hd=Sequential([
			Conv2D(5, 3, strides=(1, 1), padding='same'), #Nx5x5x5
			Flatten() #N,125
		],name="actor_hd")
		return actor_hd

	def critic_network_head(self):
		critic_hd=Sequential([
			Flatten(), #N,800
			Dense(units=1,activation=None)
		],name="critic_hd")
		return critic_hd

	def __call__(self,s):
		net=self.nn(s)
		return net

	def actor_head(self,net):
		logits=self.logits(net)
		dist=tfp.distributions.Categorical(logits=logits)
		return dist

	def critic_head(self,net):
		val=self.val(net)
		return val

	def action_log_prob_value(self,s):
		net=self.__call__(s)
		dist=self.actor_head(net)
		val=self.critic_head(net)
		a=dist.sample().numpy()
		#a,log_prob(s,a),v(s) --nograd
		return a,dist.log_prob(a).numpy().reshape(-1,1),val.numpy()

	def log_prob_value_entropy(self,s,a):
		net=self.__call__(s)
		dist=self.actor_head(net)
		val=self.critic_head(net)
		#log_prob(s,a),v(s),entropy(s) --withgrad
		return tf.reshape(dist.log_prob(a),shape=(-1,1)),val,dist.entropy()

	def value(self,s):
		net=self.__call__(s)
		val=self.critic_head(net)
		return val.numpy()

	def learned_action(self,s):
		net=self.__call__(s)
		logits=self.logits(net)
		return tf.math.argmax(logits,axis=1).numpy()

	def summary(self):
		print("common net")
		self.nn.summary()
		print("actor-head")
		self.logits.summary()
		print("critic-head")
		self.val.summary()