import tensorflow as tf

class Memory:
	def __init__(self,num_uav=10,n_steps=512,epochs=10,steps_per_epoch=32,shuffle_buffer_size=1024):
		self.num_uav=num_uav
		self.n_steps=n_steps
		self.epochs=epochs
		self.steps_per_epoch=steps_per_epoch
		self.num_of_batch=self.steps_per_epoch*self.epochs
		self.shuffle_buffer_size=shuffle_buffer_size
		self.batch_size=(self.num_uav*self.n_steps)//self.steps_per_epoch

	def initilize(self,tensors):
		self.dataset=tf.data.Dataset.from_tensor_slices(tensors).\
									 shuffle(self.shuffle_buffer_size).\
									 repeat(self.epochs).\
									 batch(self.batch_size,drop_remainder=True).\
									 prefetch(tf.data.experimental.AUTOTUNE)