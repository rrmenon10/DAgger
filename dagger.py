import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import load_policy
import gym
import argparse
import pickle


class Imitator():


	def __init__(self, obj_dims, action_dims):

		with tf.variable_scope("DAgger"):
			self.inputs = tf.placeholder(tf.float32, shape=[None, obj_dims], name= "State")
			self.actions = tf.placeholder(tf.float32, shape=[None, action_dims], name="Actions")
			net = layers.fully_connected(self.inputs, num_outputs = 128, activation_fn=tf.nn.relu)
			net = layers.fully_connected(net, num_outputs = 64, activation_fn=tf.nn.relu)
			net = layers.fully_connected(net, num_outputs = 32, activation_fn=tf.nn.relu)
			self.pred_actions = layers.fully_connected(net, num_outputs = action_dims, activation_fn=None)
			self._loss = tf.reduce_mean(tf.square(self.pred_actions - self.actions))
			self._train = tf.train.AdamOptimizer().minimize(self._loss)


	def train(self, sess, obs, acts):

		sess.run(self._train, feed_dict={self.inputs:obs, self.actions:acts})

	def predict(self, sess, obs):

		return sess.run(self.pred_actions, feed_dict={self.inputs:obs})

	def loss(self, sess, obs, acts):

		return sess.run(self._loss, feed_dict={self.inputs:obs, self.actions:acts})



def train_dagger(policy_fn, dataset, env, minibatch_size, num_dagger, num_rollouts, obj_dims, action_dims):


	observations, actions = dataset
	obs_mean = np.mean(observations, axis=0)
	obs_std = np.std(observations, axis=0)
	env = gym.make(env)
	max_steps = env.spec.timestep_limit

	with tf.Session() as sess:
		network = Imitator(obj_dims, action_dims)
		sess.run(tf.global_variables_initializer())

		for i in range(num_dagger):
			print('Training iteration %d'%(i+1))
			

			#Training with data
			idx = np.arange(observations.shape[0])
			np.random.shuffle(idx)
			for j in range(observations.shape[0]//minibatch_size):
				o_train, a_train = observations[idx[j*minibatch_size: (j+1)*minibatch_size]], actions[idx[j*minibatch_size: (j+1)*minibatch_size]]
				network.train(sess, o_train, a_train)

			#Collecting new data
			observations_new = []
			actions_new = []
			returns = []
			for _ in range(num_rollouts):
				obs = env.reset()
				env.render()
				done = False
				totalr = 0
				steps = 0

				while not done:
					obs = obs.reshape([1, -1])
					action = network.predict(sess, obs)
					action_expert = policy_fn(obs)
					actions_new.append(action_expert)
					observations_new.append(obs)
					obs, r, done, _ = env.step(action)
					totalr += r
					steps += 1
					env.render()
					if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
					if steps >= max_steps:
						break
				returns.append(totalr)

			#Aggregating dataset
			observations_new, actions_new = np.array(observations_new), np.array(actions_new)
			observations_new = observations_new.reshape([-1, obj_dims])
			actions_new = actions_new.reshape([-1, action_dims])
			observations = np.concatenate((observations, observations_new), axis=0)
			actions = np.concatenate((actions, actions_new), axis=0)

			print("Mean Returns", np.mean(returns))
			print("Std of Returns", np.std(returns))


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type = str, default = 'Humanoid-v2')
	parser.add_argument('--env_expert', type = str, default = 'experts/Humanoid-v2.pkl')
	parser.add_argument('--num_rollouts', type= int, default = 25, help = 'Number of rollouts of network for each data aggregation step')
	parser.add_argument('--num_dagger', type = int, default = 50, help = 'Number of DAgger iterations')
	parser.add_argument('--minibatch_size', type = int, default = 25, help = 'Minibatch size per DAgger iteration')
	args = parser.parse_args()

	print('Loading Dataset')
	expert = pickle.load(open(str(args.env)+'.pkl', 'rb'))
	obj_dims, action_dims = expert['observations'].shape[-1], expert['actions'].shape[-1]
	dataset = expert['observations'], expert['actions'].reshape([expert['observations'].shape[0], -1])
	print('Completed Loading Dataset')
	
	policy_fn = load_policy.load_policy(args.env_expert)

	train_dagger(policy_fn, dataset, args.env, args.minibatch_size, args.num_dagger, args.num_rollouts, obj_dims, action_dims)


if __name__=='__main__':
	main()
