import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import h5py
import os
import sys

cd = os.getcwd()


env = gym.make('CartPole-v1')

# Neural network parameters
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.001
n_hidden = 20

# Environment parameters
n_of_episodes = 10
n_of_steps = 1000
steps_executed = 0
starting_epsilon = 1.0
ending_epsilon = 0.01
decay_rate = 0.0001
memory = []
memory_size = 2000
batch_size = 32
discount_factor = 1



def neural_network_model(input_size,output_size,learning_rate):

	model = Sequential()
	model.add(Dense(n_hidden,input_dim=input_size,activation='relu'))
	model.add(Dense(n_hidden,activation='relu'))
	model.add(Dense(output_size,activation='linear'))

	model.compile(loss='mse',optimizer=Adam(lr=learning_rate))

	return model

def save_model(model,fname):

	model.save_weights(fname)

def load_model(model,fname):

	model.load_weights(fname)
	return model


main_dqn = neural_network_model(input_size,output_size,learning_rate)
target_dqn = neural_network_model(input_size,output_size,learning_rate)

if os.path.exists(cd+"/model/cartpole-900.h5"):
	print("Weights loaded successfully !")
	main_dqn = load_model(main_dqn,"model/cartpole-900.h5")
target_dqn.set_weights(main_dqn.get_weights())

scores = []

if sys.argv[1] == "train":

	for episode in range(n_of_episodes):
		current_state = env.reset()
		episode_score = 0
		for steps in range(n_of_steps):
			steps_executed += 1
			exploration_prob = ending_epsilon + (starting_epsilon - ending_epsilon) * np.exp(-decay_rate * steps_executed)
			if exploration_prob >= np.random.rand():
				action = env.action_space.sample()
			else:
				action = np.argmax(main_dqn.predict(np.array([current_state]))[0])
			next_state,reward,done,info = env.step(action)
			if len(memory) >= memory_size:
				del memory[0]
			if done:
				memory.append((current_state,action,next_state,-1,done))
			else:
				memory.append((current_state,action,next_state,reward,done))
			current_state = next_state
			if done:
				target_dqn.set_weights(main_dqn.get_weights())
				break
			if len(memory) >= batch_size:
				batch = random.sample(memory,batch_size)
				x_train = []
				y_train = []
				for c_s,a,n_s,r,d in batch:
					y = main_dqn.predict(np.array([c_s]))[0]
					if d:
						y[a] = r
					else:
						y_target = target_dqn.predict(np.array([n_s]))[0]
						y_bellman = r + discount_factor * np.max(y_target)
						y[a] = y_bellman
					x_train.append(c_s)
					y_train.append(y)
				main_dqn.fit(np.array(x_train),np.array(y_train),epochs=1,verbose=0)
			episode_score += reward
		scores.append(episode_score)
		if episode > 0 and episode % 100 == 0:
			save_model(main_dqn,"model/cartpole-"+str(episode)+".h5")
		print ("Episode {}/{} completed, episode score {}, exploration_prob {:2f}".format(episode,n_of_episodes,episode_score,exploration_prob))

elif sys.argv[1] == "test":

	for episode in range(n_of_episodes):
		current_state = env.reset()
		episode_score = 0 
		for steps in range(n_of_steps):
			env.render()
			action = np.argmax(main_dqn.predict(np.array([current_state]))[0])
			print("Action: ",action)
			next_state,reward,done,info = env.step(action)
			if done:
				break
			current_state = next_state
			episode_score += reward
		scores.append(episode_score)
		print("Episode {}/{} completed, episode_score {}".format(episode,n_of_episodes,episode_score))

print("Mean: ",np.mean(scores))
print("Max:" ,np.max(scores))
print("Min:",np.min(scores))


env.close()





