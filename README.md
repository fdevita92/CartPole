# CartPole
Deep Reinforcement Learning applied on AI gym Cartpole environment

# Training
The DQN is trained for 1000 episodes, but you can edit this parameter by changing the variable n_of_episodes. Every 100 episodes the script will save the network weights inside the model directory.To launch the script in train mode just type on the command line: python3 dqn_cartpole.py train

# Testing
To test the system just type: python3 dqn_cartpole.py test (I suggest you to load the best weights provided by the script during train mode for best performances). Note: the environment will render only if the number of testing episodes will be less or equal 20 (you can change this setting if you want)

# Best Model
The script can be run in a third mode called best. After training, by running this mode the script will search for the best model among the files saved inside the model directory (this may take a lot of time, so be patient). It will test every model for 10 episodes and it will return the best one with the highest score (on average). 

# Main Dependencies
1. Tensorflow
2. Keras
3. gym
4. h5py


For any question or suggestion feel free to contact me.
