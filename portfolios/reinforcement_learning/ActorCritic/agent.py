import numpy as np
import keras
import tensorflow as tf
# from tensorflow_probability.distributions import Beta
from keras.models import Sequential
from keras.layers import Dense, Lambda
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import random




def AddValue(output_size, value):
    return Lambda(lambda x: x + value, output_shape=(output_size,))



from enum import Enum, auto


class BaselineMethod(Enum):
  SIMPLE = auto()
  ADAPTIVE = auto()
  


class AgentActorCritic(object):
    
    def __init__(self, n_obs, action_space, policy_learning_rate, value_learning_rate, 
                 discount, baseline = None, entropy_cost = 0, max_ent_cost = 0, num_layers=3, **kwargs):
      

        #We need the state and action dimensions to build the network
        self.n_obs = n_obs  
        self.n_act = action_space
        
        self.plr = policy_learning_rate
        self.vlr = value_learning_rate
        self.gamma = discount
        self.entropy_cost = entropy_cost
        self.max_ent_cost = max_ent_cost
        self.num_layers = num_layers

        #parameter that indicates if the simple baseline should be used or not
        self.use_simple_baseline = baseline == BaselineMethod.SIMPLE

        #parameter that indicates if the adaptive baseline should be used or not
        self.use_adaptive_baseline = baseline == BaselineMethod.ADAPTIVE
        

        #These lists stores the cumulative observations for this episode
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        #Build the keras network
        self._build_network()

        #get hyperparameters
        self.steps_done = 0
        self.skip_trades = kwargs.get("skip_trades", 0)
        self.epsilon_start = kwargs.get("epsilon_start", 0.9)
        self.epsilon_end = kwargs.get("epsilon_end", 0.05)
        self.epsilon_decay = kwargs.get("epsilon_decay", 2000)

    def observe(self, state, action, reward):
        """ This function takes the observations the agent received from the environment and stores them
            in the lists above."""
        self.episode_actions.append(action)
        self.episode_observations.append(state)
        self.episode_rewards.append(reward)
        
        
    def decide(self, state):
        """ This function feeds the observed state to the network, which returns a distribution
            over possible actions. Sample an action from the distribution and return it."""
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.steps_done / self.epsilon_decay)
        
        
        if sample > eps_threshold:
            return self.policy_network(state[np.newaxis, ...])
        else:
            rnd = np.random.rand(1, self.n_act)
            return rnd/np.sum(rnd)

        #return np.random.choice(self.n_act, p=actions_probs)

    def train(self):
        """ When this function is called, the accumulated episode observations, actions and discounted rewards
            should be fed into the network and used for training. Use the _get_returns function to first turn 
            the episode rewards into discounted returns. 
            Apply simple or adaptive baselines if needed, depending on parameters."""

        self.steps_done += 1
        discounted_returns = self._get_returns()
        X= np.array(self.episode_observations)
        y= np.concatenate(self.episode_actions)

        if (self.use_adaptive_baseline):
               self.value_network.train_on_batch( X, discounted_returns )

               adaptive_baseline= np.array(self.value_network(X))
               
               
               discounted_returns -= adaptive_baseline

        self.policy_network.train_on_batch( X, y, sample_weight=discounted_returns)
        
       

        #reinitialize the values 
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

    def _get_returns(self):
        """ This function should process self.episode_rewards and return the discounted episode returns
            at each step in the episode."""
        returns = np.empty_like(self.episode_rewards)
        dis = 0
        moving_baseline_count= 0
        moving_baseline_sum=0

        for i in reversed(range(len(self.episode_rewards))): 
           dis = self.episode_rewards[i] + dis * self.gamma
           returns[i] = dis

           if (self.use_simple_baseline):
             moving_baseline_sum+=dis
             moving_baseline_count+=1 
             returns[i] -= moving_baseline_sum / moving_baseline_count

        return returns 


    def _build_network(self):
        """ This function should build the network that can then be called by decide and train. 
            The network takes observations as inputs and has a policy distribution as output."""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(256, input_shape=(self.n_obs ,), activation='selu'))
        model.add(tf.keras.layers.Dense(64, activation='selu'))
        #model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(self.n_act, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=self.plr))
        
        self.policy_network = model 

        if (self.use_adaptive_baseline):
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Dense(256, input_shape=(self.n_obs ,), activation='selu'))
                model.add(tf.keras.layers.Dense(64, activation='selu'))
                #model.add(tf.keras.layers.Dense(16, activation='relu'))
                model.add(tf.keras.layers.Dense(1))
                model.compile(loss='mse',
                              optimizer=Adam(learning_rate=self.vlr))
                
                self.value_network = model 