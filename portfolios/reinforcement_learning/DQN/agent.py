
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import Transition









class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




class DQN(nn.Module):

    def __init__(self, cfg, input_dim, output_dim, *args):
        super(DQN, self).__init__()

        self.cfg = cfg
        self.layer_sequence = list(zip((input_dim,) + args, args + (output_dim,)))
        self.layers = nn.ModuleList([nn.Linear(*layer) for layer in self.layer_sequence])
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(layer) for layer in args])


    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path, cpu=False):
        if cpu:
            self.load_state_dict(torch.load(path,
                map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))


    def forward(self, x):
        x = x.to(self.cfg.device).float()
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x) 
        #x /= torch.linalg.norm(x, ord=1)  
        return x   




class AgentDQN: 
    


    def __init__ (self, cfg, env, *args, **kwargs):

        self.env = env
        self.cfg = cfg
        self.policy_net = DQN(cfg, self.env.state_dim, self.env.action_dim, *args).to(self.cfg.device)
        self.target_net = DQN(cfg, self.env.state_dim, self.env.action_dim, *args).to(self.cfg.device)
        print(self.policy_net)
        for name, param in self.policy_net.named_parameters():
            print(name, param)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []
        self.num_episodes = kwargs.get("num_epsiodes", 100)
        self.target_update = kwargs.get("target_update", 5)
        self.skip_trades = kwargs.get("skip_trades", 0)
        self.epsilon_start = kwargs.get("epsilon_start", 0.9)
        self.epsilon_end = kwargs.get("epsilon_end", 0.05)
        self.epsilon_decay = kwargs.get("epsilon_decay", 200)
        self.batch_size = kwargs.get("batch_size", 64)
        self.gamma = kwargs.get("gamma", 0.99)
        #self.display = Display(visible=0, size=(1400, 900))
        #self.display.start()



    def optimize_model(self):

        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
       

        batch = Transition(*zip(*transitions))

       
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.cfg.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.cfg.BATCH_SIZE, device=self.cfg.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

       
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()






    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state)
        else:
            return torch.tensor(np.random.rand(1, self.env.action_dim), device=self.cfg.device, dtype=torch.float)






    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        
        #self.display.clear_output(wait=True)
        #self.display.display(plt.gcf())





    def train(self):
    
        for i_episode in range(self.num_episodes):
            # Initialize the environment and state
            for t, state in self.env.__iter__():
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done,  = self.env.step(t, action)
                reward = torch.tensor([reward], device=self.cfg.device)


                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)


                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break
                # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

