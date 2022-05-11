import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
import torch.nn.functional as F
from agent_dir.agent import Agent
from collections import namedtuple
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        # ##################
        # # YOUR CODE HERE #
        # """
        # Initialize Deep Q Network
        # Args:
        #     in_channels (int): number of input channels
        #     n_actions (int): number of outputs
        # """
        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(hidden_size*2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(hidden_size*2)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, output_size)
        ##################
        # pass

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        x = inputs[np.newaxis, :]
        x = torch.Tensor(x)
        x = x.float() / 255
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.fc5(x)
        ##################
        # pass


class ReplayBuffer:
    def __init__(self, buffer_size):
        ##################
        # YOUR CODE HERE #
        self.buffer_size = buffer_size
        self.buffer = []
        ##################
        # pass

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        return len(self.buffer)
        ##################
        # pass

    def push(self, *transition):
        ##################
        # YOUR CODE HERE #
        if len(self.buffer) == self.buffer_size: #buffer full
            self.buffer.pop(0)
        self.buffer.append(transition)
        ##################
        # pass

    def sample(self, batch_size):
        ##################
        # YOUR CODE HERE #
        index = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in index]
        return zip(*batch)
        ##################
        # pass

    def clean(self):
        ##################
        # YOUR CODE HERE #
        self.buffer.clear()
        ##################
        # pass


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        self.env = env
        self.h = self.env.observation_space.shape[0]#observation_dim,input size
        self.w = self.env.observation_space.shape[1]
        self.c = self.env.observation_space.shape[2]
        self.action_dim = self.env.action_space.n#output_size
        self.action_space = []
        self.hidden_size = args.hidden_size
        self.seed = args.seed
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.grad_norm_clip = args.grad_norm_clip
        self.max_episode = args.max_episode
        self.eps = args.eps
        self.eps_min = args.eps_min
        self.eps_decay = args.eps_decay
        self.test = args.test
        self.use_cuda = args.use_cuda
        self.n_frames = args.n_frames
        self.learning_freq = args.learning_freq
        self.target_update_freq = args.target_update_freq
        self.buffer_size = args.buffer_size
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        #构建网络
        self.eval_dqn = QNetwork(self.h, self.hidden_size, self.action_dim).cuda()
        self.target_dqn = QNetwork(self.h, self.hidden_size, self.action_dim).cuda()
        self.optim = optim.Adam(self.eval_dqn.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0
        
        ##################
        # pass
    
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        # if len(self.replay_buffer) < self.replay_buffer.buffer_size:
        #     loss = 0
        #     return loss
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
        else:
            self.eps = self.eps_min
        
        if self.learn_step % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.eval_dqn.state_dict())
        self.learn_step += 1

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)
        actions = torch.LongTensor(actions)
        dones = torch.IntTensor(dones)
        rewards = torch.FloatTensor(rewards)

        q_eval = self.eval_dqn(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_next = self.target_dqn(next_obs).detach()
        q_target = rewards + self.gamma * (1-dones) * torch.max(q_next, dim = -1)[0]
        Loss = self.loss_fn(q_eval, q_target)
        self.optim.zero_grad()
        Loss.backward()
        self.optim.step()
        return Loss
        ##################
        # pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        ##################
        # YOUR CODE HERE #
        if np.random.uniform() <= self.eps:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action_value = self.eval_dqn(observation)
            action = torch.max(action_value, dim = -1)[1].numpy()
        return int(action)
        ##################
        # pass

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        # step = 0 #记录现在走到第几步了
        for i_episode in range(self.max_episode):
            obs = self.env.reset() #获得初始观测值
            episode_reward = 0
            done = False
            while not done:
                action = self.make_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                self.store_transition = (obs, action, reward, next_obs) #存储记忆
                self.replay_buffer.push(self.store_transition)
                # if agent.buffer.__len__() >= args.buffer_size:
                # if (step > 200) and (step % 5 == 0): #当走了200次之后再每走5次学习一次
                #     loss = self.train()
                episode_reward += reward
                obs = next_obs
                if self.replay_buffer.__len__() >= self.buffer_size:
                    loss = self.train()
                if done:
                    break
                # step += 1
            # print(i_episode, "reward:", episode_reward, "loss:", loss)
            print(i_episode, "reward:", episode_reward)
        ##################
        # pass
