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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 定义一个元组表征经验存储的格式
# Transition = namedtuple('Transion', 
#                         ('state', 'action', 'next_state', 'reward'))

                        ## 超参数
# epsilon = 0.9
# BATCH_SIZE = 32
# GAMMA = 0.99
# EPS_START = 1
# EPS_END = 0.02
# EPS_DECAY = 1000000
# TARGET_UPDATE = 1000
# RENDER = False
# # lr = 1e-3
# INITIAL_MEMORY = 10000
# MEMORY_SIZE = 10 * INITIAL_MEMORY
# n_episode = 2000
MIN_MEMPORY_LEN = 100

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
        # #super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(hidden_size*2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(hidden_size*2)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, output_size)
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, output_size)


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
        self.memory = []
        # self.next_idx = 0
        ##################
        # pass

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        return len(self.memory)
        ##################
        # pass

    def push(self, *transition):
        ##################
        # YOUR CODE HERE #
        if len(self.memory) == self.buffer_size: #buffer not full
            self.memory.pop(0)
        # self.memory[self.next_idx] = Transition(*transition)
        # self.next_idx = (self.next_idx + 1) % self.buffer_size #移动指针，经验池满了之后从最开始的位置开始将最近的经验存进经验池
        self.memory.append(transition)

        ##################
        # pass

    def sample(self, batch_size):
        ##################
        # YOUR CODE HERE #
        # if len(self.memory) > batch_size:
        #     index = np.random.choice(len(self.memory),batch_size)
        # else:
        #     index = len(self.memory)
        # batch = [self.memory[i] for i in index]
        return random.sample(self.memory, batch_size)# 从经验池中随机采样
        # return zip(*batch)
        ##################
        # pass

    def clean(self):
        ##################
        # YOUR CODE HERE #
        self.memory.clear()
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
        self.h = self.env.observation_space.shape[0]#observation_dim
        self.w = self.env.observation_space.shape[1]
        self.c = self.env.observation_space.shape[2]
        self.action_dim = self.env.action_space.n
        self.action_space = []
        self.hidden_size = args.hidden_size
        self.seed = args.seed
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.grad_norm_clip = args.grad_norm_clip
        self.max_episodes = args.max_episode
        self.eps = args.eps
        self.eps_min = args.eps_min
        self.eps_decay = args.eps_decay
        self.update_target = args.update_target
        self.test = args.test
        self.use_cuda = args.use_cuda
        self.n_frames = args.n_frames
        self.learning_freq = args.learning_freq
        self.target_update_freq = args.target_update_freq
        self.buffer_size = args.buffer_size
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        #构建网络
        self.eval_dqn = QNetwork(input_size, self.hidden_size, output_size).cuda()
        self.target_dqn = QNetwork(input_size, self.hidden_size, output_size).cuda()
        self.optim = optim.Adam(self.eval_net.parameters(), lr=self.lr)
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

    def train(self, args):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        if len(self.replay_buffer) < self.replay_buffer.buffer_size:
            loss = 0
            return loss
        if self.eps > args.eps_min:
            self.eps *= args.eps_decay
        else:
            self.eps = args.eps_min
        
        if self.learn_step % args.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        obs, actions, rewards, next_obs, dones = self.buffer.sample(args.batch_size)
        actions = torch.LongTensor(actions)
        dones = torch.IntTensor(dones)
        rewards = torch.FloatTensor(rewards)

        q_eval = self.eval_net(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_next = self.target_net(next_obs).detach()
        q_target = rewards + args.gamma * (1-dones) * torch.max(q_next, dim = -1)[0]
        loss = self.loss_fn(q_eval, q_target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss
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
        # self.stepdone += 1
        # observation = observation.to(device)
        # epsilon = 0.99
        # # epsilon = EPS_END + (EPS_START - EPS_END)* \
        # #     math.exp(-1. * self.stepdone / EPS_DECAY)            # 随机选择动作系数epsilon 衰减，也可以使用固定的epsilon
        # # epsilon-greedy策略选择动作
        # if random.random()<epsilon:
        #     action = torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)
        # else:
        #     action = self.DQN(observation).detach().max(1)[1].view(1,1)  # 选择Q值最大的动作并view
            
        # return action  
        if np.random.uniform() <= self.eps:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action_value = self.eval_dqn(observation)
            action = torch.max(action_value, dim = -1)[1].cpu().numpy()
        return int(action)

        ##################
        # pass

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        step = 0 #记录现在走到第几步了
        input_size = self.env.observation_space.shape[0]
        output_size = self.env.action_space.n
        for i_episode in range(self.max_episodes):
            obs = self.env.reset() #获得初始观测值
            episode_reward = 0
            done = False
            while not done:
                action = self.make_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                self.store_transition = (obs, action, reward, next_obs) #存储记忆
                self.replay_buffer.push(self.store_transition)
                # if agent.buffer.__len__() >= args.buffer_size:
                if (step > 200) and (step % 5== 0): #当走了200次之后再每走5次学习一次
                    loss = self.train()
                
                episode_reward += reward
                obs = next_obs

                if done:
                    break

                step += 1
            print(i_episode, "reward:", episode_reward, "loss:", loss)

        ##################
        # pass
