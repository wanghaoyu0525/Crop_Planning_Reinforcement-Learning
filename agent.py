import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from common.memory import ReplayBuffer
from common.model import MLP


class DQN:
    def __init__(self, state_dim, action_dim, cfg):

        self.action_dim = action_dim  # 总的动作个数
        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma  # 奖励的折扣因子
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
            (cfg.epsilon_start - cfg.epsilon_end) * \
            math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.cfg = cfg
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) # 优化器
        self.memory = ReplayBuffer(cfg.memory_capacity)

    def choose_action(self, state, is_current_done):
        #选择动作
        self.frame_idx += 1
        if not is_current_done:
            #print('nothing')
            return 0
        if random.random() > self.epsilon(self.frame_idx):
        # if random.random() > 0.1: # greedy
            with torch.no_grad():
                #把state变成tensor使之能够输入到神经网络
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item() # 选择Q值最大的动作

        else:
            action = random.randrange(self.action_dim)
        return action

    def predict(self, state, is_current_done):
        if not is_current_done:
            return 0
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()
        return action

    def update(self):
        if len(self.memory) < self.batch_size: # 当memory中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # 转为张量
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(
            1)
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(
            done_batch), device=self.device)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch) # 计算当前状态(s_t,a)对应的Q(s_t, a)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0] # 计算下一时刻的状态(s_t_,a)对应的Q值
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失
        # 优化更新模型
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
            # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
