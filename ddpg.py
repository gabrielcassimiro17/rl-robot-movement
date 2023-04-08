import random
from collections import deque
import torch
import torch.nn as nn
import numpy as np

from actor_critic import Actor, Critic

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim, buffer_size, batch_size, actor_lr, critic_lr, tau, gamma):
        self.actor = Actor(state_dim, hidden_dim, action_dim, actor_lr)
        self.actor_target = Actor(state_dim, hidden_dim, action_dim, actor_lr)
        self.critic = Critic(state_dim, action_dim, hidden_dim, critic_lr)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim, critic_lr)

        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

        self._update_target_networks(tau=1)  # initialize target networks

    def act(self, state, noise=0.0):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        return np.clip(action + noise, -1, 1)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Update Critic
        self.critic.optimizer.zero_grad()

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * target_q_values

        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        critic_loss.backward()
        self.critic.optimizer.step()

        # Update Actor
        self.actor.optimizer.zero_grad()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        self._update_target_networks()

    def _update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
