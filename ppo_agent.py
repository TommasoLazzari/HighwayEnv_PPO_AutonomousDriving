"""
PPO components for the HighwayEnv autonomous driving project.

This file only defines:
- Actor-Critic neural network
- Trajectory buffer
- PPO agent
- PPO update logic

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes linear layers orthogonally.
    Standard standard deviations:
    - hidden layers: sqrt(2)
    - policy head: 0.01 (to encourage exploration)
    - value head: 1.0
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Separate neural networks for Actor (policy head) and Critic (value head).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, state):
        logits = self.actor(state)
        value = self.critic(state).squeeze(-1)

        return logits, value

    def get_action_and_value(self, state):
        """
        Samples an action from pi_theta(a | s).
        Returns:
        - sampled action
        - log probability of that action
        - value estimate V_phi(s)
        """

        logits, value = self.forward(state)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(self, states, actions):
        """
        Evaluates old sampled actions under the current policy.

        Used during the PPO update to compute:
        pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
        """

        logits, values = self.forward(states)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy


class PPOBuffer:
    """
    Stores one batch of trajectories D_k.

    The buffer stores transitions collected using the current policy pi_k.
    After each PPO update, the buffer is cleared.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO-Clip agent.

    Main methods:
    - select_action(state): used in training.py to interact with the environment
    - store_transition(...): saves one transition
    - update(): performs the PPO update
    - save(path), load(path): save/load model weights
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        policy_lr=3e-4,
        value_coef=0.5,
        entropy_coef=0.01,
        update_epochs=10,
        batch_size=64,
        max_grad_norm=0.5,
        device=None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=policy_lr, eps=1e-5)

        self.buffer = PPOBuffer()

    def select_action(self, state):
        """
        Selects an action using the current policy pi_theta.

        Returns the action as an integer so it can be passed directly to env.step(action).
        """

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value = self.model.get_action_and_value(state_tensor)

        return (action.item(),log_prob.item(),value.item())

    def select_deterministic_action(self, state):
        """
        Selects the action with the highest probability (argmax) under the current policy.
        """

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model.forward(state_tensor)
            action = torch.argmax(logits, dim=-1)

        return action.item()

    def store_transition(self, state, action, reward, done, log_prob, value):
        """
        Stores one transition in the current trajectory batch.
        """

        self.buffer.store(
            state=np.array(state, dtype=np.float32),
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value
        )

    def compute_rewards_to_go_and_advantages(self, next_value=0.0):
        """
        Computes:
        - rewards-to-go R_hat_t
        - advantage estimates A_hat_t

        Advantages are computed with GAE.
        This is still coherent with the pseudocode because it says
        advantage estimates can be computed using any method.
        """

        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        values = np.array(self.buffer.values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        rewards_to_go = np.zeros_like(rewards, dtype=np.float32)

        gae = 0.0
        next_val = float(next_value)
        next_ret = float(next_value)

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

            next_ret = rewards[t] + self.gamma * next_ret * mask
            rewards_to_go[t] = next_ret

            next_val = values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return rewards_to_go, advantages

    def update(self, next_value=0.0):
        """
        Performs the PPO update.

        This corresponds to:
        1. compute rewards-to-go
        2. compute advantages
        3. maximize PPO clipped objective
        4. fit value function by regression
        """

        if len(self.buffer) == 0:
            return

        rewards_to_go, advantages = self.compute_rewards_to_go_and_advantages(next_value)

        states = torch.tensor(np.array(self.buffer.states),dtype=torch.float32,device=self.device)

        actions = torch.tensor(self.buffer.actions,dtype=torch.long,device=self.device)

        old_log_probs = torch.tensor(self.buffer.log_probs,dtype=torch.float32,device=self.device)

        rewards_to_go = torch.tensor(rewards_to_go,dtype=torch.float32,device=self.device)

        advantages = torch.tensor(advantages,dtype=torch.float32,device=self.device)

        n_samples = states.size(0)

        for _ in range(self.update_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = rewards_to_go[batch_idx]
                batch_advantages = advantages[batch_idx]

                new_log_probs, values, entropy = self.model.evaluate_actions(batch_states,batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                unclipped_objective = ratio * batch_advantages
                clipped_ratio = torch.clamp(ratio,1.0 - self.clip_epsilon,1.0 + self.clip_epsilon)
                clipped_objective = clipped_ratio * batch_advantages

                policy_loss = -torch.mean(torch.min(unclipped_objective, clipped_objective))

                value_loss = torch.mean((values - batch_returns) ** 2)

                entropy_bonus = torch.mean(entropy)

                total_loss = (policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus)

                self.optimizer.zero_grad()
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(),self.max_grad_norm)

                self.optimizer.step()

        self.buffer.clear()

    def save(self, path):
        """
        Saves model parameters.
        """

        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Loads model parameters.
        """

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()