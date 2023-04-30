import copy
import pickle
import random
import gymnasium as gym
import torch
from collections import deque, namedtuple
from gymnasium.utils.save_video import save_video
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from utils import *


# Class for training an RL agent with Actor-Critic
class ACTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.agent = ACAgent(env=self.env, params=self.params)
        self.actor_net = ActorNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.critic_net = CriticNet(input_size=self.env.observation_space.shape[0], output_size=1, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.actor_optimizer = Adam(params=self.actor_net.parameters(), lr=self.params['actor_lr'])
        self.critic_optimizer = Adam(params=self.critic_net.parameters(), lr=self.params['critic_lr'])
        self.trajectory = None

    def run_training_loop(self):
        list_ro_reward = list()
        for ro_idx in range(self.params['n_rollout']):
            self.trajectory = self.agent.collect_trajectory(policy=self.actor_net)
            self.update_critic_net()
            self.estimate_advantage()
            self.update_actor_net()
            # TODO: Calculate avg reward for this rollout
            # HINT: Add all the rewards from each trajectory. There should be "ntr" trajectories within a single rollout.
            avg_ro_reward = ???
            print(f'End of rollout {ro_idx}: Average trajectory reward is {avg_ro_reward: 0.2f}')
            # Append average rollout reward into a list
            list_ro_reward.append(avg_ro_reward)
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ro_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def update_critic_net(self):
        for critic_iter_idx in range(self.params['n_critic_iter']):
            self.update_target_value()
            for critic_epoch_idx in range(self.params['n_critic_epoch']):
                critic_loss = self.estimate_critic_loss_function()
                critic_loss.backward()
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()

    def update_target_value(self, gamma=0.99):
        # TODO: Update target values
        # HINT: Use definition of target-estimate from equation 7 of teh assignment PDF

        obs_list =  self.trajectory['obs']
        reward_list = self.trajectory['reward']
        state_values = [self.critic_net.forward(obs) for obs in obs_list]
        next_state_value = 0.0
        target_values = []
        for i in reversed(range(len(obs_list))):
            reward = torch.tensor(reward_list[i]).unsqueeze(1)
            target_value = reward + gamma * state_values[i]
            target_values.insert(0, target_value)
        self.trajectory['state_value'] = state_values
        self.trajectory['target_value'] = target_values


    def estimate_advantage(self, gamma=0.99):
        # TODO: Estimate advantage
        # HINT: Use definition of advantage-estimate from equation 6 of teh assignment PDF

        advantages = []
        for state_value, target_value in  zip(self.trajectory['state_value'], self.trajectory['target_value']):
    # compute advantage for current element
            advantage = target_value - state_value
            advantages.append(advantage)

# concatenate list of tensors into a single tensor
        advantages = torch.cat(advantages, dim=0)

# assign computed advantages to trajectory dictionary
        self.trajectory['advantage'] = advantages

    def update_actor_net(self):
        actor_loss = self.estimate_actor_loss_function()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

    def estimate_critic_loss_function(self):
        # TODO: Compute critic loss function
        # HINT: Use definition of critic-loss from equation 7 of teh assignment PDF. It is the MSE between target-values and state-values.
           critic_losses = []

# iterate over each observation in the batch
        for state_value, target_value in zip(self.trajectory['state_value'], self.trajectory['target_value']):
    # compute the critic loss for this observation using mean squared error
            
            
            critic_loss = torch.nn.functional.mse_loss(state_value.float(), target_value.float())
    # append the critic loss to the list of losses
        critic_losses.append(critic_loss)

# compute the mean of the critic losses for the entire batch
        critic_loss = torch.mean(torch.stack(critic_losses))
        return critic_loss

    def estimate_actor_loss_function(self):
        actor_loss = list()
        for t_idx in range(self.params['n_trajectory_per_rollout']):
            advantage = apply_discount(self.trajectory['advantage'][t_idx])
            # TODO: Compute actor loss function
        actor_loss = [torch.mean(-self.trajectory['log_prob'][t_idx] * apply_discount(self.trajectory['advantage'][t_idx].tolist())) for t_idx in range(self.params['n_trajectory_per_rollout'])]
        return torch.mean(torch.stack(actor_loss))

    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action_idx, log_prob = self.actor_net(torch.tensor(obs, dtype=torch.float32, device=get_device()))
            obs, reward, terminated, truncated, info = self.env.step(self.agent.action_space[action_idx.item()])
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)


# CLass for actor-net
class ActorNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(ActorNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        action_probs = self.network(obs)
        dist = Categorical(action_probs)
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        return action_index, log_prob


# CLass for actor-net
class CriticNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(CriticNet, self).__init__()
        # TODO: Define the critic net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, obs):
        return self.network(obs)



# Class for agent
class ACAgent:
    def __init__(self, env, params=None):
        self.env = env
        self.params = params
        self.action_space = [action for action in range(self.env.action_space.n)]

    def collect_trajectory(self, policy):
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        rollout_buffer = list()
        for _ in range(self.params['n_trajectory_per_rollout']):
            trajectory_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
            while True:
                obs = torch.tensor(obs, dtype=torch.float32, device=get_device())
                # Save observation
                trajectory_buffer['obs'].append(obs)
                action_idx, log_prob = policy(obs)
                obs, reward, terminated, truncated, info = self.env.step(self.action_space[action_idx.item()])
                # Save log-prob and reward into the buffer
                trajectory_buffer['log_prob'].append(log_prob)
                trajectory_buffer['reward'].append(reward)
                # Check for termination criteria
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    rollout_buffer.append(trajectory_buffer)
                    break
        rollout_buffer = self.serialize_trajectory(rollout_buffer)
        return rollout_buffer

    # Converts a list-of-dictionary into dictionary-of-list
    @staticmethod
    def serialize_trajectory(rollout_buffer):
        serialized_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
        for trajectory_buffer in rollout_buffer:
            serialized_buffer['obs'].append(torch.stack(trajectory_buffer['obs']))
            serialized_buffer['log_prob'].append(torch.stack(trajectory_buffer['log_prob']))
            serialized_buffer['reward'].append(trajectory_buffer['reward'])
        return serialized_buffer


class DQNTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.q_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = self.params['init_epsilon']
        self.optimizer = Adam(params=self.q_net.parameters(), lr=self.params['lr'])
        self.replay_memory = ReplayMemory(capacity=self.params['rm_cap'])

    def run_training_loop(self):
        list_ep_reward = list()
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        for idx_episode in range(self.params['n_episode']):
            ep_len = 0
            while True:
                ep_len += 1
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    self.epsilon = max(self.epsilon*self.params['epsilon_decay'], self.params['min_epsilon'])
                    next_obs = None
                    self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
                    list_ep_reward.append(ep_len)
                    print(f'End of episode {idx_episode} with epsilon = {self.epsilon: 0.2f} and reward = {ep_len}, memory = {len(self.replay_memory.buffer)}')
                    obs, _ = self.env.reset()
                    break
                self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
                obs = copy.deepcopy(next_obs)
                self.update_q_net()
                self.update_target_net()
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ep_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def get_action(self, obs):
        # TODO: Implement the epsilon-greedy behavior
        # HINT: The agent will will choose action based on maximum Q-value with
        # '1-ε' probability, and a random action with 'ε' probability.
        action = ???
        return action

    def update_q_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        # TODO: Update Q-net
        # HINT: You should draw a batch of random samples from the replay buffer
        # and train your Q-net with that sampled batch.

        predicted_state_value = ???
        target_value = ???

        criterion = nn.SmoothL1Loss()
        q_loss = criterion(predicted_state_value, target_value.unsqueeze(1))
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        q_net_state_dict = self.q_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = self.params['tau']*q_net_state_dict[key] + (1 - self.params['tau'])*target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        self.epsilon = 0.0
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action = self.get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)


class ReplayMemory:
    # TODO: Implement replay buffer
    # HINT: You can use python data structure deque to construct a replay buffer
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, n_samples):
        batch = random.sample(self.buffer, n_samples)
        batch = [e for e in batch if e is not None]
        states, actions, rewards, next_states, dones = zip(*batch)
        next_states = [
        np.zeros_like(state) if next_state is None else next_state
        for state, next_state in zip(states, next_states)
        ]
    
        batch = [elem for elem in batch if elem is not None]

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones


class QNet(nn.Module):
    # TODO: Define Q-net
    # This is identical to policy network fro
    class QNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(QNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, obs):
        return self.network(obs)
