import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
import pybullet_envs
import gym
from tqdm import trange
import time
class ReplayBuffer:
def __init__(self, max_size, input_shape, n_actions):
    self.m_size = max_size
    self.m_cntr = 0
    self.state_m = np.zeros((self.m_size, *input_shape))
    self.new_state_m = np.zeros((self.m_size, *input_shape))
    self.action_m = np.zeros((self.m_size, n_actions))
    self.reward_m = np.zeros(self.m_size)
    self.terminal_m = np.zeros(self.m_size, dtype=bool)
def store(self, state, action, reward, state_, done):
    id = self.m_cntr % self.m_size
    self.state_m[id] = state
    self.new_state_m[id] = state_
    self.action_m[id] = action
    self.reward_m[id] = reward
    self.terminal_m[id] = done
    self.m_cntr += 1
def sample(self, batch_size):
    max_m = min(self.m_cntr, self.m_size)
    batch = np.random.choice(max_m, batch_size)
    return (self.state_m[batch], self.action_m[batch], self.reward_m[batch],
            self.new_state_m[batch], self.terminal_m[batch])
class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='tmp/super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)
        self.model_name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')
    def call(self, state, action):
        x = self.fc1(tf.concat([state, action], axis=1))
        x = self.fc2(x)
        return self.q(x)
class ValueNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)
        self.model_name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.v(x)
class ActorNetwork(keras.Model):
    def __init__(self, max_action, fc1_dims=256, fc2_dims=256, n_actions=2, name='actor', chsuper(ActorNetwork, self).__init__()
        self.max_action = max_action
        self.noise = 1e-6
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.mu = Dense(n_actions)
        self.sigma = Dense(n_actions)
        self.model_name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)
        sigma = tf.clip_by_value(self.sigma(x), self.noise, 1)
        return mu, sigma
    def sample_normal(self, state, reparameterize=True):
    mu, sigma = self.call(state)
    dist = tfp.distributions.Normal(mu, sigma)
    actions = dist.sample() if not reparameterize else dist.sample()
    action = tf.tanh(actions) * self.max_action
    log_probs = dist.log_prob(actions)
    log_probs -= tf.math.log(1 - tf.pow(action, 2) + self.noise)
    log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)
    return action, log_probs
class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], env=None,
        gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
        layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.scale = reward_scale
        self.actor = ActorNetwork(n_actions=n_actions, name='actor', max_action=env.action_sself.critic_1 = CriticNetwork(n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(name='value')
        self.target_value = ValueNetwork(name='target_value')
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))
        self.update_network_parameters(tau=1)
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions[0].numpy()
    def remember(self, state, action, reward, new_state, done):
        self.memory.store(state, action, reward, new_state, done)
    def update_network_parameters(self, tau=None):
        tau = tau or self.tau
        weights = [w * tau + t * (1 - tau) for w, t in zip(self.value.get_weights(), self.taself.target_value.set_weights(weights)
    def learn(self):
        if self.memory.m_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample(self.batch_size)
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        dones = tf.convert_to_tensor(done, dtype=tf.float32)
        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            value_ = tf.squeeze(self.target_value(states_), 1)
            new_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs, 1)
            q1 = tf.squeeze(self.critic_1(states, new_actions), 1)
            q2 = tf.squeeze(self.critic_2(states, new_actions), 1)
            critic_value = tf.minimum(q1, q2)
            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)
        grads = tape.gradient(value_loss, self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(grads, self.value.trainable_variables))
        with tf.GradientTape() as tape:
            new_actions, log_probs = self.actor.sample_normal(states, reparameterize=True)
            log_probs = tf.squeeze(log_probs, 1)
            q1 = tf.squeeze(self.critic_1(states, new_actions), 1)
            q2 = tf.squeeze(self.critic_2(states, new_actions), 1)
            critic_value = tf.minimum(q1, q2)
            actor_loss = tf.reduce_mean(log_probs - critic_value)
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        with tf.GradientTape(persistent=True) as tape:
            q_hat = self.scale * rewards + self.gamma * value_ * (1 - dones)
            q1_old = tf.squeeze(self.critic_1(states, actions), 1)
            q2_old = tf.squeeze(self.critic_2(states, actions), 1)
            c1_loss = 0.5 * keras.losses.MSE(q1_old, q_hat)
            c2_loss = 0.5 * keras.losses.MSE(q2_old, q_hat)
grads1 = tape.gradient(c1_loss, self.critic_1.trainable_variables)
grads2 = tape.gradient(c2_loss, self.critic_2.trainable_variables)
self.critic_1.optimizer.apply_gradients(zip(grads1, self.critic_1.trainable_variableself.critic_2.optimizer.apply_gradients(zip(grads2, self.critic_2.trainable_variableself.update_network_parameters()
def main():
    startt = time.time()
    env = gym.make('InvertedPendulumBulletEnv-v0')
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_spacn_games = 100
    best_score = -np.inf
    score_history = []
    for i in trange(n_games, desc="Training"):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
        if i % 100 == 0:
            print(f"\nEpisode {i}, Score: {score:.1f}, Avg Score: {avg_score:.1f}")
        endt = time.time()
        print("Total Execution Time:", endt - startt)
if __name__ == '__main__':
    main()
