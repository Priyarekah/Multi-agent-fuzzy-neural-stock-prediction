import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Define Actor and Critic Networks for DDPG
class Actor:
    def __init__(self, state_dim, action_dim):
        self.model = self.build_model(state_dim, action_dim)

    def build_model(self, state_dim, action_dim):
        inputs = layers.Input(shape=(state_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(action_dim, activation='tanh')(x)  # Actions are bounded
        return models.Model(inputs, outputs)

class Critic:
    def __init__(self, state_dim, action_dim):
        self.model = self.build_model(state_dim, action_dim)

    def build_model(self, state_dim, action_dim):
        state_input = layers.Input(shape=(state_dim,))
        action_input = layers.Input(shape=(action_dim,))
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(1)(x)  # Q-value
        return models.Model([state_input, action_input], outputs)

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005  # Soft update parameter

    def update_target_networks(self):
        for target, source in zip(self.target_actor.model.variables, self.actor.model.variables):
            target.assign(self.tau * source + (1 - self.tau) * target)
        for target, source in zip(self.target_critic.model.variables, self.critic.model.variables):
            target.assign(self.tau * source + (1 - self.tau) * target)

    def train(self, state, action, reward, next_state, done):
        # Update critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor.model(next_state)
            target_q_values = self.target_critic.model([next_state, target_actions])
            target_q = reward + self.gamma * (1 - done) * target_q_values
            current_q = self.critic.model([state, action])
            critic_loss = tf.reduce_mean(tf.square(current_q - target_q))
        critic_gradients = tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.optimizer.apply_gradients(zip(critic_gradients, self.critic.model.trainable_variables))
        
        # Update actor
        with tf.GradientTape() as tape:
            actions = self.actor.model(state)
            actor_loss = -tf.reduce_mean(self.critic.model([state, actions]))
        actor_gradients = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_gradients, self.actor.model.trainable_variables))
        
        # Update target networks
        self.update_target_networks()

# Example usage
state_dim = 10  # Replace with actual state dimensions
action_dim = 1  # Replace with actual action dimensions
agent = DDPGAgent(state_dim, action_dim)

# Simulating a simple training loop
n_steps = 100  # Number of steps in the training loop
for step in range(n_steps):
    state = np.random.randn(1, state_dim)  # Simulated state
    action = np.random.randn(1, action_dim)  # Simulated action
    reward = np.random.randn(1)  # Simulated reward
    next_state = np.random.randn(1, state_dim)  # Simulated next state
    done = np.random.choice([0, 1])  # Simulated done flag

    agent.train(state, action, reward, next_state, done)
    
    # Output the progress every 10 steps
    if step % 10 == 0:
        print(f"Step {step}/{n_steps} - Training in progress...")