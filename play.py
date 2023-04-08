import torch
import numpy as np
from unityagents import UnityEnvironment
from ddpg import DDPG


def play(agent, env, brain_name, num_agents, episodes=5, max_steps=1000, load_actor_weights=None, load_critic_weights=None):
    if load_actor_weights is not None:
        agent.actor.load_state_dict(torch.load(load_actor_weights))
    if load_critic_weights is not None:
        agent.critic.load_state_dict(torch.load(load_critic_weights))

    for episode in range(1, episodes + 1):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        agent_scores = np.zeros(num_agents)

        for step in range(max_steps):
            actions = np.array([agent.act(states[i], noise=0.0) for i in range(num_agents)])
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            states = next_states
            agent_scores += rewards

            if np.any(dones):
                break

        print(f"Episode {episode}, Score: {np.mean(agent_scores):.2f}")

if __name__ == "__main__":
    env = UnityEnvironment(file_name='Reacher_20.app')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    num_agents = 20

    state_dim = 33
    action_dim = brain.vector_action_space_size

    hidden_dim = 400
    batch_size = 256
    actor_lr = 5e-4
    critic_lr = 1e-3
    tau = 5e-3
    gamma = 0.995

    agent = DDPG(state_dim, action_dim, hidden_dim=hidden_dim, buffer_size=200000, batch_size=batch_size,
                 actor_lr=actor_lr, critic_lr=critic_lr, tau=tau, gamma=gamma)

    play(agent, env, brain_name, num_agents, episodes=5, max_steps=1000, load_actor_weights="actor_final.pth", load_critic_weights="critic_final.pth")
