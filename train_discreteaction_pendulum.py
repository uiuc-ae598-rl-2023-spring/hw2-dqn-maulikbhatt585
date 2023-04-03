import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import discreteaction_pendulum
import random
from collections import namedtuple, deque
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from DQN import *
import discreteaction_pendulum

def value_fun(policy_net, x, y):
    s = torch.tensor([x,y])
    s1 = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
    return torch.max(policy_net(s1)).item()

def main():
    batch_size = 64
    gamma = 0.95
    eps_start = 1
    eps_end = 0.1
    num_episodes = 100

    env = discreteaction_pendulum.Pendulum()

    n_actions = env.num_actions
    n_states = env.num_states

    num_runs = 20

    reward_array_list = []

    for run in range(num_runs):

        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

        policy_net = DQN(n_states, n_actions)
        target_net = DQN(n_states, n_actions)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95)
        memory = ReplayMemory(1000000, Transition)

        policy_net, reward_array = train_network(num_episodes, policy_net, target_net, optimizer, memory, env, n_actions, n_states, batch_size, eps_start, eps_end, gamma, Transition)

        reward_array_list.append(reward_array)


    mean_reward = np.mean(reward_array_list, axis=0)

    plt.plot(mean_reward)
    plt.title("Mean Reward - Standard Algorithm")
    plt.xlabel("Number of epochs")
    plt.ylabel("Mean reward")
    plt.legend()
    plt.savefig('figures/Final/learning_curve.png')

    policy = lambda s: (policy_net(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).max(1)[1].view(1, 1)).item()

    env.video(policy, filename='figures/final/test_discreteaction_pendulum.gif')


    dx, dy = 0.15, 0.05
    y, x = np.mgrid[-3:3+dy:dy, -15:15+dx:dx]
    X = np.arange(-3,3,dy)
    Y = np.arange(-15,15,dx)
    z = np.zeros(y.shape)
    print(z.shape)
    print(X.shape)
    print(Y.shape)

    for i in range(len(X)):
        for j in range(len(Y)):
            z[i,j] = value_fun(policy_net,X[i],Y[j])
    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = z[:-1, :-1]
    z_min, z_max = np.min(z), np.max(z)

    fig, ax = plt.subplots(1,1)

    c = ax.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('State Value Function')
    ax.set_xlabel("Theta")
    ax.set_ylabel("Theta_dot")
    fig.colorbar(c, ax=ax)

    plt.savefig('figures/Final/state_value_function.png')

    z = np.zeros(y.shape)

    for i in range(len(X)):
        for j in range(len(Y)):
            s = torch.tensor([X[i],Y[j]])
            #s1 = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            z[i,j] = policy(s)
    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = z[:-1, :-1]
    z_min, z_max = np.min(z), np.max(z)

    fig, ax = plt.subplots(1,1)

    c = ax.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('Policy')
    ax.set_xlabel("Theta")
    ax.set_ylabel("Theta_dot")
    fig.colorbar(c, ax=ax)

    plt.savefig('figures/Final/policy.png')

    s = env.reset()

    # Create dict to store data from simulation
    data = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = policy(s)
        (s, r, done) = env.step(a)
        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(r)

    # Parse data from simulation
    data['s'] = np.array(data['s'])
    theta = data['s'][:, 0]
    thetadot = data['s'][:, 1]
    tau = [env._a_to_u(a) for a in data['a']]

    # Plot data and save to png file
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(data['t'], theta, label='theta')
    ax[0].plot(data['t'], thetadot, label='thetadot')
    ax[0].legend()
    ax[1].plot(data['t'][:-1], tau, label='tau')
    ax[1].legend()
    ax[2].plot(data['t'][:-1], data['r'], label='r')
    ax[2].legend()
    ax[2].set_xlabel('time step')
    plt.tight_layout()
    plt.savefig('figures/Final/test_discreteaction_pendulum.png')

    reward_array_list_2 = []

    for run in range(num_runs):

        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

        policy_net = DQN(n_states, n_actions)
        target_net = DQN(n_states, n_actions)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95)
        memory = ReplayMemory(1000000, Transition)

        policy_net, reward_array = train_network_without_targ(num_episodes, policy_net, target_net, optimizer, memory, env, n_actions, n_states, batch_size, eps_start, eps_end, gamma, Transition)

        reward_array_list_2.append(reward_array)


    mean_reward_2 = np.mean(reward_array_list_2, axis=0)

    reward_array_list_3 = []

    for run in range(num_runs):

        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

        policy_net = DQN(n_states, n_actions)
        target_net = DQN(n_states, n_actions)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95)
        memory = ReplayMemory(batch_size, Transition)

        policy_net, reward_array = train_network(num_episodes, policy_net, target_net, optimizer, memory, env, n_actions, n_states, batch_size, eps_start, eps_end, gamma, Transition)

        reward_array_list_3.append(reward_array)


    mean_reward_3 = np.mean(reward_array_list_3, axis=0)

    reward_array_list_4 = []

    for run in range(num_runs):

        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

        policy_net = DQN(n_states, n_actions)
        target_net = DQN(n_states, n_actions)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95)
        memory = ReplayMemory(batch_size, Transition)

        policy_net, reward_array = train_network_without_targ(num_episodes, policy_net, target_net, optimizer, memory, env, n_actions, n_states, batch_size, eps_start, eps_end, gamma, Transition)

        reward_array_list_4.append(reward_array)


    mean_reward_4 = np.mean(reward_array_list_4, axis=0)

    fig, ax = plt.subplots(1,1)
    plt.plot(mean_reward, label="With replay, with target Q")
    plt.plot(mean_reward_2, label="With replay, without target Q")
    plt.plot(mean_reward_3, label="Without replay, with target Q")
    plt.plot(mean_reward_4, label="Without replay, without target Q")
    plt.title("Mean Rewards")
    plt.xlabel("Number of epochs")
    plt.ylabel("Mean reward")
    plt.legend()
    plt.savefig('figures/Final/ablation_study.png')


if __name__ == '__main__':
    main()
