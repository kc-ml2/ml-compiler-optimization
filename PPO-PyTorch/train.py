import json
import os
from datetime import datetime
from pathlib import Path

import compiler_gym
import networkx as nx
import numpy as np
import torch
import torch_geometric as pyg
from compiler_gym.datasets import Datasets
from compiler_gym.wrappers import CommandlineWithTerminalAction, TimeLimit

from PPO import PPO

_dir = os.path.join(Path(__file__).parent, 'vocab.json')
with open(_dir, 'r') as fp:
    vocab = json.load(fp)

node_features = ['text', 'type']
edge_features = ['flow', 'position']


def parse_nodes(ns):
    # in-place
    for nid in ns:
        n = ns[nid]
        n.pop('function', None)
        n.pop('block', None)
        n.pop('features', None)
        n['text'] = vocab.get(n['text'], len(vocab))


def parse_edges(es):
    # in-place
    for eid in es:
        e = es[eid]
        e['position'] = min(e['position'], 5120)


def to_pyg(g):
    parse_nodes(g.nodes)
    parse_edges(g.edges)
    g = nx.DiGraph(g)
    g = pyg.utils.from_networkx(
        g,
        group_node_attrs=node_features,
        group_edge_attrs=edge_features
    )

    return g


################################### Training ###################################
def train():
    print(
        "============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "llvm-ic-v0"

    max_ep_len = 256  # max timesteps in one episode
    num_epi = 8192

    max_training_timesteps = max_ep_len * 4
    # break training loop if timeteps > max_training_timesteps

    save_model_freq = int(4096)  # save model frequency (in num timesteps)

    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = 4  # update policy every n timesteps
    K_epochs = 4  # update policy for K epochs in one PPO update

    eps_clip = 0.1  # clip parameter for PPO
    gamma = 0.9  # discount factor

    lr_actor = 0.0001  # learning rate for actor network
    lr_critic = 0.0001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    env = compiler_gym.make(
        "llvm-ic-v0",
        observation_space="Programl",
        reward_space="IrInstructionCountOz",
    )
    env = CommandlineWithTerminalAction(env)
    # ds = env.datasets["benchmark://mibench-v1"]
    # env = CycleOverBenchmarks(env, ds.benchmarks())

    # action space dimension
    action_dim = env.action_space.n

    ################### checkpointing ###################
    run_num_pretrained = datetime.now().strftime(
        '%Y%m%d%H%M%S')  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(
        env_name,
        random_seed,
        run_num_pretrained
    )
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    ############# print all hyperparameters #############
    print(
        "--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print(
        "--------------------------------------------------------------------------------------------")
    print("action space dimension : ", action_dim)
    print(
        "--------------------------------------------------------------------------------------------")
    print("Initializing a discrete action space policy")
    print(
        "--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print(
        "--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print(
            "--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print(
        "============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print(
        "============================================================================================")
    time_step = 0
    i_episode = 0

    # training loop
    # time_limits = [128]
    time_limits = [env.action_space.n]
    env = TimeLimit(env, 256)
    env.datasets = Datasets([
        # "benchmark://mibench-v1",
        env.datasets["benchmark://cbench-v1"],
        env.datasets["benchmark://blas-v0"],
    ])
    # for lim in time_limits:
    # for lim in time_limits:
    # if isinstance(env, TimeLimit):
    # print(f'time limit {env._max_episode_steps}')
    while i_episode < 8192:  # and time_step <= max_training_timesteps:
        print('epi', i_episode)
        done = False
        try:
            env.benchmark = env.datasets.random_benchmark(
                weighted=True,
            )
        except Exception as e:
            continue
        print(f'{env.benchmark}')
        state = env.reset()
        state = to_pyg(state)

        current_ep_reward = 0
        t = 0
        while not done:

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # if action == len(env.action_space.flags) - 1:
            #     print(action)
            #     reward -= 0.1

            state = to_pyg(state)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # save model weights
            if time_step % save_model_freq == 0:
                print(
                    "--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ",
                      datetime.now().replace(microsecond=0) - start_time)
                print(
                    "--------------------------------------------------------------------------------------------")

            t += 1

        print(f'{t}')
        print(f'{current_ep_reward:0.3f}')
        print()
        print()
        i_episode += 1

    env.close()

    # print total training time
    print(
        "============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print(
        "============================================================================================")


if __name__ == '__main__':
    train()
