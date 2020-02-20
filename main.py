import os
from typing import Dict
import argparse

import matplotlib
import yaml

from actorcritic import TableCritic, ANNCritic, Actor, Critic, TableActor
from agent import Agent, RandomAgent
from peg_solitaire.agent import PegSolitaireAgent
from peg_solitaire.env import PegSolitaireEnvironment
import matplotlib.pyplot as plt

def parse_args_and_config():
    """
    :return: arguments, configuration dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graphics", default=False, nargs='?', const=True)
    parser.add_argument("-i", "--interactive", default=False, nargs='?', const=True)
    parser.add_argument("-cfg", "--config", type=str, default=["config.yml"], nargs=1)
    args = parser.parse_args()

    if not os.path.isfile(args.config[0]):
        print(f"unable to open config file: \"{args.config[0]}\"")
        args.config[0] = "config.yml"
    with open(args.config[0]) as cfg_file:
        print("using config file: ", args.config[0])
        config = yaml.load(cfg_file, Loader=yaml.FullLoader)
    return args, config


def build_actor_critic(acm_cfg: Dict, state_shape, action_shape):
    """
    Read configs to build an actor and a critic
    :param acm_cfg: the actor-critic config dictionary
    :param state_shape: the shape of the environment representation passed on to the agent
    :param action_shape: the shape of the agent's action
    :return: an actor and a critic
    """
    critic: Critic
    if acm_cfg['critic_type'] == 'table':
        expected = {key: acm_cfg['critic'][key] for key in TableCritic.__init__.__code__.co_varnames[3:]}
        critic = TableCritic(state_shape=state_shape, action_shape=action_shape, **expected)
    else:
        expected = {key: acm_cfg['critic'][key] for key in ANNCritic.__init__.__code__.co_varnames[3:]}
        critic = ANNCritic(**expected, state_shape=state_shape, action_shape=action_shape)
    actor = TableActor(**acm_cfg['actor'], action_shape=action_shape)
    return actor, critic


if __name__ == '__main__':

    args, config = parse_args_and_config()

    matplotlib.use(backend="TkAgg")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Create Environment and set it up
    env = PegSolitaireEnvironment()
    env.setup(config)

    if args.interactive:
        env.user_modify()

    env.should_render = args.graphics

    # Create agent start training
    cfg_agent = config['agent']
    agent = PegSolitaireAgent(*build_actor_critic(cfg_agent['acm'], (env.board.hole_count,), (cfg_agent['action_axes']),))

    agent.learn(env, cfg_agent['episodes'])

    env.should_render = True
    agent.test(env, cfg_agent['tests'])

    plt.show()
