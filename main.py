from typing import Dict
import argparse
import yaml

from actorcritic import TableCritic, ANNCritic, Actor, Critic, TableActor
from agent import Agent, RandomAgent
from app_UNUSED import App
from peg_solitaire.env import PegSolitaireEnvironment
import matplotlib.pyplot as plt


def build_actor_critic(cfg: Dict, state_shape, action_shape):
    critic: Critic
    if cfg['critic_type'] == 'table':
        expected = {key: cfg['critic'][key] for key in TableCritic.__init__.__code__.co_varnames[3:]}
        critic = TableCritic(state_shape=state_shape, action_shape=action_shape, **expected)
    else:
        critic = ANNCritic(**cfg['critic'], state_shape=state_shape, action_shape=action_shape)
    actor = TableActor(**cfg['actor'], state_shape=state_shape, action_shape=action_shape)
    # actor = Actor(**cfg['actor'], state_shape=state_shape, action_shape=action_shape)
    return actor, critic


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-nd", "--no_draw", default=False, nargs='?', const=True)
    parser.add_argument("-g", "--greedy", default=False, nargs='?', const=True)
    parser.add_argument("-i", "--user_input", default=False, nargs='?', const=True)
    args = parser.parse_args()

    with open("config.yml", 'r') as cfg_file:
        config = yaml.load(cfg_file, Loader=yaml.FullLoader)

    env = PegSolitaireEnvironment()
    env.setup(config)

    env.user_modify(args.user_input)

    cfg_agent = config['agent']
    rand_agent = RandomAgent(*build_actor_critic(cfg_agent['acm'], (env.board.hole_count,), (cfg_agent['action_axes']),))
    # agent = PegSolitaireAgent(env, *build_actor_critic(cfg_agent['acm'], (env.board.hole_count,), (cfg_agent['action_axes']),))

    env.should_render = not args.no_draw
    rand_agent.learn(env, cfg_agent['episodes'])

    # saps = env.generate_state_action_pairs()
    # print(saps)
    plt.show()
