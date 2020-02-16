from typing import Dict
import argparse
import yaml

from actorcritic import TableCritic, ANNCritic, Actor, Critic
from agent import Agent
from app import App
from peg_solitaire.env import PegSolitaireEnvironment
import matplotlib.pyplot as plt


def build_actor_critic(cfg: Dict, state_shape, action_shape):
    critic: Critic
    if cfg['critic_type'] == 'table':
        critic = TableCritic(state_shape=state_shape, action_shape=action_shape)
    else:
        critic = ANNCritic(**cfg['critic'], state_shape=state_shape, action_shape=action_shape)
    actor = Actor(**cfg['actor'], state_shape=state_shape, action_shape=action_shape)
    return actor, critic


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-nd", "--no_draw", default=False)
    args = parser.parse_args()

    with open("config.yml", 'r') as cfg_file:
        config = yaml.load(cfg_file, Loader=yaml.FullLoader)

    env = PegSolitaireEnvironment()
    env.setup(config)

    cfg_agent = config['agent']
    # agent = Agent(env, *build_actor_critic(cfg_agent['acm'], (env.board.hole_count,), (cfg_agent['action_axes']),))

    env.user_modify()

    # TODO: Add oop for user to set up scenario...
    # app.mainloop()
    plt.show()