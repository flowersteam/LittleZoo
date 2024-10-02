from gymnasium.envs.registration import register
import numpy as np
from .littlezoo import *

import sys
sys.path.append('../')

for v in ['1']:
    register(id='PlaygroundNavigation-v' + v,
             entry_point='little_zoo.playground.playgroundnavv' + v + ':PlayGroundNavigationV' + v,
             max_episode_steps=100)

    register(id='PlaygroundNavigationHuman-v' + v,
             entry_point='little_zoo.playground.playgroundnavv' + v + ':PlayGroundNavigationV' + v,
             max_episode_steps=100,
             kwargs=dict(human=True, render_mode="human"))

    register(id='PlaygroundNavigationRender-v' + v,
             entry_point='little_zoo.playground.playgroundnavv' + v + ':PlayGroundNavigationV' + v,
             max_episode_steps=100,
             kwargs=dict(human=False, render_mode="human"))
