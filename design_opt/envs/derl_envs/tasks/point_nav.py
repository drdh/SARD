import numpy as np
from gym import utils

from design_opt.envs.derl_envs.config import cfg
from design_opt.envs.derl_envs.modules.agent import Agent
from design_opt.envs.derl_envs.modules.objects import Objects
from design_opt.envs.derl_envs.modules.terrain import Terrain
from design_opt.envs.derl_envs.tasks.unimal import UnimalEnv
from design_opt.envs.derl_envs.wrappers.hfield import HfieldObs2D
from design_opt.envs.derl_envs.wrappers.hfield import StandReward
from design_opt.envs.derl_envs.wrappers.hfield import TerminateOnFalling
from design_opt.envs.derl_envs.wrappers.hfield import TerminateOnTorsoHitFloor
from design_opt.envs.derl_envs.wrappers.hfield import TerminateOnRotation
from design_opt.envs.derl_envs.wrappers.hfield import UnimalHeightObs
from design_opt.envs.derl_envs.wrappers.metrics import ReachMetric
from design_opt.envs.derl_envs.wrappers.reach import ReachReward
from design_opt.envs.derl_envs.wrappers.multi_env_wrapper import MultiUnimalNodeCentricAction
from design_opt.envs.derl_envs.wrappers.multi_env_wrapper import MultiUnimalNodeCentricObservation


class PointNavTask(UnimalEnv, utils.EzPickle):
    def __init__(self, xml_str, unimal_id):
        UnimalEnv.__init__(self, xml_str, unimal_id)

    ###########################################################################
    # Sim step and reset
    ###########################################################################

    def step(self, action):
        xy_pos_before = self.sim.data.get_body_xpos("torso/0")[:2].copy()
        self.do_simulation(action)
        xy_pos_after = self.sim.data.get_body_xpos("torso/0")[:2].copy()

        ctrl_cost = self.control_cost(action)
        reward = -ctrl_cost
        observation = self._get_obs()

        # Calculate distance between agent and goal
        goal_pos = self.modules["Objects"].goal_pos[:2]
        agent_goal_d = np.linalg.norm(goal_pos - xy_pos_after)

        info = {
            "x_pos": xy_pos_after[0] - cfg.TERRAIN.SIZE[0],
            "y_pos": xy_pos_after[1],
            "xy_pos_before": xy_pos_before,
            "xy_pos_after": xy_pos_after,
            "agent_goal_d": agent_goal_d,
            "goal_pos": goal_pos,
            "__reward__energy": self.calculate_energy(),
            "__reward__ctrl": ctrl_cost,
        }

        # Update viewer with markers, if any
        if self.viewer is not None:
            self.viewer._markers[:] = []
            for marker in self.metadata["markers"]:
                self.viewer.add_marker(**marker)

        return observation, reward, False, info


def make_env_point_nav(xml, unimal_id):
    env = PointNavTask(xml, unimal_id)
    # Add modules
    for module in cfg.ENV.MODULES:
        env.add_module(globals()[module])
    env.reset()
    # Add all wrappers
    env = UnimalHeightObs(env)
    env = StandReward(env)
    env = TerminateOnFalling(env)
    env = HfieldObs2D(env)
    if "ReachReward" in cfg.ENV.WRAPPERS:
        env = ReachReward(env)

    env = ReachMetric(env)
    env = TerminateOnTorsoHitFloor(env)
    env = TerminateOnRotation(env)

    for wrapper_func in cfg.MODEL.WRAPPERS:
        env = globals()[wrapper_func](env)
    return env
