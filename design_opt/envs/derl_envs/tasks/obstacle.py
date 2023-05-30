from gym import utils

from design_opt.envs.derl_envs.config import cfg
from design_opt.envs.derl_envs.modules.agent import Agent
from design_opt.envs.derl_envs.modules.objects import Objects
from design_opt.envs.derl_envs.modules.terrain import Terrain
from design_opt.envs.derl_envs.tasks.unimal import UnimalEnv
from design_opt.envs.derl_envs.wrappers.hfield import HfieldObs2D
from design_opt.envs.derl_envs.wrappers.hfield import StandReward
from design_opt.envs.derl_envs.wrappers.hfield import TerminateOnFalling
from design_opt.envs.derl_envs.wrappers.hfield import TerminateOnTerrainEdge
from design_opt.envs.derl_envs.wrappers.hfield import TerminateOnTorsoHitFloor
from design_opt.envs.derl_envs.wrappers.hfield import TerminateOnRotation
from design_opt.envs.derl_envs.wrappers.hfield import UnimalHeightObs
from design_opt.envs.derl_envs.wrappers.multi_env_wrapper import MultiUnimalNodeCentricAction
from design_opt.envs.derl_envs.wrappers.multi_env_wrapper import MultiUnimalNodeCentricObservation


class ObstacleTask(UnimalEnv, utils.EzPickle):
    def __init__(self, xml_str, unimal_id):
        UnimalEnv.__init__(self, xml_str, unimal_id)

    ###########################################################################
    # Sim step and reset
    ###########################################################################

    def step(self, action):
        xy_pos_before = self.sim.data.get_body_xpos("torso/0")[:2].copy()
        mj_step_error = self.do_simulation(action)
        xy_pos_after = self.sim.data.get_body_xpos("torso/0")[:2].copy()

        xy_vel = (xy_pos_after - xy_pos_before) / self.dt
        x_vel, y_vel = xy_vel

        forward_reward = cfg.ENV.FORWARD_REWARD_WEIGHT * x_vel

        ctrl_cost = self.control_cost(action)
        reward = forward_reward - ctrl_cost
        observation = self._get_obs()

        info = {
            "__reward__ctrl": ctrl_cost,
            "__reward__energy": self.calculate_energy(),
            "x_pos": xy_pos_after[0],
            "x_vel": x_vel,
            "xy_pos_before": xy_pos_before,
            "xy_pos_after": xy_pos_after,
            "__reward__forward": forward_reward,
            "metric": xy_pos_after[0],
            "name": self.unimal_id,
            "mj_step_error": mj_step_error
        }

        return observation, reward, False, info


def make_env_obstacle(xml, unimal_id):
    env = ObstacleTask(xml, unimal_id)
    # Add modules
    for module in cfg.ENV.MODULES:
        env.add_module(globals()[module])
    # Reset is needed to setup observation spaces, sim etc which might be
    # needed by wrappers
    env.reset()
    # Add all wrappers
    env = UnimalHeightObs(env)
    env = StandReward(env)
    env = TerminateOnFalling(env)
    env = HfieldObs2D(env)
    env = TerminateOnTerrainEdge(env)
    env = TerminateOnTorsoHitFloor(env)
    env = TerminateOnRotation(env)

    for wrapper_func in cfg.MODEL.WRAPPERS:
        env = globals()[wrapper_func](env)
    return env
