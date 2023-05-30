import argparse
import sys
import gym
import time
from collections import defaultdict
from collections import deque
import numpy as np

from design_opt.envs.derl_envs.config import cfg
from design_opt.envs.derl_envs.env_viewer import EnvViewer


def make_env(env_id, seed, rank, xml_file=None):
    def _thunk():
        if xml_file:
            env = gym.make(env_id, xml_path=xml_file)
        else:
            env = gym.make(env_id)
        # Note this does not change the global seeds. It creates a numpy
        # rng gen for env.
        env.seed(seed + rank)
        # Don't add wrappers above TimeLimit
        if str(env.__class__.__name__).find("TimeLimit") >= 0:
            env = TimeLimitMask(env)
        # Store the un-normalized rewards
        env = RecordEpisodeStatistics(env)
        return env

    return _thunk

# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info["timeout"] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.t0 = (
            time.time()
        )
        self.episode_return = 0.0
        # Stores individual components of the return. For e.g. return might
        # have separate reward for speed and standing.
        self.episode_return_components = defaultdict(int)
        self.episode_length = 0
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = super(RecordEpisodeStatistics, self).reset(**kwargs)
        self.episode_return = 0.0
        self.episode_length = 0
        return observation

    def step(self, action):
        observation, reward, done, info = super(
            RecordEpisodeStatistics, self
        ).step(action)
        self.episode_return += reward
        self.episode_length += 1
        for key, value in info.items():
            if "__reward__" in key:
                self.episode_return_components[key] += value

        if done:
            info["episode"] = {
                "r": self.episode_return,
                "l": self.episode_length,
                "t": round(time.time() - self.t0, 6),
            }
            for key, value in self.episode_return_components.items():
                info["episode"][key] = value
                self.episode_return_components[key] = 0

            self.return_queue.append(self.episode_return)
            self.length_queue.append(self.episode_length)
            self.episode_return = 0.0
            self.episode_length = 0
        return observation, reward, done, info

def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Create random morphologies.")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.ENV.WALKER = "envs/derl_envs/assets/unimal_ant.xml"
    cfg.freeze()

    env = make_env(cfg.ENV_NAME, cfg.RNG_SEED, 1)()

    act_size = env.action_space.shape[0]

    obs = env.reset()
    for t in range(100):
        obs, reward, done, info = env.step(np.random.randn(act_size))

    # env_viewer = EnvViewer(env)
    # env_viewer.run()


if __name__ == "__main__":
    main()
