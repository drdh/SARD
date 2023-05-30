from gym.envs.registration import register

register(
    id="Unimal-v0",
    entry_point="design_opt.envs.derl_envs.tasks.task:make_env",
    max_episode_steps=1000,
)

