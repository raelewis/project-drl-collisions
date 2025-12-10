# Once the gymnasium environment is fully set up, this can be uncommented
from gymnasium.envs.registration import register

register(
    id="HallwayEnv-v0",
    entry_point="drl_collisions_env.envs:HallwayEnv",
    max_episode_steps=2000
)