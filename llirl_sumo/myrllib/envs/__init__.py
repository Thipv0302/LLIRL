"""
Environment registration for LLIRL SUMO
"""
from gym.envs.registration import register

# SUMO Single Intersection Environment
# Note: sumo_config_path should be passed when creating the environment
register(
    'SUMO-SingleIntersection-v1',
    entry_point='myrllib.envs.sumo_env:SUMOEnv',
    max_episode_steps=14400  # 4 giờ = 14400 giây
)

