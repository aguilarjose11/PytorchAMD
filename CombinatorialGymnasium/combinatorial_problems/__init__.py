from gymnasium.envs.registration import register

register(
    id="combinatorial_problems/GridWorld-v0",
    entry_point="combinatorial_problems.envs:GridWorldEnv",
    max_episode_steps=300,
)

register(
    id="combinatorial_problems/TravelingSalesman-v0",
    entry_point="combinatorial_problems.envs:TravelingSalesmanEnv",
    max_episode_steps=300,
)

register(
    id="combinatorial_problems/Phase1Env-v0",
    entry_point="combinatorial_problems.envs:Phase1Env",
    max_episode_steps=50,
)

register(
    id="combinatorial_problems/Phase2Env-v0",
    entry_point="combinatorial_problems.envs:Phase2Env",
    max_episode_steps=50,
)