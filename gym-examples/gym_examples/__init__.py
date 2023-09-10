from gym.envs.registration import register

register(
    id="gym_examples/GridWorld01-v1",
    entry_point="gym_examples.envs:GridWorldEnv_v01",
)

register(
    id="gym_examples/GridWorld02-v1",
    entry_point="gym_examples.envs:GridWorldEnv_v02",
)

register(
    id="gym_examples/GridWorld03-v1",
    entry_point="gym_examples.envs:GridWorldEnv_v03",
)

