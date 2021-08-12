from gym.envs.registration import register

register(
    id="turtle2d-v0",
    entry_point="gym_turtle2d.envs:Turtle2DEnv",
)
