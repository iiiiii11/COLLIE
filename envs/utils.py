import gym
import akro
import numpy as np


def convert_gym_spaces_to_akro(env):
    """
    Converts the action and observation spaces of a gym environment to akro spaces.

    Parameters:
        env (gym.Env): The gym environment.

    Returns:
        tuple: A tuple containing the akro action space and akro observation space.
    """
    def convert_space(gym_space):
        if isinstance(gym_space, gym.spaces.Discrete):
            return akro.Discrete(gym_space.n)
        elif isinstance(gym_space, gym.spaces.Box):
            return akro.Box(low=gym_space.low, high=gym_space.high, dtype=gym_space.dtype)
        elif isinstance(gym_space, gym.spaces.Tuple):
            return akro.Tuple([convert_space(space) for space in gym_space.spaces])
        elif isinstance(gym_space, gym.spaces.Dict):
            return akro.Dict({k: convert_space(space) for k, space in gym_space.spaces.items()})
        elif isinstance(gym_space, gym.spaces.MultiDiscrete):
            return akro.MultiDiscrete(gym_space.nvec)
        elif isinstance(gym_space, gym.spaces.MultiBinary):
            return akro.MultiBinary(gym_space.n)
        else:
            raise ValueError(f"Unsupported space type: {type(gym_space)}")

    akro_action_space = convert_space(env.action_space)
    akro_observation_space = convert_space(env.observation_space)

    return akro_action_space, akro_observation_space


# Example usage:
if __name__ == "__main__":
    # Create a gym environment
    env = gym.make("CartPole-v1")

    # Convert the spaces
    akro_action_space, akro_observation_space = convert_gym_spaces_to_akro(env)

    print("Akro Action Space:", akro_action_space)
    print("Akro Observation Space:", akro_observation_space)
