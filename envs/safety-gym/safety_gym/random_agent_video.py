

import argparse
import os
import gym
import safety_gym
import numpy as np
import imageio


def test_env(env_name):
    env = gym.make(env_name)
    obs = env.reset()

    assert env.observation_space.contains(obs)
    act = env.action_space.sample()
    assert env.action_space.contains(act)

    obs, reward, done, info = env.step(act)

    print(f"Environment {env_name} tested successfully.")

    env.close()


def run_random(env_name, output_file):

    env = gym.make(env_name)
    obs = env.reset()
    done = False
    ep_ret = 0
    ep_cost = 0

    frames = []

    while not done:
        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f' % (ep_ret, ep_cost))
            ep_ret, ep_cost = 0, 0
            obs = env.reset()

        frame = env.render(mode='rgb_array', camera_id=0)
        frames.append(frame)

        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)

        obs, reward, done, info = env.step(act)
        ep_ret += reward
        ep_cost += info.get('cost', 0)

    if os.path.exists('./videos') is False:
        os.makedirs('./videos')
    imageio.mimsave(f'./videos/{output_file}_{env_name}.mp4', frames, fps=30)
    print(f"Video saved to ./videos/{output_file}_{env_name}.mp4")
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='', help='The output video file')
    args = parser.parse_args()

    task_list = ['Goal', 'Button', 'Push']
    robot_list = ['Point', 'Car', 'Doggo']
    for task in task_list:
        for robot in robot_list:
            for difficulty in range(3):
                env_name = f'Safexp-{robot}{task}{difficulty}-v0'
                print(f'Running environment: {env_name}')

                run_random(env_name, args.output)
