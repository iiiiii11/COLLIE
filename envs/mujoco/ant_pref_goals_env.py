import akro
import numpy as np
from envs.mujoco.ant_env import AntEnv
from envs.mujoco.mujoco_utils import convert_observation_to_space
from gym import utils
from pref.oracle_pref import SAFE_CENTER_ANT, SAFE_R_ANT, HOLE_CENTERS_ANT, HOLE_R_ANT

UNSAFE_PENALTY = 20.0


def loc_is_safe(pref_task, x, y):
    if pref_task == "n":
        safe_flag = y > np.abs(x)
    elif pref_task == "range":
        loc = np.array([x, y])
        dists_sq = np.sum((loc - SAFE_CENTER_ANT)**2)
        safe_flag = (dists_sq <= SAFE_R_ANT**2).item()
    elif pref_task == "hole2":
        safe_flag = 1
        loc = np.array([x, y])
        for center in HOLE_CENTERS_ANT[1]:
            dists_sq = np.sum((loc - center)**2)
            in_hole = (dists_sq <= HOLE_R_ANT**2).astype(np.float32)
            safe_flag = min(1. - in_hole, safe_flag)
        safe_flag = safe_flag > 0.5
    else:
        raise Exception(f"pref_task ({pref_task}) is invalid")
    return safe_flag


def sample_safe_goals(pref_task, goal_range, num_goals, max_attempts=1000):
    goals = []
    attempts = 0
    while len(goals) < num_goals and attempts < max_attempts:
        x, y = np.random.uniform(-goal_range, goal_range, (2,))
        if loc_is_safe(pref_task, x, y):
            goals.append(np.array([x, y]))
        attempts += 1
    if len(goals) < num_goals:
        raise RuntimeError("sample failed")
    return goals


class AntPrefGoalEnv(AntEnv):
    def __init__(
            self,
            pref_task,
            max_path_length,
            goal_range,
            num_goal_steps,
            reward_type='sparse',
            zero_shot=False,
            **kwargs,
    ):
        self.max_path_length = max_path_length
        self.reward_type = reward_type

        self.pref_task = pref_task
        self.zero_shot = zero_shot

        self.goal_epsilon = 3.
        self.goal_range = goal_range
        self.num_goal_steps = num_goal_steps

        self.goals = sample_safe_goals(pref_task, self.goal_range, num_goals=1)

        self.cur_goal = self.goals[0]
        self.num_steps = 0
        self.goal_success = {
            'goal_1': 0,
            'goal_2': 0,
            'goal_3': 0,
            'goal_4': 0,
        }
        self.goal_staying = {
            'goal_1': 0,
            'goal_2': 0,
            'goal_3': 0,
            'goal_4': 0,
        }
        self.goal_idx = 1

        super().__init__(**kwargs)
        utils.EzPickle.__init__(self, max_path_length=max_path_length, goal_range=goal_range,
                                num_goal_steps=num_goal_steps, reward_type=reward_type, **kwargs)

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        low = np.full((2,), -float('inf'), dtype=np.float32)
        high = np.full((2,), float('inf'), dtype=np.float32)
        return akro.concat(self.observation_space, akro.Box(low=low, high=high, dtype=self.observation_space.dtype))

    def reset_model(self):
        self.cur_goal = np.random.uniform(-self.goal_range, self.goal_range, (2,))
        self.num_steps = 0
        self.goal_idx = 1
        self.goal_success = {
            'goal_1': 0,
            'goal_2': 0,
            'goal_3': 0,
            'goal_4': 0,
        }
        self.goal_staying = {
            'goal_1': 0,
            'goal_2': 0,
            'goal_3': 0,
            'goal_4': 0,
        }

        return super().reset_model()

    def _get_obs(self):
        obs = super()._get_obs()

        if not self.zero_shot:

            obs = np.concatenate([obs, self.cur_goal])

        return obs

    def step(self, *args, **kwargs):
        ob, reward, done, info = super().step(*args, **kwargs)
        for k in self.goal_success:
            info[k] = self.goal_success[k]
            info[f'{k}_staying'] = self.goal_staying[k]

        return ob, reward, done, info

    def _get_done(self):
        return self.num_steps == self.max_path_length

    def compute_reward(self, xposbefore, yposbefore, xposafter, yposafter):
        self.num_steps += 1
        delta = np.linalg.norm(self.cur_goal - np.array([xposafter, yposafter]))
        if self.reward_type == 'sparse':
            if self.num_steps % self.num_goal_steps == 0:
                reward = -delta
            else:
                reward = 0.
        elif self.reward_type == 'esparse':
            if self.num_steps != 1 and delta <= self.goal_epsilon:

                reward = 1.0
                self.goal_success[f'goal_{self.goal_idx}'] = 1
                self.goal_staying[f'goal_{self.goal_idx}'] += 1

            else:
                reward = -0.
        elif self.reward_type == 'ddense':
            delta_before = np.linalg.norm(self.cur_goal - np.array([xposbefore, yposbefore]))
            reward = delta_before - delta
        elif self.reward_type == 'dense':
            reward = -delta / self.max_path_length
        elif self.reward_type == 'motion':
            forward_reward = (xposafter - xposbefore) / self.dt
            sideward_reward = (yposafter - yposbefore) / self.dt

            survive_reward = 1.0

            reward = np.max(np.abs(np.array([forward_reward, sideward_reward
                                             ]))) + survive_reward

        safe_flag = float(loc_is_safe(self.pref_task, xposafter, yposafter))
        final_reward = reward - UNSAFE_PENALTY * (1 - safe_flag)

        return final_reward
