
import akro
import gym
import numpy as np
from .maze_env import ComplexMazeEnv


class WrappedComplexMazeEnv(gym.Env):
    def __init__(self,
                 n=None,
                 maze_config='',
                 ):
        maze_type = None
        use_antigoal = False
        ddiff = False
        ignore_reset_start = False
        done_on_success = False

        random = False
        num_skills = None
        train_random = False

        maze_config_list = maze_config.split('-')
        maze_type = maze_config_list[0]

        self.maze_env = ComplexMazeEnv(
            n=n,
            maze_type=maze_type,
            use_antigoal=use_antigoal,
            ddiff=ddiff,
            ignore_reset_start=ignore_reset_start,
            done_on_success=done_on_success,
            random=random,
            num_skills=num_skills,
            train_random=train_random,
        )

        self.action_range = self.maze_env.action_range
        self.observation_space = akro.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = akro.Box(low=-self.action_range, high=self.action_range, shape=(2,))

    @property
    def _cur_step(self):
        return self.maze_env._state['n']

    @property
    def state(self):
        return self.maze_env._state['state']

    def reset(self):
        self.maze_env.reset()
        return self.maze_env._state['s0']

    def step(self, action, render=False):
        self.maze_env.step(action)
        state = self.maze_env._state['state']
        done = self.maze_env._state['done']
        reward = self.maze_env.reward.item()
        return state, reward, done, {
            'coordinates': self.maze_env._state['prev_state'],
            'next_coordinates': self.maze_env._state['state'],
            'ori_obs': self.maze_env._state['prev_state'],
            'next_ori_obs': self.maze_env._state['state'],
        }

    def _plot_walls(self, ax):
        for x, y in self.maze_env.maze._walls:
            ax.plot(x, y, 'k-')

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        rmin, rmax = None, None

        self._plot_walls(ax)
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

            if rmin is None or rmin > np.min(trajectory[:, :2]):
                rmin = np.min(trajectory[:, :2])
            if rmax is None or rmax < np.max(trajectory[:, :2]):
                rmax = np.max(trajectory[:, :2])

        if plot_axis == 'nowalls':
            rcenter = (rmax + rmin) / 2.0
            rmax = rcenter + (rmax - rcenter) * 1.2
            rmin = rcenter + (rmin - rcenter) * 1.2
            plot_axis = [rmin, rmax, rmin, rmax]

        if plot_axis is not None:
            ax.axis(plot_axis)
        else:
            ax.axis('scaled')

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            if trajectory['env_infos']['coordinates'].ndim == 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'],
                    [trajectory['env_infos']['next_coordinates'][-1]]
                ]))
            elif trajectory['env_infos']['coordinates'].ndim > 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'].reshape(-1, 2),
                    trajectory['env_infos']['next_coordinates'].reshape(-1, 2)[-1:]
                ]))

        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        return {}

    def render(self, mode):
        return
