import akro
import gym
import numpy as np
from safety_gym.envs.engine import Engine
from envs.utils import convert_gym_spaces_to_akro

CAMERA_ID = 0
DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 64


class SafetyGymWrapper(gym.Wrapper):

    def __init__(self, env: Engine):
        super().__init__(env)
        self.env: Engine
        self.max_path_length = env.config['num_steps']
        self.action_space, self.observation_space = convert_gym_spaces_to_akro(env)
        self.last_obs = None
        self.obs = None
        self.last_ori_obs = None
        self.ori_obs = None

    def reset(self):
        self.last_ori_obs = self.ori_obs
        self.last_obs = self.obs
        self.obs = self.env.reset()
        constraint_info = self._get_constraint_info()
        self.ori_obs = np.concatenate([self.obs, constraint_info], axis=-1)

        if self.last_obs is None:
            self.last_obs = self.obs
        if self.last_ori_obs is None:
            self.last_ori_obs = self.ori_obs
        return self.obs

    def step(self, action, render=False):
        self.last_ori_obs = self.ori_obs
        self.last_obs = self.obs
        self.obs, reward, done, info = self.env.step(action=action, render=False)
        constraint_info = self._get_constraint_info()
        self.ori_obs = np.concatenate([self.obs, constraint_info], axis=-1)

        if render:
            frame = self.env.render(mode='rgb_array', camera_id=0, width=DEFAULT_WIDTH,
                                    height=DEFAULT_HEIGHT)
            info['render'] = frame.transpose(2, 0, 1)

        info['ori_obs'] = self.last_ori_obs
        info['next_ori_obs'] = self.ori_obs

        if self.env.steps >= self.env.num_steps:
            real_done = True
        else:
            real_done = False
        return self.obs, reward, real_done, info

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        return self.env.plot_trajectories(trajectories, colors, plot_axis, ax)

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        return self.env.render_trajectories(trajectories, colors, plot_axis, ax)

    def calc_eval_metrics(self, trajectories, is_option_trajectories, coord_dims=None):

        eval_metrics = {}

        if coord_dims is not None:
            coords = []
            for traj in trajectories:
                traj1 = traj['env_infos']['coordinates'][:, coord_dims]
                traj2 = traj['env_infos']['next_coordinates'][-1:, coord_dims]
                coords.append(traj1)
                coords.append(traj2)
            coords = np.concatenate(coords, axis=0)
            uniq_coords = np.unique(np.floor(coords), axis=0)
            eval_metrics.update({
                'MjNumTrajs': len(trajectories),
                'MjAvgTrajLen': len(coords) / len(trajectories) - 1,
                'MjNumCoords': len(coords),
                'MjNumUniqueCoords': len(uniq_coords),
            })

        return eval_metrics

    def render(self, mode, width, height):
        return self.env.render(mode=mode, camera_id=0, width=width, height=height)

    def _get_constraint_info(self):

        cost = {}
        cost['cost_hazards'] = 0
        for h_pos in self.env.hazards_pos:
            h_dist = self.env.dist_xy(h_pos)
            if h_dist <= self.env.hazards_size:
                cost['cost_hazards'] += (self.hazards_size - h_dist)

        cost['cost_vases_displace'] = 0
        for i in range(self.env.vases_num):
            name = f'vase{i}'
            dist = np.sqrt(np.sum(np.square(self.env.data.get_body_xpos(name)[:2] - self.env.reset_layout[name])))
            if dist > self.env.vases_displace_threshold:
                cost['cost_vases_displace'] += dist

        cost['cost_vases_velocity'] = 0
        for i in range(self.env.vases_num):
            name = f'vase{i}'
            vel = np.sqrt(np.sum(np.square(self.data.get_body_xvelp(name))))
            if vel >= self.env.vases_velocity_threshold:
                cost['cost_vases_velocity'] += vel
        constraint_info = [
            cost['cost_hazards'],
            cost['cost_vases_displace'],
            cost['cost_vases_velocity'],
        ]
        return constraint_info


class PixelSafetyGymWrapper(SafetyGymWrapper):
    def __init__(self, env: Engine):
        super().__init__(env)

        self.ob_info = dict(
            type='pixel',
            pixel_shape=(64, 64, 3),
        )

    def reset(self):
        self.last_ori_obs = self.ori_obs
        raw_ori_obs = self.env.reset()
        constraint_info = self._get_constraint_info()
        self.ori_obs = np.concatenate([raw_ori_obs, constraint_info], axis=-1)

        self.last_obs = self.obs
        render_img = self.env.render(mode='rgb_array', camera_id=0, width=DEFAULT_WIDTH,
                                     height=DEFAULT_HEIGHT).transpose(2, 0, 1)
        self.obs = render_img.flatten()
        if self.last_obs is None:
            self.last_obs = self.obs
        if self.last_ori_obs is None:
            self.last_ori_obs = self.ori_obs
        return self.obs

    def step(self, action, render=False):
        self.last_ori_obs = self.ori_obs
        self.last_obs = self.obs
        raw_ori_obs, reward, done, info = self.env.step(action=action, render=False)
        render_img = self.env.render(mode='rgb_array', camera_id=0, width=DEFAULT_WIDTH,
                                     height=DEFAULT_HEIGHT).transpose(2, 0, 1)
        self.obs = render_img.flatten()
        info['render'] = render_img

        constraint_info = self._get_constraint_info()
        self.ori_obs = np.concatenate([raw_ori_obs, constraint_info], axis=-1)

        info['ori_obs'] = self.last_ori_obs
        info['next_ori_obs'] = self.ori_obs

        if self.env.steps >= self.env.num_steps:
            real_done = True
        else:
            real_done = False
        return self.obs, reward, real_done, info
