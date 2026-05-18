import math
import pathlib
from math import ceil
from typing import Dict, List

import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from torch.nn import functional as F

from iod.metra import METRA
from iod.utils import FigManager
from pref.query_pref import QueryTrajPref
from pref.state_score_model_gpu import get_state_score_model_class
from pref.utils import save_queries_pickle

COLOR_MAP = {
    -1: [1, 0, 0, 0.8],
    0: [0, 1, 0, 0.8],
    1: [0, 0, 1, 0.8],
}
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    'custom', [((k + 1.) / 2, v[:-1] + [1.]) for k, v in COLOR_MAP.items()])


class MetraPrefQuery(METRA):
    """
    metra + pref (query pref)
    """

    def __init__(
            self,
            pb_capacity,
            labeled_state_capacity,
            query_warmup,
            query_freq,
            query_limit,
            query_batchsize,
            query_segmentlen,
            discriminator_batchsize,
            score_model_name,
            weight_func,
            weight_softmax_temp,
            n_sample_times_for_distance,
            use_phi_cache,

            query_method,
            query_large_batch_rate,
            query_state_entropy_batch_size,
            weight_smooth_decay_speed,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.pref_model = QueryTrajPref(kwargs["env_name"], self.pref_task)
        self.state_score_model = get_state_score_model_class(score_model_name)(
            traj_encoder=self.state_traj_encoder,
            device=self.device,
            pref_model=self.pref_model,
            query_segmentlen=query_segmentlen,
            query_method=query_method,
            query_large_batch_rate=query_large_batch_rate,
            query_state_entropy_batch_size=query_state_entropy_batch_size,
            capacity=pb_capacity,
            labeled_state_capacity=labeled_state_capacity,
            discriminator_batchsize=discriminator_batchsize,
            weight_func=weight_func,
            weight_softmax_temp=weight_softmax_temp,
            n_sample_times_for_distance=n_sample_times_for_distance,
            use_phi_cache=use_phi_cache,
            )
        self.query_limit = query_limit

        self.query_warmup = query_warmup
        self.query_freq = query_freq
        if self.query_warmup <= 0:
            self.query_warmup = self.query_freq

        self.query_number = 0
        self.query_counter = 0
        self.query_batchsize = query_batchsize
        self.discriminator_batchsize = discriminator_batchsize

        self.base_visualize_area = None

        total_n_queries = ceil(self.query_limit / self.query_batchsize)
        sample_weight_times = total_n_queries * self._trans_optimization_epochs * self.query_freq * 2

        self.weight_smooth_decay_speed = weight_smooth_decay_speed
        if self.weight_smooth_decay_speed > 0:
            self.weight_smooth_rate = 1
            self.weight_smooth_decay_val = (self.weight_smooth_rate / sample_weight_times) * \
                self.weight_smooth_decay_speed
        else:
            self.weight_smooth_rate = 0

    def weight_smooth_decay(self):
        if self.weight_smooth_decay_speed <= 0 or self.query_number == 0:
            return
        self.weight_smooth_rate = max(self.weight_smooth_rate - self.weight_smooth_decay_val, 0)

    @torch.no_grad()
    def state_traj_encoder(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        cur_z = self.traj_encoder(obs).mean
        cur_z_np = cur_z
        return cur_z_np

    def _train_once_inner(self, path_data: Dict[str, List[np.ndarray]]) -> Dict:

        self._update_replay_buffer(path_data)
        self.state_score_model.add_data(path_data)

        if self.query_number < self.query_limit:
            if self.query_number == 0 and self.query_counter >= self.query_warmup:
                do_query = True
            elif self.query_number > 0 and (self.query_counter - self.query_warmup) % self.query_freq == 0:
                do_query = True
            else:
                do_query = False
            self.query_counter += 1

            if do_query:
                if self.query_batchsize + self.query_number > self.query_limit:
                    query_batchsize = self.query_limit - self.query_number
                else:
                    query_batchsize = self.query_batchsize

                cur_trajs, cur_labels = self.state_score_model.sample_and_label(self.query_batchsize)
                self.query_number += query_batchsize
                print(f"query {query_batchsize} data, total {self.query_number} / {self.query_limit}")
                self._plot_query_informations(cur_trajs, cur_labels)

        epoch_data = self._flatten_data(path_data)

        tensors = self._train_components(epoch_data)

        self.state_score_model.print_cache_rate(clear_stats=True)
        print(f"weight_smooth_rate: {self.weight_smooth_rate}")

        return tensors

    def _plot_query_informations(self, cur_trajs, cur_labels):

        cur_colors = np.array([COLOR_MAP[val] for val in cur_labels.astype(int)])

        cur_trajs_loc = cur_trajs[:, :, :2]

        with FigManager(self.runner, 'Selected_Queries') as fm:
            self.runner._env.plot_trajectories(
                cur_trajs_loc, cur_colors, self.eval_plot_axis, fm.ax
            )

        good_states, bad_states, neutral_states, \
            good_infos, bad_infos, neutral_infos = self.state_score_model.get_last_labeled_states()
        save_queries_pickle(
            pathlib.Path(self.runner._snapshotter.snapshot_dir) / 'query' / f'query_{self.query_counter}.pkl',
            good_states, bad_states, neutral_states,
            good_infos, bad_infos, neutral_infos
        )

        good_states_loc = good_infos[:, :2]
        bad_states_loc = bad_infos[:, :2]
        neutral_states_loc = neutral_infos[:, :2]
        norm = Normalize(vmin=-1, vmax=1)

        with FigManager(self.runner, 'Selected_States') as fm:
            self.runner._env.plot_trajectories(
                cur_trajs_loc, cur_colors, self.eval_plot_axis, fm.ax
            )
            if good_states.size > 0:
                fm.ax.scatter(good_states_loc[:, 0], good_states_loc[:, 1],
                              c=np.zeros_like(good_states_loc[:, 0]) + 1,
                              cmap=CUSTOM_CMAP,
                              s=2., zorder=1,
                              norm=norm,)
            if bad_states.size > 0:
                fm.ax.scatter(bad_states_loc[:, 0], bad_states_loc[:, 1],
                              c=np.zeros_like(bad_states_loc[:, 0]) - 1,
                              cmap=CUSTOM_CMAP,
                              s=2., zorder=1,
                              norm=norm,)
            if neutral_states.size > 0:
                fm.ax.scatter(neutral_states_loc[:, 0], neutral_states_loc[:, 1],
                              c=np.zeros_like(neutral_states_loc[:, 0]),
                              cmap=CUSTOM_CMAP,
                              s=2,
                              norm=norm,)

        good_states, bad_states, neutral_states, \
            good_infos, bad_infos, neutral_infos = self.state_score_model.get_all_labeled_states()

        good_states_loc = good_infos[:, :2]
        bad_states_loc = bad_infos[:, :2]
        neutral_states_loc = neutral_infos[:, :2]

        with FigManager(self.runner, 'All_Selected_States') as fm:
            if good_states.size > 0:
                fm.ax.scatter(good_states_loc[:, 0], good_states_loc[:, 1],
                              c=np.zeros_like(good_states_loc[:, 0]) + 1,
                              cmap=CUSTOM_CMAP,
                              s=2.,
                              norm=norm,)
            if bad_states.size > 0:
                fm.ax.scatter(bad_states_loc[:, 0], bad_states_loc[:, 1],
                              c=np.zeros_like(bad_states_loc[:, 0]) - 1,
                              cmap=CUSTOM_CMAP,
                              s=2,
                              norm=norm,)
            if neutral_states.size > 0:
                fm.ax.scatter(neutral_states_loc[:, 0], neutral_states_loc[:, 1],
                              c=np.zeros_like(neutral_states_loc[:, 0]),
                              cmap=CUSTOM_CMAP,
                              s=2,
                              norm=norm,)

    def _update_replay_buffer(self, data: Dict[str, List[np.ndarray]]) -> None:
        """Update the replay buffer with newly collected data.

        Args:
            data (Dict[str, List[np.ndarray]]): data to add to replay buffer
        """
        super()._update_replay_buffer(data)

    def _optimize_te(self, train_store: Dict, mini_batch: Dict) -> None:
        ret = super()._optimize_te(train_store, mini_batch)

        self.state_score_model.update_cache()

        return ret

    def _update_rewards(self, train_store: Dict, mini_batch: Dict) -> None:
        """Compute the rewards for the current mini batch using the learned representations.

        Args:
            train_store (Dict): training store
            mini_batch (Dict): mini batch data
        """
        obs = mini_batch['obs']
        next_obs = mini_batch['next_obs']
        raw_weight = self.state_score_model.get_weight(obs, next_obs)

        raw_weight = torch.from_numpy(raw_weight).to(self.device).to(torch.float32)
        weight = raw_weight * (1 - self.weight_smooth_rate) + self.weight_smooth_rate
        mini_batch['next_obs_pref'] = weight
        self.weight_smooth_decay()

        if self.inner:
            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean

            target_z = next_z - cur_z

            if self.no_diff_in_rep:
                target_z = cur_z

            if self.self_normalizing:
                target_z = target_z / target_z.norm(dim=-1, keepdim=True)

            if self.log_sum_exp:
                if self.sample_new_z:
                    new_z = torch.randn(self.num_negative_z, self.dim_option, device=mini_batch['options'].device)
                    if self.unit_length:
                        new_z /= torch.norm(new_z, dim=-1, keepdim=True)
                    pairwise_scores = target_z @ new_z.t()
                else:
                    pairwise_scores = target_z @ mini_batch['options'].t()
                log_sum_exp = torch.logsumexp(pairwise_scores, dim=-1)

            next_obs_pref = mini_batch["next_obs_pref"]
            if self.discrete:
                masks = (mini_batch['options'] - mini_batch['options'].mean(dim=1, keepdim=True)) * \
                    self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                rewards = (target_z * masks * self.pref_coef).sum(dim=1) * next_obs_pref
            else:
                inner = (target_z * mini_batch['options'] * self.pref_coef).sum(dim=1) * next_obs_pref
                rewards = inner

            mini_batch.update({
                'cur_z': cur_z,
                'next_z': next_z,
            })

        elif self.metra_mlp_rep:

            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean
            mini_batch.update({
                'cur_z': cur_z,
                'next_z': next_z,
            })

            rep = self.f_encoder(obs, next_obs)
            rewards = (rep * mini_batch['options']).sum(dim=1)

            if self.log_sum_exp:
                if self.sample_new_z:
                    new_z = torch.randn(self.num_negative_z, self.dim_option, device=mini_batch['options'].device)
                    if self.unit_length:
                        new_z /= torch.norm(new_z, dim=-1, keepdim=True)
                    pairwise_scores = rep @ new_z.t()
                else:
                    pairwise_scores = rep @ mini_batch['options'].t()
                log_sum_exp = torch.logsumexp(pairwise_scores, dim=-1)

        else:
            target_dists = self.traj_encoder(next_obs)

            if self.discrete:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(logits,
                                                             mini_batch['options'].argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(mini_batch['options'])

            if self.diayn_include_baseline:
                rewards -= torch.log(torch.tensor(1 / self.dim_option))

        train_store.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })

        mini_batch['rewards'] = rewards
        if self.log_sum_exp:
            mini_batch['log_sum_exp'] = log_sum_exp
